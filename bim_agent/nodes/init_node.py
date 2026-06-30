"""init_node: starts the Jupyter kernel and injects BIM objects.

Injection strategy: generate self-contained Python code that reconstructs
all objects from scratch inside the kernel.  We do NOT use cloudpickle for
IFCModel because ifcopenshell.file is a C extension that is not picklable.

Objects available in every kernel session:
  IFCModel      - ifcopenshell wrapper for the uploaded IFC file
  BIMMetadata   - pre-extracted schema / element counts / storey names
  tools         - BIMToolsModule (tools.IFC.*, tools.BIMGeometry.*)
  InputImages   - list of PIL images [floor_plan, *key_frames]
  ReturnAnswer  - class; agent calls ReturnAnswer(value) to submit answer
  show(img)     - no-op in headless mode
  vlm           - None (no vision model without GPU)
  Reconstruction- None (no depth reconstruction without Blender)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from PIL import Image

from bim_agent.state import AgentState

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kernel boot: standard library imports available to LLM-generated cells
# ---------------------------------------------------------------------------

_INIT_CODE = """\
import sys, os
sys.path.insert(0, '/app')

import numpy as np
import math, json, re, collections, itertools, functools
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    pass
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    pass
try:
    import scipy.spatial, scipy.ndimage
except ImportError:
    pass

print("BIMClaw kernel ready")
"""


async def init_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Initialize kernel, inject BIM objects, build the first user message."""
    cfg = config["configurable"]
    km = cfg["kernel_manager"]
    tools_module = cfg["tools_module"]
    system_prompt = cfg["system_prompt"]
    input_images = cfg["input_images"]
    bim_metadata = cfg["bim_metadata"]
    key_frames = cfg.get("key_frames", [])
    ifc_path = state["ifc_path"]

    # 1. Start or reset kernel
    if km.is_running:
        try:
            await km.reset_namespace()
            kernel_id = state.get("kernel_id") or "reused"
        except Exception as exc:
            _logger.warning("[init_node] reset failed (%s), restarting", exc)
            await km.shutdown()
            kernel_id = await km.start()
    else:
        kernel_id = await km.start()

    km.set_init_code(_INIT_CODE)
    result = await km.execute(_INIT_CODE, timeout=120)
    if result.error:
        raise RuntimeError(f"Kernel init failed: {result.error}")

    # 2. Save input images to temp files so the kernel can load them
    img_tmp_paths = _save_images_to_tmp(list(input_images))

    # 3. Build and run the injection code
    injection_code = _build_injection_code(ifc_path, img_tmp_paths)
    km.set_injection_code(injection_code)
    result = await km.execute(injection_code, timeout=300)
    if result.error:
        _logger.error("[init_node] Injection stderr: %s", result.stderr)
        raise RuntimeError(f"Object injection failed: {_strip_ansi(result.error)}")

    # 4. Build initial user message (images only if model supports vision)
    agent_config = cfg.get("agent_config")
    vision_enabled = getattr(agent_config, "vision_enabled", True)
    user_content = _build_user_message(
        instruction=state["instruction"],
        key_frames=key_frames if vision_enabled else [],
        bim_metadata=bim_metadata,
    )

    return {
        "kernel_id": kernel_id,
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ],
        "variable_registry": {},
        "step_count": 0,
        "failure_count": 0,
        "total_tool_calls": 0,
        "total_show_images": 0,
    }


# ---------------------------------------------------------------------------
# Image serialisation helpers
# ---------------------------------------------------------------------------

def _save_images_to_tmp(images: List[Image.Image]) -> List[str]:
    """Write PIL images to temp JPEG files; return list of paths."""
    paths = []
    for i, img in enumerate(images):
        if not isinstance(img, Image.Image):
            continue
        fd, path = tempfile.mkstemp(suffix=f"_bimclaw_{i}.jpg")
        os.close(fd)
        img.convert("RGB").save(path, "JPEG", quality=85)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Injection code — pure Python, no cloudpickle
# ---------------------------------------------------------------------------

def _build_injection_code(ifc_path: str, img_tmp_paths: List[str]) -> str:
    """Generate kernel Python code that reconstructs all BIM objects.

    Objects are built from scratch inside the kernel using the IFC file path.
    Images are loaded from temp JPEG files written by the main process.
    """
    img_paths_repr = repr(img_tmp_paths)
    ifc_path_repr = repr(str(ifc_path))

    return f"""\
import sys, os
sys.path.insert(0, '/app')

# ── IFC objects ──────────────────────────────────────────────────────────────
from bim_agent.kernel_types.ifc_model import IFCModel as _IFCModelClass
from bim_agent.kernel_types.bim_metadata import BIMMetadata as _BIMMetaClass
from bim_agent.tools import BIMToolsModule as _BTM

_ifc_path = {ifc_path_repr}

IFCModel   = _IFCModelClass(_ifc_path)
BIMMetadata = _BIMMetaClass.from_ifc(_ifc_path)
tools       = _BTM(ifc_model=IFCModel)

# ── Input images ─────────────────────────────────────────────────────────────
from bim_agent.kernel_types.input_images import InputImages as _InputImagesClass
from PIL import Image as _PILImage

_img_paths = {img_paths_repr}
_pil_imgs  = []
for _p in _img_paths:
    try:
        _pil_imgs.append(_PILImage.open(_p).convert("RGB"))
    except Exception as _e:
        print(f"[init] Could not load image {{_p}}: {{_e}}")

InputImages = _InputImagesClass(_pil_imgs)

# ── ReturnAnswer (uses builtins sentinel — detected by feedback_node) ─────────
from bim_agent.kernel_types.return_answer import ReturnAnswer

# ── Stubs ────────────────────────────────────────────────────────────────────
def show(img_or_imgs=None, label=""):
    \"\"\"Display an image (no-op in headless Docker mode).\"\"\"
    pass

vlm           = None
Reconstruction = None

print(f"BIM kernel ready — IFCModel, BIMMetadata, tools, InputImages({{len(_pil_imgs)}}), ReturnAnswer")
"""


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def _build_user_message(instruction: str, key_frames: List, bim_metadata):
    """Build the initial user message: question + metadata summary, with optional images.

    Returns a plain string when there are no images (text-only LLMs need this);
    returns a list of content blocks when images are present (multimodal format).
    """
    text_block = (
        f"**Question:** {instruction}\n\n"
        f"**BIM Model Summary:**\n{bim_metadata.to_text()}"
    )

    if not key_frames:
        return text_block

    content = []
    for i, img in enumerate(key_frames):
        if not isinstance(img, Image.Image):
            continue
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        label = "Floor plan (BEV top-down)" if i == 0 else f"Interior view {i}"
        content.append({"type": "text", "text": f"[{label}]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    content.append({"type": "text", "text": text_block})
    return content


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from a string."""
    import re
    return re.sub(r"\x1b\[[0-9;]*[mGKHF]", "", text)
