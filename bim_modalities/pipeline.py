"""BIMModalitiesPipeline: native Python IFC floor plan, no Blender required.

Replaces the BIM-Walker subprocess route. The floor plan is rendered
directly from IFC geometry using ifcopenshell.geom + matplotlib.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class BIMModalitiesConfig:
    bev_resolution: int = 1024
    num_key_frames: int = 0   # no walkthrough frames without Blender
    render_frames: int = 0
    blender: str = ""
    conda_env: str = ""
    walker_timeout_min: int = 0


@dataclass
class BIMModalitiesOutput:
    ifc_path: Path
    out_dir: Path
    floor_plan_image: Image.Image
    key_rgb_paths: List[Path]
    reconstruction: None = None
    walker_output: None = None   # kept for API compat, always None

    @property
    def key_rgb_images(self) -> List[Image.Image]:
        return [Image.open(p).convert("RGB") for p in self.key_rgb_paths]

    def summary(self) -> dict:
        return {
            "ifc_path": str(self.ifc_path),
            "out_dir": str(self.out_dir),
            "num_key_frames": len(self.key_rgb_paths),
            "has_reconstruction": False,
            "has_floor_plan": self.floor_plan_image is not None,
        }


class BIMModalitiesPipeline:
    """Renders an IFC floor plan using native Python (ifcopenshell.geom)."""

    def __init__(
        self,
        config: Optional[BIMModalitiesConfig] = None,
        walker_dir: Optional[Path] = None,   # ignored — kept for API compat
    ):
        self.config = config or BIMModalitiesConfig()

    def extract(self, ifc_path: Path, out_dir: Path) -> BIMModalitiesOutput:
        """Render floor plan for ``ifc_path`` and save to ``out_dir``.

        Cached: if ``floor_plan_bev.png`` already exists the render is skipped.
        """
        from bim_modalities.ifc_floor_plan import render_ifc_floor_plan

        ifc_path = Path(ifc_path).resolve()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        floor_plan_path = out_dir / "floor_plan_bev.png"

        if floor_plan_path.exists():
            logger.info("[BIMModalities] Using cached floor plan: %s", floor_plan_path)
            floor_plan = Image.open(floor_plan_path).convert("RGB")
        else:
            logger.info("[BIMModalities] Rendering floor plan for %s", ifc_path.name)
            try:
                floor_plan = render_ifc_floor_plan(
                    ifc_path=str(ifc_path),
                    out_path=str(floor_plan_path),
                    resolution=self.config.bev_resolution,
                )
            except Exception as exc:
                logger.error("[BIMModalities] Floor plan render failed: %s", exc, exc_info=True)
                floor_plan = Image.new("RGB", (self.config.bev_resolution,) * 2, (245, 245, 245))

        # Write cache sentinel
        summary_path = out_dir / "render_summary.json"
        if not summary_path.exists():
            summary_path.write_text(json.dumps({
                "ifc": str(ifc_path),
                "floor_plan": str(floor_plan_path),
                "renderer": "ifc_floor_plan_native",
            }))

        return BIMModalitiesOutput(
            ifc_path=ifc_path,
            out_dir=out_dir,
            floor_plan_image=floor_plan,
            key_rgb_paths=[],
            reconstruction=None,
            walker_output=None,
        )
