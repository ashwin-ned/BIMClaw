"""FastAPI backend for BIMClaw.

Endpoints:
    POST /api/upload                   Upload IFC → {model_id, metadata}
    POST /api/render                   Trigger floor plan render (background)
    GET  /api/render/{model_id}        Poll render status
    GET  /api/model/{model_id}         Stream raw IFC file for browser
    GET  /api/images/{model_id}/floor_plan   BEV floor plan PNG
    POST /api/query                    Ask a spatial question
    GET  /api/health                   Health check
"""

from __future__ import annotations

import logging
import logging.config
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from bim_frontend.api.bim_session import get_store

# ---------------------------------------------------------------------------
# Logging: structured, no ANSI, stdout so Docker captures it
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {
                "format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            }
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "plain",
            }
        },
        "root": {"level": "INFO", "handlers": ["stdout"]},
        "loggers": {
            "uvicorn": {"propagate": True},
            "uvicorn.access": {"propagate": True},
            "bim_agent": {"level": "DEBUG", "propagate": True},
            "bim_modalities": {"level": "DEBUG", "propagate": True},
        },
    })


_configure_logging()
logger = logging.getLogger(__name__)

_UPLOAD_BASE    = Path(os.environ.get("BIMCLAW_UPLOAD_DIR",    "/app/data/uploads"))
_MODALITIES_BASE = Path(os.environ.get("BIMCLAW_MODALITIES_DIR", "/app/data/modalities"))

_UPLOAD_BASE.mkdir(parents=True, exist_ok=True)
_MODALITIES_BASE.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="BIMClaw API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[mGKHF]", "", str(text))


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_ifc(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.lower().endswith(".ifc"):
        raise HTTPException(400, "Only .ifc files are accepted")

    store = get_store()
    safe_name = file.filename.replace(" ", "_")
    upload_dir = _UPLOAD_BASE / safe_name.replace(".ifc", "")
    upload_dir.mkdir(parents=True, exist_ok=True)
    ifc_path = str(upload_dir / safe_name)

    content = await file.read()
    with open(ifc_path, "wb") as f:
        f.write(content)
    logger.info("[upload] Saved %d bytes → %s", len(content), ifc_path)

    try:
        from bim_agent.kernel_types.bim_metadata import BIMMetadata
        meta = BIMMetadata.from_ifc(ifc_path)
        meta_dict = {
            "schema": meta.schema,
            "storey_names": meta.storey_names,
            "space_names": meta.space_names[:50],
            "element_counts": meta.element_counts,
            "has_spaces": meta.has_spaces,
            "has_materials": meta.has_materials,
            "has_quantities": meta.has_quantities,
        }
        logger.info("[upload] BIMMetadata: %s", meta.to_text().splitlines()[0])
    except Exception as exc:
        logger.warning("[upload] BIMMetadata extraction failed: %s", exc)
        meta_dict = {"error": _strip_ansi(str(exc))}

    session = store.create(ifc_path, upload_dir, _MODALITIES_BASE)
    logger.info("[upload] model_id=%s", session.model_id)
    return {"model_id": session.model_id, "metadata": meta_dict}


# ---------------------------------------------------------------------------
# Render (native Python floor plan — no Blender)
# ---------------------------------------------------------------------------

class RenderRequest(BaseModel):
    model_id: str
    frames: int = 64        # ignored — kept for API compat
    num_key_frames: int = 16


@app.post("/api/render")
async def trigger_render(req: RenderRequest, background_tasks: BackgroundTasks) -> Dict:
    store = get_store()
    session = store.get(req.model_id)
    if session is None:
        raise HTTPException(404, "model_id not found")
    if session.rendering:
        return {"status": "already_rendering", "model_id": req.model_id}

    store.update(req.model_id, rendering=True, modalities_ready=False)
    background_tasks.add_task(_render_background, req.model_id)
    return {"status": "started", "model_id": req.model_id}


@app.get("/api/render/{model_id}")
async def render_status(model_id: str) -> Dict:
    session = get_store().get(model_id)
    if session is None:
        raise HTTPException(404, "model_id not found")
    return {
        "model_id": model_id,
        "rendering": session.rendering,
        "ready": session.modalities_ready,
        "error": session.modalities_error,
    }


def _render_background(model_id: str) -> None:
    store = get_store()
    session = store.get(model_id)
    if session is None:
        return
    logger.info("[render] Starting floor plan render for model_id=%s", model_id)
    try:
        from bim_modalities.pipeline import BIMModalitiesPipeline, BIMModalitiesConfig
        pipeline = BIMModalitiesPipeline(config=BIMModalitiesConfig())
        out = pipeline.extract(Path(session.ifc_path), session.modalities_dir)
        store.update(
            model_id,
            modalities_output=out,
            modalities_ready=True,
            rendering=False,
        )
        logger.info("[render] Done for model_id=%s", model_id)
    except Exception as exc:
        err = _strip_ansi(str(exc))
        logger.error("[render] Failed model_id=%s: %s", model_id, err, exc_info=True)
        store.update(model_id, modalities_error=err, modalities_ready=False, rendering=False)


# ---------------------------------------------------------------------------
# IFC file stream
# ---------------------------------------------------------------------------

@app.get("/api/model/{model_id}")
async def stream_ifc(model_id: str) -> FileResponse:
    session = get_store().get(model_id)
    if session is None:
        raise HTTPException(404, "model_id not found")
    return FileResponse(
        session.ifc_path,
        media_type="application/octet-stream",
        filename=Path(session.ifc_path).name,
    )


# ---------------------------------------------------------------------------
# Floor plan image
# ---------------------------------------------------------------------------

@app.get("/api/images/{model_id}/floor_plan")
async def floor_plan_image(model_id: str) -> FileResponse:
    session = get_store().get(model_id)
    if session is None:
        raise HTTPException(404, "model_id not found")
    if not session.modalities_ready:
        raise HTTPException(425, "Render not yet complete")
    img_path = session.modalities_dir / "floor_plan_bev.png"
    if not img_path.exists():
        raise HTTPException(404, "Floor plan image not found")
    return FileResponse(str(img_path), media_type="image/png")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    model_id: str
    question: str


@app.post("/api/query")
async def query(req: QueryRequest) -> Dict[str, Any]:
    store = get_store()
    session = store.get(req.model_id)
    if session is None:
        raise HTTPException(404, "model_id not found")

    logger.info("[query] model_id=%s question=%r", req.model_id, req.question[:80])

    from bim_agent.config import BIMAgentConfig
    from bim_agent.workflow import BIMAgentWorkflow

    cfg = BIMAgentConfig(use_rendered_images=session.modalities_ready)
    workflow = BIMAgentWorkflow(config=cfg)

    try:
        result = await workflow.arun(
            ifc_path=session.ifc_path,
            question=req.question,
            modalities_out_dir=str(session.modalities_dir),
        )
        logger.info(
            "[query] Done model_id=%s steps=%d answer=%r",
            req.model_id, result["steps"], str(result["answer"])[:80],
        )
        return {
            "answer": _strip_ansi(str(result["answer"])),
            "session_id": result["session_id"],
            "steps": result["steps"],
            "termination_reason": result["termination_reason"],
        }
    except Exception as exc:
        err = _strip_ansi(str(exc))
        logger.error("[query] Failed model_id=%s: %s", req.model_id, err, exc_info=True)
        raise HTTPException(500, f"Agent failed: {err}")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> Dict:
    return {"status": "ok", "upload_dir": str(_UPLOAD_BASE)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
