"""Native Python IFC floor plan renderer — no Blender required.

Uses ifcopenshell.geom to extract 3D geometry and projects it top-down.
Replaces the BIM-Walker PLY→BEV route.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Render order matters: spaces first (fill), then structure on top
_CLASSES = [
    ("IfcSpace",            "#e8f4e8", "none",    0.5, 0.35),
    ("IfcSlab",             "#cccccc", "#888888", 0.8, 0.6),
    ("IfcWall",             "#2d2d2d", "#111111", 1.0, 0.9),
    ("IfcWallStandardCase", "#2d2d2d", "#111111", 1.0, 0.9),
    ("IfcCurtainWall",      "#aaddff", "#5599cc", 0.6, 0.7),
    ("IfcColumn",           "#444444", "#222222", 1.0, 0.9),
    ("IfcStair",            "#999999", "#666666", 0.8, 0.7),
    ("IfcRailing",          "#666666", "#444444", 0.5, 0.7),
    ("IfcDoor",             "#c8864a", "#8B4513", 0.8, 0.8),
    ("IfcWindow",           "#87CEEB", "#4499aa", 0.5, 0.7),
    ("IfcFurnishingElement","#ddc89a", "#aa9960", 0.4, 0.5),
]


def render_ifc_floor_plan(
    ifc_path: str,
    out_path: Optional[str] = None,
    resolution: int = 1024,
    padding_m: float = 1.5,
) -> "PIL.Image.Image":
    """Render a top-down 2D floor plan for an IFC file.

    Parameters
    ----------
    ifc_path : path to IFC file
    out_path : optional; if given, PNG is saved here
    resolution : output image size in pixels (square)
    padding_m : margin around the model in metres

    Returns
    -------
    PIL.Image.Image in RGB mode
    """
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    try:
        import ifcopenshell
        import ifcopenshell.geom
    except ImportError:
        logger.error("[IFCFloorPlan] ifcopenshell not available")
        return _blank(resolution, out_path)

    logger.info("[IFCFloorPlan] Opening %s", ifc_path)
    try:
        model = ifcopenshell.open(str(ifc_path))
    except Exception as exc:
        logger.error("[IFCFloorPlan] Cannot open IFC: %s", exc)
        return _blank(resolution, out_path)

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    # Collect per-class bounding-box polygons
    polys_by_class: dict[str, list[np.ndarray]] = {c: [] for c, *_ in _CLASSES}
    all_x: list[float] = []
    all_y: list[float] = []

    for ifc_class, *_ in _CLASSES:
        if not model.by_type(ifc_class):
            continue
        try:
            it = ifcopenshell.geom.iterator(settings, model, include=[ifc_class])
            if not it.initialize():
                continue
            while True:
                shape = it.get()
                verts = np.array(shape.geometry.verts, dtype=np.float32).reshape(-1, 3)
                xy = verts[:, :2]
                if len(xy) >= 3:
                    bbox = _bbox_quad(xy)
                    polys_by_class[ifc_class].append(bbox)
                    all_x.extend(bbox[:, 0])
                    all_y.extend(bbox[:, 1])
                if not it.next():
                    break
        except Exception as exc:
            logger.warning("[IFCFloorPlan] %s: %s", ifc_class, exc)

    if not all_x:
        logger.warning("[IFCFloorPlan] No geometry found, returning blank.")
        return _blank(resolution, out_path)

    x_min = min(all_x) - padding_m
    x_max = max(all_x) + padding_m
    y_min = min(all_y) - padding_m
    y_max = max(all_y) + padding_m

    dpi = 100
    px = resolution / dpi
    fig, ax = plt.subplots(figsize=(px, px), dpi=dpi, facecolor="white")
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis("off")

    for ifc_class, facecolor, edgecolor, lw, alpha in _CLASSES:
        polys = polys_by_class.get(ifc_class, [])
        if not polys:
            continue
        coll = PolyCollection(
            polys, facecolor=facecolor, edgecolor=edgecolor,
            linewidth=lw, alpha=alpha, closed=True,
        )
        ax.add_collection(coll)

    _scale_bar(ax, x_min, x_max, y_min, y_max)

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    if out_path:
        img.save(out_path)
    logger.info("[IFCFloorPlan] Done: %dx%d", resolution, resolution)
    return img


def _bbox_quad(xy: np.ndarray) -> np.ndarray:
    """Return the 4-corner axis-aligned bounding box of an XY point cloud."""
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def _scale_bar(ax, x_min, x_max, y_min, y_max) -> None:
    try:
        span = x_max - x_min
        scale_m = max(1, round(span / 5))
        bx = x_min + span * 0.05
        by = y_min + (y_max - y_min) * 0.03
        ax.plot([bx, bx + scale_m], [by, by], "k-", linewidth=3, solid_capstyle="butt")
        ax.text(bx + scale_m / 2, by + (y_max - y_min) * 0.015,
                f"{scale_m} m", ha="center", va="bottom", fontsize=8)
    except Exception:
        pass


def _blank(resolution: int, out_path: Optional[str]) -> "PIL.Image.Image":
    from PIL import Image
    img = Image.new("RGB", (resolution, resolution), (245, 245, 245))
    if out_path:
        img.save(out_path)
    return img
