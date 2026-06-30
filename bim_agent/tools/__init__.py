"""BIMToolsModule: assembles all tools into the kernel's ``tools`` namespace."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BIMToolsModule:
    """Namespace object injected into the kernel as ``tools``.

    Agent writes code like:
        elements = tools.IFC.query_elements("IfcDoor")
        area = tools.BIMGeometry.room_area("Living Room")
        Reconstruction.render_bev()
    """

    def __init__(
        self,
        ifc_model,
        reconstruction=None,
    ):
        from bim_agent.tools.ifc_tool import IFCTool
        from bim_agent.tools.bim_geometry_tool import BIMGeometryTool

        self.IFC = IFCTool(ifc_model)
        self.BIMGeometry = BIMGeometryTool(self.IFC, ifc_model)
        self.Reconstruction = reconstruction  # BIMReconstruction or None

        # Import lightweight reusable tools from SpatialClaw
        try:
            from bim_agent.tools.geometry_utils import GeometryUtils
            self.Geometry = GeometryUtils()
        except Exception as exc:
            logger.warning("[Tools] GeometryUtils unavailable: %s", exc)
            self.Geometry = None

        try:
            from bim_agent.tools.graph_drawer import GraphDrawer
            self.Graph = GraphDrawer()
        except Exception as exc:
            logger.warning("[Tools] GraphDrawer unavailable: %s", exc)
            self.Graph = None

        try:
            from bim_agent.tools.draw_utils import DrawUtils
            self.Draw = DrawUtils()
        except Exception as exc:
            logger.warning("[Tools] DrawUtils unavailable: %s", exc)
            self.Draw = None

        try:
            from bim_agent.tools.mask_utils import MaskUtils
            self.Mask = MaskUtils()
        except Exception as exc:
            logger.warning("[Tools] MaskUtils unavailable: %s", exc)
            self.Mask = None

    def get_prompt_descriptions(self) -> str:
        """Aggregate all tool prompt descriptions for the system prompt."""
        from bim_agent.tools.ifc_tool import IFCTool
        from bim_agent.tools.bim_geometry_tool import BIMGeometryTool
        parts = [
            IFCTool.get_prompt_description(),
            BIMGeometryTool.get_prompt_description(),
        ]
        # Add Geometry/Graph/Draw if available
        for cls_name in ("GeometryUtils", "GraphDrawer", "DrawUtils", "MaskUtils"):
            try:
                mod_map = {
                    "GeometryUtils": "bim_agent.tools.geometry_utils",
                    "GraphDrawer": "bim_agent.tools.graph_drawer",
                    "DrawUtils": "bim_agent.tools.draw_utils",
                    "MaskUtils": "bim_agent.tools.mask_utils",
                }
                import importlib
                mod = importlib.import_module(mod_map[cls_name])
                cls = getattr(mod, cls_name)
                desc = cls.get_prompt_description()
                if desc:
                    parts.append(desc)
            except Exception:
                pass
        return "\n\n".join(p for p in parts if p)

    def __repr__(self) -> str:
        available = ["IFC", "BIMGeometry"]
        if self.Reconstruction is not None:
            available.append("Reconstruction (via BIMReconstruction)")
        if self.Geometry:
            available.append("Geometry")
        if self.Graph:
            available.append("Graph")
        if self.Draw:
            available.append("Draw")
        if self.Mask:
            available.append("Mask")
        return f"BIMToolsModule(tools={available})"
