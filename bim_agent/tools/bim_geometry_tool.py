"""BIMGeometryTool: spatial computations on IFC element placements.

Injected into the Jupyter kernel as ``tools.BIMGeometry``.
All methods read from the ``IFCModel`` kernel object.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from bim_agent.tools.base import CPUTool

logger = logging.getLogger(__name__)


TOOL_PROMPT_DESCRIPTION = """\
## tools.BIMGeometry — spatial computations on IFC elements

Operates on element placements and quantities extracted from IFC.

```python
dist = tools.BIMGeometry.distance("GlobalId1", "GlobalId2")
# → float  (metres between element placement origins)

nearby = tools.BIMGeometry.elements_in_radius("GlobalId_center", radius=5.0, global_ids=door_ids)
# → list[str]  GlobalIds of elements within radius metres of center element

area = tools.BIMGeometry.room_area("Living Room")
# → float  m² (from IfcElementQuantity; falls back to bounding-box XY area)

z = tools.BIMGeometry.storey_elevation("Ground Floor")
# → float  Z elevation of the storey datum in metres

h = tools.BIMGeometry.floor_height("Ground Floor")
# → float  distance to the next storey above (metres); None if no storey above
```

Tips:
- Combine with tools.IFC.query_elements to get GlobalIds, then use BIMGeometry for distances.
- Use room_area to compare space sizes without computing from geometry.
"""


class BIMGeometryTool(CPUTool):
    """Spatial geometry tool for BIM agent — operates on IFC placement matrices."""

    TOOL_PROMPT_DESCRIPTION = TOOL_PROMPT_DESCRIPTION

    def __init__(self, ifc_tool, ifc_model):
        """
        Parameters
        ----------
        ifc_tool:
            The IFCTool instance (already constructed, has _model).
        ifc_model:
            The IFCModel kernel object.
        """
        self._ifc_tool = ifc_tool
        self._model = ifc_model.model

    def distance(self, global_id_1: str, global_id_2: str) -> Optional[float]:
        """Euclidean distance between the placement origins of two elements (metres).

        Returns None if either element's placement cannot be resolved.
        """
        T1 = self._ifc_tool.get_placement(global_id_1)
        T2 = self._ifc_tool.get_placement(global_id_2)
        if T1 is None or T2 is None:
            return None
        p1 = T1[:3, 3]
        p2 = T2[:3, 3]
        return float(np.linalg.norm(p1 - p2))

    def elements_in_radius(
        self,
        center_global_id: str,
        radius: float,
        global_ids: List[str],
    ) -> List[str]:
        """Return the subset of ``global_ids`` within ``radius`` metres of ``center_global_id``.

        Parameters
        ----------
        center_global_id:
            GlobalId of the reference element.
        radius:
            Search radius in metres.
        global_ids:
            List of GlobalIds to check.
        """
        T_center = self._ifc_tool.get_placement(center_global_id)
        if T_center is None:
            return []
        p_center = T_center[:3, 3]

        result = []
        for gid in global_ids:
            T = self._ifc_tool.get_placement(gid)
            if T is None:
                continue
            p = T[:3, 3]
            if np.linalg.norm(p - p_center) <= radius:
                result.append(gid)
        return result

    def room_area(self, space_name: str) -> Optional[float]:
        """Return the floor area of a named space in m².

        Tries in order:
          1. IfcElementQuantity (GrossFloorArea / NetFloorArea)
          2. IfcPropertySet Pset_SpaceCommon.GrossFloorArea
          3. Approximate from bounding-box XY extent
        Returns None if the space is not found.
        """
        space = self._ifc_tool._find_space_by_name(space_name)
        if space is None:
            return None

        # Try IfcElementQuantity
        for rel in self._model.by_type("IfcRelDefinesByProperties"):
            if space not in (rel.RelatedObjects or []):
                continue
            pdef = rel.RelatingPropertyDefinition
            if pdef.is_a("IfcElementQuantity"):
                for qty in pdef.Quantities or []:
                    if "area" in (qty.Name or "").lower():
                        val = getattr(qty, "AreaValue", None)
                        if val is not None:
                            return float(val)

        # Try Pset_SpaceCommon
        for rel in self._model.by_type("IfcRelDefinesByProperties"):
            if space not in (rel.RelatedObjects or []):
                continue
            pdef = rel.RelatingPropertyDefinition
            if pdef.is_a("IfcPropertySet"):
                for prop in pdef.HasProperties or []:
                    if "area" in (prop.Name or "").lower():
                        val = getattr(prop, "NominalValue", None)
                        if val is not None:
                            return float(val.wrappedValue)

        # Fallback: compute from geometry
        bbox = self._ifc_tool.get_bounding_box(space.GlobalId)
        if bbox is not None:
            w, h = bbox["size"][0], bbox["size"][1]
            return float(w * h)

        return None

    def storey_elevation(self, storey_name: str) -> Optional[float]:
        """Return the Z elevation (datum) of a named storey in metres."""
        target = storey_name.lower()
        for storey in self._model.by_type("IfcBuildingStorey"):
            name = (storey.Name or "").lower()
            if target in name or name in target:
                elev = storey.Elevation
                if elev is not None:
                    return float(elev)
        return None

    def floor_height(self, storey_name: str) -> Optional[float]:
        """Return the distance from ``storey_name`` to the next storey above (metres).

        Returns None if this is the top storey or the storey is not found.
        """
        storeys = sorted(
            [s for s in self._model.by_type("IfcBuildingStorey") if s.Elevation is not None],
            key=lambda s: float(s.Elevation),
        )

        target = storey_name.lower()
        current_elev = None
        current_idx = None
        for i, s in enumerate(storeys):
            name = (s.Name or "").lower()
            if target in name or name in target:
                current_elev = float(s.Elevation)
                current_idx = i
                break

        if current_idx is None:
            return None
        if current_idx + 1 >= len(storeys):
            return None

        next_elev = float(storeys[current_idx + 1].Elevation)
        return next_elev - current_elev
