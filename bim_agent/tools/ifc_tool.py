"""IFCTool: CPU-side tool for querying IFC structured data in the agent kernel.

Injected into the Jupyter kernel as ``tools.IFC``.  All methods read from the
``IFCModel`` object that is already present in the kernel namespace; they do
NOT open files or make network calls.

API follows the plan:
    tools.IFC.query_elements(ifc_class, **filters)
    tools.IFC.get_spatial_structure()
    tools.IFC.count_in_space(space_name, ifc_class)
    tools.IFC.get_quantities(global_id)
    tools.IFC.get_properties(global_id, pset_name=None)
    tools.IFC.get_material(global_id)
    tools.IFC.get_bounding_box(global_id)
    tools.IFC.get_placement(global_id)
    tools.IFC.search_by_name(pattern, ifc_class=None)
    tools.IFC.get_space_boundaries(space_name)
    tools.IFC.get_type_info(global_id)
    tools.IFC.get_classification(global_id)
"""

from __future__ import annotations

import fnmatch
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from bim_agent.tools.base import CPUTool

logger = logging.getLogger(__name__)


TOOL_PROMPT_DESCRIPTION = """\
## tools.IFC — IFC structured data queries

``tools.IFC`` wraps the ``IFCModel`` object already in your kernel.
Use it to query exact element data from the IFC file.

### Element queries
```python
elements = tools.IFC.query_elements("IfcDoor")
# → list of dicts: [{id, GlobalId, Name, Type, storey, space, properties}, ...]

count = tools.IFC.count_in_space("Living Room", "IfcFurnishingElement")
# → int

structure = tools.IFC.get_spatial_structure()
# → nested dict: {storey_name: {space_name: [element_dicts]}}
```

### Element detail
```python
dims = tools.IFC.get_quantities(global_id)
# → {"Length": 2.1, "Width": 0.9, "Height": 2.1, "Area": 1.89, "Volume": ...}
# Reads IfcElementQuantity (exact, as modelled)

props = tools.IFC.get_properties(global_id, pset_name="Pset_DoorCommon")
# → dict of all named properties; omit pset_name for all property sets

mat = tools.IFC.get_material(global_id)
# → "Concrete - C30/37" or {"layers": [{"name":"Concrete","thickness":0.2},...]}

bbox = tools.IFC.get_bounding_box(global_id)
# → {"min": [x,y,z], "max": [x,y,z], "size": [w,h,d]}  (metres, world coords)

T = tools.IFC.get_placement(global_id)
# → 4×4 np.ndarray (float64) camera-to-world matrix of element origin

hits = tools.IFC.search_by_name("living*", ifc_class="IfcSpace")
# → list of matching element dicts; supports glob patterns and case-insensitive regex
```

### Space and type info
```python
boundaries = tools.IFC.get_space_boundaries("Kitchen")
# → list of element dicts bounding the space (walls, slabs, windows)

type_info = tools.IFC.get_type_info(global_id)
# → dict of IfcRelDefinesByType properties (door/window type dimensions, etc.)

classification = tools.IFC.get_classification(global_id)
# → {"system": "Uniclass", "code": "Ss_20_05_25", "description": "..."}
```

### Tips
- Use ``query_elements`` as the primary way to count and list elements.
- Use ``get_quantities`` for exact dimensions — faster and more accurate than
  computing from geometry.
- ``get_spatial_structure()`` is the most useful call for "what's in room X" queries.
- When IfcSpace is absent (``BIMMetadata.has_spaces == False``), fall back to
  querying by storey or querying all elements of a class directly.
"""


class IFCTool(CPUTool):
    """In-kernel tool for querying IFC structured data."""

    TOOL_PROMPT_DESCRIPTION = TOOL_PROMPT_DESCRIPTION

    def __init__(self, ifc_model):
        """
        Parameters
        ----------
        ifc_model:
            The ``IFCModel`` kernel object (wraps an ifcopenshell.file).
        """
        self._model = ifc_model.model

    # ------------------------------------------------------------------
    # Core element queries
    # ------------------------------------------------------------------

    def query_elements(self, ifc_class: str, **filters) -> List[Dict[str, Any]]:
        """Return all elements of ``ifc_class``, optionally filtered.

        Filters are matched against element attributes (Name, GlobalId, etc.).
        Example::
            tools.IFC.query_elements("IfcDoor", Name="D01")

        Returns a list of dicts with keys:
            id, GlobalId, Name, Description, storey, space
        """
        try:
            elements = self._model.by_type(ifc_class)
        except Exception as exc:
            raise ValueError(f"Unknown IFC class: {ifc_class!r}. Error: {exc}") from exc

        results = []
        for el in elements:
            if not self._matches_filters(el, filters):
                continue
            results.append(self._element_to_dict(el))
        return results

    def get_spatial_structure(self) -> Dict[str, Any]:
        """Return the full building hierarchy as a nested dict.

        Structure::
            {
                "Storey 0": {
                    "Living Room": [{"id": 123, "GlobalId": "...", "Name": "Chair", ...}],
                    "Kitchen": [...],
                    "__unassigned__": [...]   # elements in this storey but no specific space
                },
                "Storey 1": { ... },
                "__unassigned__": { ... }   # elements not assigned to any storey
            }
        """
        structure: Dict[str, Any] = {}

        # Map elements to their containing storey and space
        el_to_storey: Dict[int, str] = {}
        el_to_space: Dict[int, str] = {}

        for rel in self._model.by_type("IfcRelContainedInSpatialStructure"):
            container = rel.RelatingStructure
            container_name = _entity_name(container)
            for el in rel.RelatedElements or []:
                if container.is_a("IfcBuildingStorey"):
                    el_to_storey[el.id()] = container_name
                elif container.is_a("IfcSpace"):
                    el_to_space[el.id()] = container_name
                    # Find storey via decomposition
                    storey = self._find_storey_of_space(container)
                    if storey and el.id() not in el_to_storey:
                        el_to_storey[el.id()] = _entity_name(storey)

        # Also map spaces to storeys
        for space in self._model.by_type("IfcSpace"):
            storey = self._find_storey_of_space(space)
            storey_name = _entity_name(storey) if storey else "__unassigned__"
            space_name = _entity_name(space)
            if storey_name not in structure:
                structure[storey_name] = {}
            if space_name not in structure[storey_name]:
                structure[storey_name][space_name] = []

        # Place elements
        for rel in self._model.by_type("IfcRelContainedInSpatialStructure"):
            container = rel.RelatingStructure
            for el in rel.RelatedElements or []:
                if el.is_a("IfcSpace"):
                    continue
                storey_name = el_to_storey.get(el.id(), "__unassigned__")
                space_name = el_to_space.get(el.id(), "__unassigned__")
                if storey_name not in structure:
                    structure[storey_name] = {}
                if space_name not in structure[storey_name]:
                    structure[storey_name][space_name] = []
                structure[storey_name][space_name].append(self._element_to_dict(el))

        return structure

    def count_in_space(self, space_name: str, ifc_class: Optional[str] = None) -> int:
        """Count elements in a named space, optionally filtered by class.

        Uses IfcRelContainedInSpatialStructure.  Case-insensitive name match.
        """
        space = self._find_space_by_name(space_name)
        if space is None:
            return 0
        count = 0
        for rel in self._model.by_type("IfcRelContainedInSpatialStructure"):
            if rel.RelatingStructure == space:
                for el in rel.RelatedElements or []:
                    if ifc_class is None or el.is_a(ifc_class):
                        count += 1
        return count

    # ------------------------------------------------------------------
    # Element detail
    # ------------------------------------------------------------------

    def get_quantities(self, global_id: str) -> Dict[str, float]:
        """Return exact dimensions from IfcElementQuantity.

        Keys: Length, Width, Height, Area, Volume, GrossArea, NetArea, etc.
        Values are in metres (length) or m² (area) or m³ (volume).
        Returns empty dict if no quantities are defined.
        """
        el = self._get_by_guid(global_id)
        result: Dict[str, float] = {}
        if el is None:
            return result

        for rel in self._model.by_type("IfcRelDefinesByProperties"):
            if el not in (rel.RelatedObjects or []):
                continue
            pdef = rel.RelatingPropertyDefinition
            if not pdef.is_a("IfcElementQuantity"):
                continue
            for qty in pdef.Quantities or []:
                name = qty.Name or ""
                for attr in ("LengthValue", "AreaValue", "VolumeValue",
                             "WeightValue", "CountValue", "TimeValue"):
                    val = getattr(qty, attr, None)
                    if val is not None:
                        result[name] = float(val)
                        break

        return result

    def get_properties(
        self, global_id: str, pset_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return all Pset_* properties for an element.

        Parameters
        ----------
        global_id:
            Element GlobalId (GUID string).
        pset_name:
            If given, return only properties from this property set.
            If None, return all property sets as {pset_name: {prop_name: value}}.
        """
        el = self._get_by_guid(global_id)
        if el is None:
            return {}

        all_psets: Dict[str, Dict[str, Any]] = {}
        for rel in self._model.by_type("IfcRelDefinesByProperties"):
            if el not in (rel.RelatedObjects or []):
                continue
            pdef = rel.RelatingPropertyDefinition
            if not pdef.is_a("IfcPropertySet"):
                continue
            ps_name = pdef.Name or ""
            props: Dict[str, Any] = {}
            for prop in pdef.HasProperties or []:
                if hasattr(prop, "NominalValue") and prop.NominalValue is not None:
                    props[prop.Name] = prop.NominalValue.wrappedValue
                elif hasattr(prop, "EnumerationValues"):
                    props[prop.Name] = [
                        v.wrappedValue for v in (prop.EnumerationValues or [])
                    ]
            if props:
                all_psets[ps_name] = props

        if pset_name is not None:
            return all_psets.get(pset_name, {})
        return all_psets

    def get_material(self, global_id: str) -> Any:
        """Return material info for an element.

        Returns a string (single material name), a dict with "layers" key
        (for IfcMaterialLayerSet), or None if unassigned.
        """
        el = self._get_by_guid(global_id)
        if el is None:
            return None

        for rel in self._model.by_type("IfcRelAssociatesMaterial"):
            if el not in (rel.RelatedObjects or []):
                continue
            mat = rel.RelatingMaterial
            if mat is None:
                continue
            if mat.is_a("IfcMaterial"):
                return mat.Name
            if mat.is_a("IfcMaterialList"):
                return [m.Name for m in (mat.Materials or [])]
            if mat.is_a("IfcMaterialLayerSetUsage"):
                mat = mat.ForLayerSet
            if mat.is_a("IfcMaterialLayerSet"):
                layers = []
                for layer in mat.MaterialLayers or []:
                    entry: Dict[str, Any] = {}
                    if layer.Material:
                        entry["name"] = layer.Material.Name
                    if layer.LayerThickness is not None:
                        entry["thickness"] = float(layer.LayerThickness)
                    layers.append(entry)
                return {"layers": layers}
            if mat.is_a("IfcMaterialConstituentSet"):
                return [
                    c.Material.Name
                    for c in (mat.MaterialConstituents or [])
                    if c.Material
                ]
        return None

    def get_bounding_box(self, global_id: str) -> Optional[Dict[str, Any]]:
        """Compute approximate bounding box of an element from its shape.

        Returns {"min": [x,y,z], "max": [x,y,z], "size": [w,h,d]} in metres,
        or None if geometry cannot be computed.

        Uses ifcopenshell.geom for a proper mesh; falls back to placement only.
        """
        el = self._get_by_guid(global_id)
        if el is None:
            return None

        try:
            import ifcopenshell.geom
            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS, True)
            shape = ifcopenshell.geom.create_shape(settings, el)
            verts = np.array(shape.geometry.verts).reshape(-1, 3)
            mn = verts.min(axis=0).tolist()
            mx = verts.max(axis=0).tolist()
            sz = [mx[i] - mn[i] for i in range(3)]
            return {"min": mn, "max": mx, "size": sz}
        except Exception:
            pass

        # Fallback: just return placement as a point bbox
        T = self.get_placement(global_id)
        if T is not None:
            pos = T[:3, 3].tolist()
            return {"min": pos, "max": pos, "size": [0.0, 0.0, 0.0]}
        return None

    def get_placement(self, global_id: str) -> Optional[np.ndarray]:
        """Return the 4×4 world-space placement matrix of an element.

        Returns None if placement cannot be computed.
        """
        el = self._get_by_guid(global_id)
        if el is None:
            return None

        try:
            import ifcopenshell.util.placement as ifc_placement
            T = ifc_placement.get_local_placement(el.ObjectPlacement)
            return np.array(T, dtype=np.float64)
        except Exception:
            return None

    def search_by_name(
        self, pattern: str, ifc_class: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find elements whose Name or LongName matches ``pattern``.

        ``pattern`` supports:
          - Glob wildcards: "living*", "*room*"
          - Regex (if wrapped in ``/.../)``): "/^Kitchen.*/i"
          - Plain substring match (case-insensitive)
        """
        if ifc_class:
            try:
                candidates = self._model.by_type(ifc_class)
            except Exception:
                candidates = []
        else:
            candidates = self._model.by_type("IfcElement") + self._model.by_type("IfcSpace")

        is_glob = any(c in pattern for c in "*?[]")
        is_regex = pattern.startswith("/") and pattern.endswith("/")

        if is_regex:
            rx = re.compile(pattern[1:-1], re.IGNORECASE)
            def matches(name: str) -> bool:
                return bool(rx.search(name))
        elif is_glob:
            def matches(name: str) -> bool:
                return fnmatch.fnmatch(name.lower(), pattern.lower())
        else:
            lp = pattern.lower()
            def matches(name: str) -> bool:
                return lp in name.lower()

        results = []
        for el in candidates:
            name = getattr(el, "Name", None) or ""
            long_name = getattr(el, "LongName", None) or ""
            if matches(name) or matches(long_name):
                results.append(self._element_to_dict(el))
        return results

    def get_space_boundaries(self, space_name: str) -> List[Dict[str, Any]]:
        """Return bounding elements (walls, slabs, windows) of a named space.

        Uses IfcRelSpaceBoundary.  Falls back to IfcRelContainedInSpatialStructure
        if no boundary relationships are defined.
        """
        space = self._find_space_by_name(space_name)
        if space is None:
            return []

        boundaries = []
        for rel in self._model.by_type("IfcRelSpaceBoundary"):
            if rel.RelatingSpace == space:
                el = rel.RelatedBuildingElement
                if el is not None:
                    boundaries.append(self._element_to_dict(el))

        if not boundaries:
            # Fallback: find elements assigned to this space
            for rel in self._model.by_type("IfcRelContainedInSpatialStructure"):
                if rel.RelatingStructure == space:
                    for el in rel.RelatedElements or []:
                        boundaries.append(self._element_to_dict(el))

        # Deduplicate by id
        seen = set()
        result = []
        for d in boundaries:
            if d["id"] not in seen:
                seen.add(d["id"])
                result.append(d)
        return result

    def get_type_info(self, global_id: str) -> Dict[str, Any]:
        """Return properties from the element's IfcTypeObject (IfcDoorType, etc.).

        Useful for getting manufacturer-specified dimensions that override
        per-instance IfcElementQuantity.
        """
        el = self._get_by_guid(global_id)
        if el is None:
            return {}

        for rel in self._model.by_type("IfcRelDefinesByType"):
            if el not in (rel.RelatedObjects or []):
                continue
            type_obj = rel.RelatingType
            props: Dict[str, Any] = {
                "TypeName": type_obj.Name,
                "TypeClass": type_obj.is_a(),
            }
            # Gather property sets from the type object
            for rel2 in self._model.by_type("IfcRelDefinesByProperties"):
                if type_obj not in (rel2.RelatedObjects or []):
                    continue
                pdef = rel2.RelatingPropertyDefinition
                if pdef.is_a("IfcPropertySet"):
                    for prop in pdef.HasProperties or []:
                        if hasattr(prop, "NominalValue") and prop.NominalValue:
                            props[f"{pdef.Name}.{prop.Name}"] = prop.NominalValue.wrappedValue
            return props
        return {}

    def get_classification(self, global_id: str) -> Optional[Dict[str, str]]:
        """Return the classification code (Uniclass, Omniclass, etc.) for an element."""
        el = self._get_by_guid(global_id)
        if el is None:
            return None

        for rel in self._model.by_type("IfcRelAssociatesClassification"):
            if el not in (rel.RelatedObjects or []):
                continue
            ref = rel.RelatingClassification
            if ref is None:
                continue
            result: Dict[str, str] = {}
            if hasattr(ref, "ReferencedSource") and ref.ReferencedSource:
                result["system"] = ref.ReferencedSource.Name or ""
            result["code"] = getattr(ref, "Identification", None) or getattr(ref, "ItemReference", None) or ""
            result["description"] = getattr(ref, "Name", None) or ""
            return result
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_by_guid(self, global_id: str):
        try:
            return self._model.by_guid(global_id)
        except Exception:
            return None

    def _find_space_by_name(self, space_name: str):
        """Find an IfcSpace by name (case-insensitive)."""
        target = space_name.lower()
        for space in self._model.by_type("IfcSpace"):
            name = (space.Name or "").lower()
            long_name = (space.LongName or "").lower()
            if target in name or target in long_name or name in target or long_name in target:
                return space
        return None

    def _find_storey_of_space(self, space):
        """Walk up the spatial decomposition to find the containing storey."""
        try:
            for rel in self._model.by_type("IfcRelAggregates"):
                if space in (rel.RelatedObjects or []):
                    parent = rel.RelatingObject
                    if parent.is_a("IfcBuildingStorey"):
                        return parent
                    if parent.is_a("IfcBuilding") or parent.is_a("IfcSite"):
                        return None
                    return self._find_storey_of_space(parent)
        except Exception:
            pass
        return None

    def _element_to_dict(self, el) -> Dict[str, Any]:
        """Convert an IFC element to a serializable dict."""
        return {
            "id": el.id(),
            "GlobalId": getattr(el, "GlobalId", None),
            "Name": getattr(el, "Name", None),
            "Description": getattr(el, "Description", None),
            "Class": el.is_a(),
        }

    @staticmethod
    def _matches_filters(el, filters: dict) -> bool:
        """Check if an element matches all keyword filters (exact match)."""
        for attr, val in filters.items():
            el_val = getattr(el, attr, None)
            if el_val != val:
                return False
        return True


def _entity_name(entity) -> str:
    """Return a display name for an IfcObjectDefinition."""
    name = getattr(entity, "Name", None) or ""
    long = getattr(entity, "LongName", None) or ""
    result = name or long
    if not result:
        result = f"{entity.is_a()}_{entity.id()}"
    return result
