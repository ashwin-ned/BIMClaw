"""BIMMetadata: pre-extracted summary of an IFC file.

Injected into the Jupyter kernel as ``BIMMetadata`` and also serialised as
a text block into the planning prompt so the planner can reason about the
model without running any code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class BIMMetadata:
    """Pre-extracted structural summary of an IFC file."""

    ifc_path: str
    schema: str                                  # "IFC2X3", "IFC4", ...
    storey_names: List[str] = field(default_factory=list)
    space_names: List[str] = field(default_factory=list)
    element_counts: Dict[str, int] = field(default_factory=dict)
    has_spaces: bool = False
    has_materials: bool = False
    has_quantities: bool = False
    bounding_box: Optional[Tuple[float, float, float, float, float, float]] = None
    # (x_min, y_min, z_min, x_max, y_max, z_max) in metres

    @classmethod
    def from_ifc(cls, ifc_path: str) -> "BIMMetadata":
        """Extract metadata from an IFC file using ifcopenshell."""
        import ifcopenshell
        import ifcopenshell.util.element as ifc_util

        model = ifcopenshell.open(ifc_path)
        schema = model.schema

        storey_names = [
            s.Name or f"Storey_{i}"
            for i, s in enumerate(model.by_type("IfcBuildingStorey"))
        ]
        space_names = [
            s.Name or s.LongName or f"Space_{i}"
            for i, s in enumerate(model.by_type("IfcSpace"))
        ]

        # Count element instances per class
        ELEMENT_CLASSES = [
            "IfcWall", "IfcWallStandardCase", "IfcDoor", "IfcWindow",
            "IfcSlab", "IfcColumn", "IfcBeam", "IfcStair", "IfcStairFlight",
            "IfcRailing", "IfcRoof", "IfcFurnishingElement", "IfcSpace",
            "IfcBuildingStorey", "IfcMember", "IfcPlate", "IfcCovering",
            "IfcOpeningElement", "IfcFlow", "IfcDistributionElement",
        ]
        element_counts: Dict[str, int] = {}
        for cls_name in ELEMENT_CLASSES:
            try:
                n = len(model.by_type(cls_name))
                if n > 0:
                    element_counts[cls_name] = n
            except Exception:
                pass

        has_spaces = len(model.by_type("IfcSpace")) > 0
        has_materials = len(model.by_type("IfcMaterial")) > 0
        has_quantities = len(model.by_type("IfcElementQuantity")) > 0

        bounding_box = cls._compute_bounding_box(model)

        return cls(
            ifc_path=str(ifc_path),
            schema=schema,
            storey_names=storey_names,
            space_names=space_names,
            element_counts=element_counts,
            has_spaces=has_spaces,
            has_materials=has_materials,
            has_quantities=has_quantities,
            bounding_box=bounding_box,
        )

    @staticmethod
    def _compute_bounding_box(model) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Estimate bounding box from IfcBuildingStorey elevations or IfcSite."""
        try:
            storeys = model.by_type("IfcBuildingStorey")
            if storeys:
                elevs = []
                for s in storeys:
                    if s.Elevation is not None:
                        elevs.append(float(s.Elevation))
                if elevs:
                    z_min = min(elevs)
                    z_max = max(elevs) + 3.0  # assume ~3m floor height
                    return (0.0, 0.0, z_min, 1.0, 1.0, z_max)
        except Exception:
            pass
        return None

    def to_text(self) -> str:
        """Return a human-readable summary for the planning prompt."""
        lines = [
            f"IFC Schema: {self.schema}",
            f"File: {Path(self.ifc_path).name}",
            f"Building Storeys ({len(self.storey_names)}): {', '.join(self.storey_names) or 'none detected'}",
            f"Spaces ({len(self.space_names)}): {', '.join(self.space_names[:20]) or 'none detected'}"
            + (" ..." if len(self.space_names) > 20 else ""),
            "",
            "Element counts:",
        ]
        if self.element_counts:
            for cls_name, cnt in sorted(self.element_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {cls_name}: {cnt}")
        else:
            lines.append("  (no elements found)")
        lines += [
            "",
            f"Has IfcSpace (rooms): {self.has_spaces}",
            f"Has IfcMaterial: {self.has_materials}",
            f"Has IfcElementQuantity (dimensions): {self.has_quantities}",
        ]
        if self.bounding_box:
            x0, y0, z0, x1, y1, z1 = self.bounding_box
            lines.append(f"Bounding box (approx): X[{x0:.1f},{x1:.1f}] Y[{y0:.1f},{y1:.1f}] Z[{z0:.1f},{z1:.1f}] m")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BIMMetadata(schema={self.schema!r}, "
            f"storeys={len(self.storey_names)}, spaces={len(self.space_names)}, "
            f"elements={sum(self.element_counts.values())})"
        )
