"""IFCModel: cloudpickle-serializable wrapper around an ifcopenshell file.

Injected into the Jupyter kernel as ``IFCModel``.  The agent can call
``IFCModel.model.by_type("IfcDoor")`` directly, or use the higher-level
helper methods on IFCTool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class IFCModel:
    """Thin wrapper so the ifcopenshell file is serializable via cloudpickle.

    cloudpickle can handle most objects by pickling the class definition
    alongside the data; wrapping it here makes the intent clear and lets
    us add helpers later without touching tool code.
    """

    def __init__(self, ifc_path: str):
        import ifcopenshell
        self.ifc_path = str(ifc_path)
        self.model = ifcopenshell.open(ifc_path)
        self.schema = self.model.schema  # "IFC2X3", "IFC4", etc.

    def by_type(self, ifc_class: str):
        """Shorthand for self.model.by_type(ifc_class)."""
        return self.model.by_type(ifc_class)

    def by_id(self, ifc_id: int):
        return self.model.by_id(ifc_id)

    def by_guid(self, guid: str):
        return self.model.by_guid(guid)

    def __repr__(self) -> str:
        return f"IFCModel(schema={self.schema!r}, path={Path(self.ifc_path).name!r})"
