"""Session store: model_id → ifc_path + cached modalities output."""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class BIMSession:
    model_id: str
    ifc_path: str
    upload_dir: Path
    modalities_dir: Path
    modalities_ready: bool = False
    modalities_error: Optional[str] = None
    modalities_output: object = None  # BIMModalitiesOutput or None
    rendering: bool = False


class SessionStore:
    """Thread-safe in-memory session store."""

    def __init__(self):
        self._sessions: Dict[str, BIMSession] = {}
        self._lock = threading.Lock()

    def create(self, ifc_path: str, upload_dir: Path, modalities_base: Path) -> BIMSession:
        model_id = str(uuid.uuid4())[:12]
        session = BIMSession(
            model_id=model_id,
            ifc_path=ifc_path,
            upload_dir=upload_dir,
            modalities_dir=modalities_base / model_id,
        )
        with self._lock:
            self._sessions[model_id] = session
        return session

    def get(self, model_id: str) -> Optional[BIMSession]:
        with self._lock:
            return self._sessions.get(model_id)

    def update(self, model_id: str, **kwargs) -> None:
        with self._lock:
            session = self._sessions.get(model_id)
            if session:
                for k, v in kwargs.items():
                    setattr(session, k, v)


_store = SessionStore()


def get_store() -> SessionStore:
    return _store
