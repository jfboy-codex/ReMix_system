"""Thread-safe monotonic event log for paired serving experiments."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class EventLog:
    def __init__(self):
        self._lock = threading.Lock()
        self._events: list[dict] = []

    def record(self, kind: str, **fields) -> None:
        with self._lock:
            self._events.append({"t": time.perf_counter(), "kind": kind, **fields})

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._events)

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as out:
            for event in self.snapshot():
                out.write(json.dumps(event, ensure_ascii=False) + "\n")
