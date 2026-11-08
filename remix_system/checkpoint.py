"""Persist completed serving responses without blocking GPU worker threads."""
from __future__ import annotations

import json
import queue
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any


class RequestRecorder:
    def __init__(self, output: Path, attempt_id: str):
        self.images = output / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.path = output / f"requests.partial.{attempt_id}.jsonl"
        self._out = self.path.open("x", encoding="utf-8", buffering=1)
        self.rows: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._queue: queue.Queue[tuple[str, Future, dict | None] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="request-recorder", daemon=True)
        self._thread.start()

    def record(self, prompt: str, future: Future, meta: dict | None) -> None:
        self._queue.put((prompt, future, meta))

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    return
                prompt, future, meta = item
                try:
                    if not future.cancelled():
                        result = future.result()
                        row = {
                            "request_id": result.request_id, "prompt": prompt,
                            "path": result.path,
                            "end_to_end_s": result.end_to_end_seconds,
                            "trace_index": meta.get("index") if meta else None,
                        }
                        result.image.save(self.images / f"{result.request_id}.png")
                        self._out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        self._out.flush()
                        self.rows.append(row)
                except Exception as exc:
                    self.errors.append(f"{prompt[:80]}: {exc}")
                finally:
                    self._queue.task_done()
        finally:
            self._out.close()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
