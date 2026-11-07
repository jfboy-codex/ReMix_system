from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable

from .profiling import ProfileCoordinator
from .workers import ProfileDecoder


class DeviceManager:
    """One manager per GPU. Foreground always wins between profile chunks."""

    def __init__(self, name: str, foreground: Callable[[Any], Any],
                 profile_decoder: ProfileDecoder, profiles: ProfileCoordinator):
        self.name = name
        self.foreground = foreground
        self.profile_decoder = profile_decoder
        self.profiles = profiles
        self._front: queue.Queue[tuple[Any, Future, Callable[[Any], Any]]] = queue.Queue()
        self._wake = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, item: Any, after: Callable[[Any], Any]) -> Future:
        if self._stop:
            raise RuntimeError("device manager is closed")
        future: Future = Future()
        self._front.put((item, future, after))
        self._wake.set()
        return future

    def _run(self) -> None:
        while True:
            try:
                item, future, after = self._front.get_nowait()
            except queue.Empty:
                if self._stop:
                    return
                chunk = self.profiles.claim()
                if chunk is not None:
                    job_id, latent = chunk
                    try:
                        image = self.profile_decoder.execute(latent)
                        self.profiles.decoded(job_id, image)
                    except Exception as exc:
                        self.profiles.failed(job_id, exc)
                    continue
                self._wake.wait(0.02)
                self._wake.clear()
                continue
            try:
                future.set_result(after(self.foreground(item)))
            except Exception as exc:
                future.set_exception(exc)
            finally:
                self._front.task_done()

    def close(self) -> None:
        self._front.join()
        self._stop = True
        self._wake.set()
        self._thread.join()

    def wait_foreground(self) -> None:
        self._front.join()

