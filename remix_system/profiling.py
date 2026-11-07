from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from .cache import CacheEntry, CacheManager
from .events import EventLog
from .types import Profile, Trace


@dataclass
class ProfileJob:
    job_id: str
    entry_id: str
    entry_version: str
    source_image: Any
    source_prompt: str
    trace: Trace
    cursor: int
    scores: list[tuple[int, float]] = field(default_factory=list)
    state: str = "ready"


class ProfileCoordinator:
    """Single owner of profile state. GPU chunks are claimed by idle devices."""

    def __init__(self, cache: CacheManager, score: Callable[[Any, str], float],
                 stable_tolerance: float = 0.002, max_pending: int = 16,
                 events: EventLog | None = None):
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self.cache = cache
        self.score = score
        self.events = events
        self.stable_tolerance = stable_tolerance
        self.max_pending = max_pending
        self.dropped = 0
        self._jobs: dict[str, ProfileJob] = {}
        self._ready: deque[str] = deque()
        self._cpu_queue: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self.errors: list[tuple[str, Exception]] = []
        self._cpu_thread = threading.Thread(target=self._score_loop, name="profile-cpu", daemon=True)
        self._cpu_thread.start()

    def enqueue(self, entry: CacheEntry, trace: Trace) -> str | None:
        if not trace.latents:
            return None
        # Same 85th-percentile / two-step stable-window rule as the source code.
        deltas = tuple(trace.noise_deltas[1:len(trace.latents)])
        min_step = 0
        if len(deltas) >= 2:
            ordered = sorted(deltas)
            pos = 0.85 * (len(ordered) - 1)
            lower = int(pos)
            threshold = ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)]
                                          - ordered[lower]) * (pos - lower)
            for i in range(len(deltas) - 1):
                if deltas[i] <= threshold and deltas[i + 1] <= threshold:
                    min_step = i
                    break
        job = ProfileJob(uuid.uuid4().hex, entry.entry_id, entry.version,
                         entry.image, entry.prompt, trace, min_step)
        with self._changed:
            # Evicted entries must not consume the bounded pending budget.
            for old_id, old in tuple(self._jobs.items()):
                if not self.cache.contains(old.entry_id, old.entry_version):
                    self._jobs.pop(old_id, None)
                    if self.events:
                        self.events.record("profile_cancel", job_id=old_id,
                                           entry_id=old.entry_id, reason="evicted")
            if len(self._jobs) >= self.max_pending:
                self.dropped += 1
                if self.events:
                    self.events.record("profile_drop", entry_id=entry.entry_id,
                                       reason="pending_limit")
                return None
            self._jobs[job.job_id] = job
            self._ready.append(job.job_id)
            if self.events:
                self.events.record("profile_enqueue", job_id=job.job_id,
                                   entry_id=entry.entry_id, min_step=min_step)
            self._changed.notify_all()
        return job.job_id

    def claim(self) -> tuple[str, Any] | None:
        """Claim one latent decode; called only after a device checks foreground work."""
        with self._changed:
            while self._ready:
                job_id = self._ready.popleft()
                job = self._jobs.get(job_id)
                if job is None or job.state != "ready":
                    continue
                if not self.cache.contains(job.entry_id, job.entry_version):
                    self._jobs.pop(job_id, None)
                    if self.events:
                        self.events.record("profile_cancel", job_id=job_id,
                                           entry_id=job.entry_id, reason="evicted")
                    continue
                job.state = "decoding"
                return job_id, job.trace.latents[job.cursor]
        return None

    def decoded(self, job_id: str, image: Any) -> None:
        with self._changed:
            job = self._jobs.get(job_id)
            if job is None or job.state != "decoding":
                return
            job.state = "scoring"
        self._cpu_queue.put((job_id, image))

    def failed(self, job_id: str, error: Exception) -> None:
        with self._changed:
            self.errors.append((job_id, error))
            if self.events:
                self.events.record("profile_error", job_id=job_id, error=str(error))
            self._jobs.pop(job_id, None)
            self._changed.notify_all()

    def _score_loop(self) -> None:
        while True:
            item = self._cpu_queue.get()
            if item is None:
                return
            job_id, image = item
            with self._changed:
                job = self._jobs.get(job_id)
                if job is None or job.state != "scoring":
                    continue
                step, prompt = job.cursor, job.source_prompt
            try:
                score_started = time.perf_counter()
                value = self.score(image, prompt)
                if self.events:
                    self.events.record("profile_score", job_id=job_id,
                                       step=step, duration_s=time.perf_counter()-score_started)
                with self._changed:
                    job = self._jobs.get(job_id)
                    if job is None or not self.cache.contains(job.entry_id, job.entry_version):
                        self._jobs.pop(job_id, None)
                        self._changed.notify_all()
                        continue
                    job.scores.append((step, value))
                    recent = [v for _, v in job.scores[-3:]]
                    stable = len(job.scores) >= 4 and max(recent) - min(recent) < self.stable_tolerance
                    finished = stable or step + 1 >= len(job.trace.latents)
                    if not finished:
                        job.cursor += 1
                        job.state = "ready"
                        self._ready.append(job_id)
                        self._changed.notify_all()
                        continue
                # Final source-image score and publication happen outside the lock.
                original = self.score(job.source_image, job.source_prompt)
                profile = Profile(job.scores[0][0], step, original, tuple(job.scores))
                self.cache.publish_profile(job.entry_id, job.entry_version, profile)
                if self.events:
                    self.events.record("profile_complete", job_id=job_id,
                                       entry_id=job.entry_id, steps=len(job.scores))
                with self._changed:
                    self._jobs.pop(job_id, None)
                    self._changed.notify_all()
            except Exception as exc:
                self.failed(job_id, exc)

    def wait_empty(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._changed:
            while self._jobs:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._changed.wait(remaining)
            return True

    @property
    def pending(self) -> int:
        with self._changed:
            return len(self._jobs)

    def close(self) -> None:
        self._cpu_queue.put(None)
        self._cpu_thread.join()

