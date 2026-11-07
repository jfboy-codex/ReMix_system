from __future__ import annotations

import time
from concurrent.futures import Future
from typing import Any, Callable

from .cache import CacheManager
from .device import DeviceManager
from .policy import ReMixPolicy
from .profiling import ProfileCoordinator
from .types import Generated, Request, Result
from .workers import DecoderBackend, LargeBackend, LargeWorker, SmallBackend, SmallWorker, ProfileDecoder


class ReMixRuntime:
    """Routes requests and owns service lifetime; policies are injected."""

    def __init__(self, large: LargeBackend, small: SmallBackend,
                 large_decoder: DecoderBackend, small_decoder: DecoderBackend,
                 cache_score: Callable[[Any, str], float],
                 profile_score: Callable[[Any, str], float],
                 threshold: float = 0.26, capacity: int = 100,
                 policy: ReMixPolicy | None = None,
                 max_pending_profiles: int = 16):
        self.cache = CacheManager(cache_score, threshold, capacity)
        self.profiles = ProfileCoordinator(self.cache, profile_score,
                                           max_pending=max_pending_profiles)
        self.policy = policy or ReMixPolicy(hit_threshold=threshold)
        self.large = DeviceManager("gpu-large", LargeWorker(large).execute,
                                   ProfileDecoder(large_decoder), self.profiles)
        self.small = DeviceManager("gpu-small", SmallWorker(small).execute,
                                   ProfileDecoder(small_decoder), self.profiles)
        self._closed = False

    def submit(self, prompt: str) -> Future:
        if self._closed:
            raise RuntimeError("runtime is closed")
        request = Request(prompt)
        entry, score = self.cache.lookup(prompt)
        if entry is None:
            def complete(generated: Generated) -> Result:
                new_entry = self.cache.insert(prompt, generated.image)
                if generated.trace is not None:
                    self.profiles.enqueue(new_entry, generated.trace)
                return Result(request.request_id, generated.image, "miss",
                              request.arrived_at, time.perf_counter())
            return self.large.submit(request, complete)
        plan = self.policy.plan(entry, score, prompt)

        def complete(image: Any) -> Result:
            return Result(request.request_id, image, "hit", request.arrived_at,
                          time.perf_counter())
        return self.small.submit(plan, complete)

    def close(self, drain_profiles: bool = True, timeout: float | None = 30) -> bool:
        self._closed = True
        # Foreground requests finish before optional profile drain.
        self.large.wait_foreground()
        self.small.wait_foreground()
        drained = self.profiles.wait_empty(timeout) if drain_profiles else False
        self.large.close()
        self.small.close()
        self.profiles.close()
        return drained

