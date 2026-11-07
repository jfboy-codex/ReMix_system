from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

from .types import Profile


@dataclass(frozen=True)
class CacheEntry:
    entry_id: str
    version: str
    prompt: str
    image: Any
    profile: Profile | None = None
    image_embedding: tuple[float, ...] | None = None


class CacheManager:
    """Owns entries, retrieval, eviction and atomic profile publication only."""

    def __init__(self, score: Callable[[Any, str], float], threshold: float, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._score = score
        self._embedding_lookup = (hasattr(score, "encode_text") and
                                  hasattr(score, "encode_image"))
        self.threshold = threshold
        self.capacity = capacity
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def lookup(self, prompt: str) -> tuple[CacheEntry | None, float]:
        # Scoring is outside the cache lock; a candidate is checked again below.
        with self._lock:
            candidates = tuple(self._entries.values())
        if self._embedding_lookup:
            query = self._score.encode_text(prompt)
            ranked = sorted(((sum(a * b for a, b in zip(e.image_embedding, query)), e)
                             for e in candidates), key=lambda pair: pair[0], reverse=True)
        else:
            ranked = sorted(((self._score(e.image, prompt), e) for e in candidates),
                        key=lambda pair: pair[0], reverse=True)
        with self._lock:
            for score, entry in ranked:
                if score < self.threshold:
                    break
                current = self._entries.get(entry.entry_id)
                if current is not None and current.version == entry.version:
                    return current, score
        return None, ranked[0][0] if ranked else 0.0

    def insert(self, prompt: str, image: Any) -> CacheEntry:
        embedding = (tuple(self._score.encode_image(image))
                     if self._embedding_lookup else None)
        entry = CacheEntry(uuid.uuid4().hex, uuid.uuid4().hex, prompt, image,
                           image_embedding=embedding)
        with self._lock:
            if len(self._entries) >= self.capacity:
                self._entries.popitem(last=False)  # FIFO, replaceable policy boundary.
            self._entries[entry.entry_id] = entry
        return entry

    def contains(self, entry_id: str, version: str) -> bool:
        with self._lock:
            entry = self._entries.get(entry_id)
            return entry is not None and entry.version == version

    def publish_profile(self, entry_id: str, version: str, profile: Profile) -> bool:
        with self._lock:
            old = self._entries.get(entry_id)
            if old is None or old.version != version:
                return False
            self._entries[entry_id] = CacheEntry(old.entry_id, old.version,
                                                  old.prompt, old.image, profile,
                                                  old.image_embedding)
            return True

    def snapshot(self) -> tuple[CacheEntry, ...]:
        with self._lock:
            return tuple(self._entries.values())

