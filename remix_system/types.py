from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class Request:
    prompt: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    arrived_at: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True)
class Trace:
    """SD3.5 generation trace; latents are CPU owned between GPU chunks."""

    latents: Sequence[Any]
    noise_deltas: Sequence[float]


@dataclass(frozen=True)
class Generated:
    image: Any
    trace: Trace | None = None


@dataclass(frozen=True)
class Profile:
    min_step: int
    max_step: int
    original_score: float
    # (step, source-prompt CLIP score), in ascending step order.
    points: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class EditPlan:
    source_id: str
    source_version: str
    source_image: Any
    source_prompt: str
    target_prompt: str
    start_step: int
    strength: float
    mask_expand_end: int | None
    repaint_mode: str = "graduated"
    request_id: str = ""


@dataclass(frozen=True)
class Result:
    request_id: str
    image: Any
    path: str
    arrived_at: float
    completed_at: float

    @property
    def end_to_end_seconds(self) -> float:
        return self.completed_at - self.arrived_at

