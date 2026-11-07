from __future__ import annotations

from dataclasses import dataclass

from .cache import CacheEntry
from .types import EditPlan


@dataclass(frozen=True)
class ReMixPolicy:
    """Pure step and repaint decision; no queue, GPU or cache mutations."""

    total_steps: int = 40
    fallback_start_step: int = 20
    hit_threshold: float = 0.26
    repaint_mode: str = "graduated"

    def plan(self, entry: CacheEntry, query_score: float, prompt: str) -> EditPlan:
        profile = entry.profile
        start = self.fallback_start_step
        expand_end = None
        if profile is not None:
            # Fit a nondecreasing envelope before inverting the per-entry curve.
            points = sorted(profile.points)
            fitted = []
            for step, value in points:
                fitted.append((step, max(value, fitted[-1][1]) if fitted else value))
            if fitted:
                bottom = fitted[0][1]
                denom = profile.original_score - self.hit_threshold
                scale = ((profile.original_score - bottom) / denom
                         if denom > 1e-8 else 1.0)
                target = profile.original_score - max(0.0, profile.original_score - query_score) * scale
                if target <= fitted[0][1]:
                    start = fitted[0][0]
                elif target >= fitted[-1][1]:
                    start = fitted[-1][0]
                else:
                    for (s0, v0), (s1, v1) in zip(fitted, fitted[1:]):
                        if v0 <= target <= v1:
                            start = round(s0 + (s1 - s0) * (target - v0) / max(v1 - v0, 1e-8))
                            break
            start = max(profile.min_step, min(profile.max_step, start))
            expand_end = profile.max_step
        start = max(0, min(self.total_steps - 1, start))
        return EditPlan(entry.entry_id, entry.version, entry.image, entry.prompt,
                        prompt, start, (self.total_steps - start) / self.total_steps,
                        expand_end, self.repaint_mode)

