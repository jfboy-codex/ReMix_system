from __future__ import annotations

from typing import Any, Callable, Protocol

from .types import EditPlan, Generated, Request


class LargeBackend(Protocol):
    def generate(self, prompt: str) -> Generated: ...


class SmallBackend(Protocol):
    def edit(self, plan: EditPlan) -> Any: ...


class DecoderBackend(Protocol):
    def decode(self, latent: Any) -> Any: ...


class LargeWorker:
    def __init__(self, backend: LargeBackend):
        self.backend = backend

    def execute(self, request: Request) -> Generated:
        return self.backend.generate(request.prompt)


class SmallWorker:
    def __init__(self, backend: SmallBackend):
        self.backend = backend

    def execute(self, plan: EditPlan) -> Any:
        return self.backend.edit(plan)


class ProfileDecoder:
    """Each device gets its own SD3.5 VAE-backed decoder instance."""

    def __init__(self, backend: DecoderBackend):
        self.backend = backend

    def execute(self, latent: Any) -> Any:
        return self.backend.decode(latent)

