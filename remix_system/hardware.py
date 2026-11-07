"""NVML GPU utilization and memory samples; no cost estimation."""
from __future__ import annotations

import threading
import time


class GpuSampler:
    def __init__(self, indices: tuple[int, ...], interval_s: float):
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError("install nvidia-ml-py to collect GPU utilization") from exc
        self.nvml = pynvml
        self.indices = indices
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)

    def start(self) -> None:
        self.nvml.nvmlInit()
        self.handles = [self.nvml.nvmlDeviceGetHandleByIndex(i) for i in self.indices]
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.perf_counter()
            for index, handle in zip(self.indices, self.handles):
                try:
                    util = self.nvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = self.nvml.nvmlDeviceGetMemoryInfo(handle)
                    self.samples.append({"t": now, "gpu_index": index,
                                         "utilization_pct": util.gpu,
                                         "memory_used_mib": memory.used / 1048576,
                                         "memory_total_mib": memory.total / 1048576})
                except self.nvml.NVMLError:
                    continue
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self.nvml.nvmlShutdown()
