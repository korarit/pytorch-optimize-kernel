"""Combined HardwareFingerprint model and collection entrypoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from src.detect.gpu_probe import GPUDeviceInfo, GPUProbeResult, probe_gpu
from src.detect.host_probe import HostInfo, probe_host


@dataclass(frozen=True)
class HardwareFingerprint:
    """Combined host and GPU hardware fingerprint used by the config resolver."""

    host: HostInfo
    gpu: GPUProbeResult

    @property
    def effective_backend(self) -> str:
        return self.gpu.backend

    @property
    def primary_gpu(self) -> GPUDeviceInfo | None:
        if not self.gpu.devices:
            return None
        idx = self.gpu.selected_index
        if 0 <= idx < len(self.gpu.devices):
            return self.gpu.devices[idx]
        return self.gpu.devices[0]

    @property
    def sm_int(self) -> int:
        dev = self.primary_gpu
        return dev.sm_int if dev is not None else 0

    @property
    def vram_mb(self) -> float:
        dev = self.primary_gpu
        return dev.total_vram_mb if dev is not None else 0.0

    @property
    def is_low_shm(self) -> bool:
        return self.host.shm_total_mb is not None and self.host.shm_total_mb < 2048.0

    @property
    def is_colab(self) -> bool:
        return self.host.is_colab

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host.to_dict(),
            "gpu": self.gpu.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HardwareFingerprint":
        return cls(
            host=HostInfo.from_dict(data["host"]),
            gpu=GPUProbeResult.from_dict(data["gpu"]),
        )


def collect_fingerprint(
    torch_mod: Any = None,
    device_index: int | None = None,
    env: Mapping[str, str] | None = None,
    query_driver: bool = True,
) -> HardwareFingerprint:
    """Collect a complete HardwareFingerprint for the active runtime."""
    host = probe_host(torch_mod=torch_mod, env=env)
    gpu = probe_gpu(
        torch_mod=torch_mod,
        device_index=device_index,
        env=env,
        query_driver=query_driver,
    )
    return HardwareFingerprint(host=host, gpu=gpu)
