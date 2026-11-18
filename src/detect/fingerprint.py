"""Combined HardwareFingerprint model and collection entrypoint."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import socket
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from src.detect.gpu_probe import GPUDeviceInfo, GPUProbeResult, probe_gpu
from src.detect.host_probe import HostInfo, _probe_shm, probe_host


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


def _fast_pkg_version(pkg_name: str, torch_mod: Any = None) -> str:
    if pkg_name == "torch" and torch_mod is not None and hasattr(torch_mod, "__version__"):
        return str(torch_mod.__version__)
    loaded = sys.modules.get(pkg_name)
    if loaded is not None and hasattr(loaded, "__version__"):
        return str(loaded.__version__)
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _fingerprint_cache_path(torch_mod: Any, env_map: Mapping[str, str]) -> Path:
    cache_dir = Path(os.environ.get("PTOPT_FP_CACHE_DIR", str(Path.home() / ".cache" / "ptopt")))
    torch_ver = _fast_pkg_version("torch", torch_mod)
    triton_ver = _fast_pkg_version("triton")
    cuda_vis = env_map.get("CUDA_VISIBLE_DEVICES", "")
    # Include cached driver version anchor when torch.version.cuda changes
    cuda_ver = str(getattr(getattr(torch_mod, "version", None), "cuda", "") or "") if torch_mod else ""
    key_str = "|".join([socket.gethostname(), cuda_vis, sys.executable, torch_ver, triton_ver, cuda_ver])
    digest = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"fingerprint-{digest}.json"


def _probe_live_shm() -> tuple[float | None, float | None]:
    if not hasattr(os, "statvfs"):
        return None, None
    try:
        return _probe_shm(os.statvfs)
    except Exception:  # noqa: BLE001
        return None, None


def collect_fingerprint(
    torch_mod: Any = None,
    device_index: int | None = None,
    env: Mapping[str, str] | None = None,
    query_driver: bool = True,
) -> HardwareFingerprint:
    """Collect a complete HardwareFingerprint, caching static host/GPU fields while probing `/dev/shm` live."""
    use_cache = (
        env is None
        and device_index is None
        and query_driver
        and os.environ.get("PTOPT_NO_FP_CACHE") != "1"
    )
    cache_path = _fingerprint_cache_path(torch_mod, os.environ) if use_cache else None

    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            fp = HardwareFingerprint.from_dict(cached)
            # Always probe /dev/shm live (never serve stale container shm quota from disk cache!)
            shm_total, shm_free = _probe_live_shm()
            live_host = replace(fp.host, shm_total_mb=shm_total, shm_free_mb=shm_free)
            return HardwareFingerprint(host=live_host, gpu=fp.gpu)
        except Exception:  # noqa: BLE001
            pass

    host = probe_host(torch_mod=torch_mod, env=env)
    gpu = probe_gpu(
        torch_mod=torch_mod,
        device_index=device_index,
        env=env,
        query_driver=query_driver,
    )
    fp = HardwareFingerprint(host=host, gpu=gpu)

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Store static host/GPU fields without persisting volatile /dev/shm readings
            static_host = replace(fp.host, shm_total_mb=None, shm_free_mb=None)
            payload = {"host": static_host.to_dict(), "gpu": fp.gpu.to_dict()}
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    return fp
