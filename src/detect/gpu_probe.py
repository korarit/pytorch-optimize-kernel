"""GPU architecture and capability probing."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GPUDeviceInfo:
    """Capabilities and hardware properties for a single GPU device."""

    index: int
    name: str
    compute_capability: str
    major: int
    minor: int
    sm_int: int
    sm_name: str
    total_vram_mb: float
    sm_count: int
    bf16_supported: bool
    tf32_supported: bool
    fp8_supported: bool
    flash_sdp_supported: bool
    mem_efficient_sdp_supported: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GPUDeviceInfo":
        return cls(**dict(data))


@dataclass(frozen=True)
class GPUProbeResult:
    """Aggregated GPU detection result across all visible devices."""

    backend: str
    device_count: int
    selected_index: int
    cuda_visible_devices: str | None
    devices: list[GPUDeviceInfo] = field(default_factory=list)
    driver_version: str | None = None
    probe_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device_count": self.device_count,
            "selected_index": self.selected_index,
            "cuda_visible_devices": self.cuda_visible_devices,
            "devices": [d.to_dict() for d in self.devices],
            "driver_version": self.driver_version,
            "probe_errors": list(self.probe_errors),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GPUProbeResult":
        return cls(
            backend=str(data["backend"]),
            device_count=int(data["device_count"]),
            selected_index=int(data["selected_index"]),
            cuda_visible_devices=data.get("cuda_visible_devices"),
            devices=[GPUDeviceInfo.from_dict(d) for d in data.get("devices", [])],
            driver_version=data.get("driver_version"),
            probe_errors=list(data.get("probe_errors", [])),
        )


def _query_driver_version() -> str | None:
    """Query NVIDIA driver version via nvidia-smi if available."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=4.0,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return None


def _resolve_selected_index(
    device_count: int,
    device_index: int | None,
    env_map: Mapping[str, str],
    errors: list[str],
) -> int:
    """Determine active GPU index from explicit arg or LOCAL_RANK."""
    if device_count <= 0:
        return 0
    candidate = device_index
    if candidate is None:
        raw_rank = env_map.get("LOCAL_RANK")
        if raw_rank is not None and raw_rank.isdigit():
            candidate = int(raw_rank)
        else:
            candidate = 0
    if candidate < 0 or candidate >= device_count:
        errors.append(f"Requested device_index={candidate} out of range [0, {device_count - 1}]; clamped to 0.")
        return 0
    return candidate


def _probe_cuda_devices(torch_mod: Any, backend: str, errors: list[str]) -> list[GPUDeviceInfo]:
    """Collect GPUDeviceInfo for every CUDA/ROCm device."""
    devices: list[GPUDeviceInfo] = []
    count = int(torch_mod.cuda.device_count())
    bf16_api = getattr(torch_mod.cuda, "is_bf16_supported", None)
    global_bf16 = bool(bf16_api()) if callable(bf16_api) else False
    cuda_backends = getattr(getattr(torch_mod, "backends", None), "cuda", None)
    has_flash_api = bool(cuda_backends and hasattr(cuda_backends, "flash_sdp_enabled"))

    for idx in range(count):
        try:
            props = torch_mod.cuda.get_device_properties(idx)
            major = int(props.major)
            minor = int(props.minor)
            sm_int = major * 10 + minor
            is_nvidia = backend == "cuda"
            devices.append(
                GPUDeviceInfo(
                    index=idx,
                    name=str(props.name),
                    compute_capability=f"{major}.{minor}",
                    major=major,
                    minor=minor,
                    sm_int=sm_int,
                    sm_name=f"sm_{sm_int}",
                    total_vram_mb=round(float(props.total_memory) / (1024.0 * 1024.0), 1),
                    sm_count=int(getattr(props, "multi_processor_count", 0)),
                    bf16_supported=bool(global_bf16 and (not is_nvidia or sm_int >= 80)),
                    tf32_supported=bool(is_nvidia and sm_int >= 80),
                    fp8_supported=bool(is_nvidia and sm_int >= 89),
                    flash_sdp_supported=bool(is_nvidia and sm_int >= 80 and has_flash_api),
                    mem_efficient_sdp_supported=bool(is_nvidia and sm_int >= 70),
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Failed to probe GPU[{idx}]: {exc}")
    return devices


def probe_gpu(
    torch_mod: Any = None,
    device_index: int | None = None,
    env: Mapping[str, str] | None = None,
    query_driver: bool = True,
) -> GPUProbeResult:
    """Detect active GPU backend and hardware capabilities safely."""
    env_map = os.environ if env is None else env
    errors: list[str] = []
    cuda_vis = env_map.get("CUDA_VISIBLE_DEVICES")

    if torch_mod is None:
        try:
            import torch as imported_torch  # noqa: PLC0415

            torch_mod = imported_torch
        except ImportError as exc:
            errors.append(f"PyTorch not installed: {exc}")
            return GPUProbeResult(
                backend="cpu",
                device_count=0,
                selected_index=0,
                cuda_visible_devices=cuda_vis,
                probe_errors=errors,
            )

    cuda_avail = False
    try:
        cuda_avail = bool(torch_mod.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"torch.cuda.is_available() failed: {exc}")

    if cuda_avail:
        hip_ver = getattr(getattr(torch_mod, "version", None), "hip", None)
        backend = "rocm" if hip_ver else "cuda"
        devices = _probe_cuda_devices(torch_mod, backend, errors)
        count = len(devices)
        selected = _resolve_selected_index(count, device_index, env_map, errors)
        driver = _query_driver_version() if (query_driver and backend == "cuda") else None
        return GPUProbeResult(
            backend=backend,
            device_count=count,
            selected_index=selected,
            cuda_visible_devices=cuda_vis,
            devices=devices,
            driver_version=driver,
            probe_errors=errors,
        )

    mps_avail = False
    try:
        mps_backend = getattr(getattr(torch_mod, "backends", None), "mps", None)
        mps_avail = bool(mps_backend and mps_backend.is_available())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"torch.backends.mps.is_available() failed: {exc}")

    if mps_avail:
        return GPUProbeResult(
            backend="mps",
            device_count=1,
            selected_index=0,
            cuda_visible_devices=cuda_vis,
            probe_errors=errors,
        )

    return GPUProbeResult(
        backend="cpu",
        device_count=0,
        selected_index=0,
        cuda_visible_devices=cuda_vis,
        probe_errors=errors,
    )
