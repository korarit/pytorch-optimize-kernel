"""Host machine, OS, CPU, memory, /dev/shm, and Python stack probing."""

from __future__ import annotations

import ctypes.util
import importlib.metadata
import os
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

_INTERESTING_ISA_FLAGS = frozenset({
    "avx2",
    "avx512f",
    "avx512_bf16",
    "amx_bf16",
    "amx_tile",
    "asimd",
    "neon",
})


@dataclass(frozen=True)
class HostInfo:
    """Host machine architecture and software stack details."""

    hostname: str
    os_name: str
    os_release: str
    cpu_arch: str
    cpu_model: str
    physical_cores: int | None
    logical_cores: int | None
    isa_flags: list[str] = field(default_factory=list)
    total_ram_mb: float | None = None
    available_ram_mb: float | None = None
    shm_total_mb: float | None = None
    shm_free_mb: float | None = None
    is_colab: bool = False
    colab_release_tag: str | None = None
    python_version: str = ""
    torch_version: str | None = None
    cuda_version: str | None = None
    hip_version: str | None = None
    triton_version: str | None = None
    jemalloc_path: str | None = None
    tcmalloc_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HostInfo":
        return cls(**dict(data))


def _default_pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parse_cpuinfo(proc_root: Path, fallback_model: str) -> tuple[str, int | None, list[str]]:
    cpuinfo_path = proc_root / "cpuinfo"
    if not cpuinfo_path.is_file():
        return fallback_model or "unknown", None, []

    model_name = "unknown"
    flags: set[str] = set()
    physical_pairs: set[tuple[str, str]] = set()
    phys_id = "0"
    core_id: str | None = None

    for line in cpuinfo_path.read_text(errors="replace").splitlines():
        if not line.strip():
            if core_id is not None:
                physical_pairs.add((phys_id, core_id))
            phys_id, core_id = "0", None
            continue
        if ":" not in line:
            continue
        key, val = (part.strip() for part in line.split(":", 1))
        if key in ("model name", "Model") and model_name == "unknown":
            model_name = val
        elif key == "physical id":
            phys_id = val
        elif key == "core id":
            core_id = val
        elif key in ("flags", "Features"):
            flags.update(val.split())

    if core_id is not None:
        physical_pairs.add((phys_id, core_id))

    if model_name == "unknown" and fallback_model:
        model_name = fallback_model
    isa = sorted(flags & _INTERESTING_ISA_FLAGS)
    return model_name, (len(physical_pairs) if physical_pairs else None), isa


def _parse_meminfo(proc_root: Path) -> tuple[float | None, float | None]:
    meminfo_path = proc_root / "meminfo"
    if not meminfo_path.is_file():
        return None, None
    values_kb: dict[str, int] = {}
    for line in meminfo_path.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if parts and parts[0].isdigit():
            values_kb[key] = int(parts[0])
    total = values_kb.get("MemTotal")
    avail = values_kb.get("MemAvailable")
    return (
        round(total / 1024.0, 1) if total is not None else None,
        round(avail / 1024.0, 1) if avail is not None else None,
    )


def _probe_shm(statvfs_fn: Callable[[Path], Any]) -> tuple[float | None, float | None]:
    shm_path = Path("/dev/shm")
    try:
        st = statvfs_fn(shm_path)
        total_mb = round((st.f_frsize * st.f_blocks) / (1024.0 * 1024.0), 1)
        free_mb = round((st.f_frsize * st.f_bavail) / (1024.0 * 1024.0), 1)
        return total_mb, free_mb
    except OSError:
        return None, None


def _find_allocator_libs(find_lib_fn: Callable[[str], str | None]) -> tuple[str | None, str | None]:
    candidates = {
        "jemalloc": ["jemalloc", "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2", "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"],
        "tcmalloc": ["tcmalloc", "tcmalloc_minimal", "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"],
    }
    resolved: dict[str, str | None] = {"jemalloc": None, "tcmalloc": None}
    for key, probes in candidates.items():
        for probe in probes:
            if probe.startswith("/"):
                if Path(probe).exists():
                    resolved[key] = probe
                    break
            else:
                found = find_lib_fn(probe)
                if found:
                    resolved[key] = found
                    break
    return resolved["jemalloc"], resolved["tcmalloc"]


def probe_host(
    torch_mod: Any = None,
    env: Mapping[str, str] | None = None,
    proc_root: Path | None = None,
    statvfs_fn: Callable[[Path], os.statvfs_result] | None = None,
    pkg_version_fn: Callable[[str], str | None] | None = None,
    find_lib_fn: Callable[[str], str | None] | None = None,
    os_name_override: str | None = None,
    cpu_arch_override: str | None = None,
    logical_cores_override: int | None = None,
) -> HostInfo:
    """Collect host architecture, memory, /dev/shm, Colab flag, and Python stack."""
    env_map = os.environ if env is None else env
    proc_dir = proc_root or Path("/proc")
    pkg_ver = pkg_version_fn or _default_pkg_version
    statvfs_impl = statvfs_fn or os.statvfs
    find_lib_impl = find_lib_fn or ctypes.util.find_library

    os_name = os_name_override or platform.system()
    cpu_arch = cpu_arch_override or platform.machine()
    cpu_model, phys_cores, isa_flags = _parse_cpuinfo(proc_dir, platform.processor())
    logical_cores = logical_cores_override if logical_cores_override is not None else os.cpu_count()
    total_ram_mb, avail_ram_mb = _parse_meminfo(proc_dir)

    if os_name.lower() == "linux" and statvfs_fn is None:
        shm_total_mb, shm_free_mb = _probe_shm(statvfs_impl)
    elif statvfs_fn is not None:
        shm_total_mb, shm_free_mb = _probe_shm(statvfs_impl)
    else:
        shm_total_mb, shm_free_mb = None, None

    colab_tag = env_map.get("COLAB_RELEASE_TAG") or env_map.get("COLAB_BACKEND_VERSION")
    colab_gpu = env_map.get("COLAB_GPU")
    is_colab = bool(colab_tag or colab_gpu or pkg_ver("google-colab"))

    if torch_mod is None:
        try:
            import torch as imported_torch  # noqa: PLC0415

            torch_mod = imported_torch
        except ImportError:
            torch_mod = None

    torch_version = getattr(torch_mod, "__version__", None) if torch_mod is not None else None
    torch_ver_obj = getattr(torch_mod, "version", None) if torch_mod is not None else None
    cuda_version = getattr(torch_ver_obj, "cuda", None) if torch_ver_obj is not None else None
    hip_version = getattr(torch_ver_obj, "hip", None) if torch_ver_obj is not None else None
    jemalloc_path, tcmalloc_path = _find_allocator_libs(find_lib_impl)

    return HostInfo(
        hostname=platform.node(),
        os_name=os_name,
        os_release=platform.release(),
        cpu_arch=cpu_arch,
        cpu_model=cpu_model,
        physical_cores=phys_cores,
        logical_cores=logical_cores,
        isa_flags=isa_flags,
        total_ram_mb=total_ram_mb,
        available_ram_mb=avail_ram_mb,
        shm_total_mb=shm_total_mb,
        shm_free_mb=shm_free_mb,
        is_colab=is_colab,
        colab_release_tag=colab_tag,
        python_version=platform.python_version(),
        torch_version=torch_version,
        cuda_version=cuda_version,
        hip_version=hip_version,
        triton_version=pkg_ver("triton"),
        jemalloc_path=jemalloc_path,
        tcmalloc_path=tcmalloc_path,
    )
