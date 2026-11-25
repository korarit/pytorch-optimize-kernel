"""Pass 3.1b: CUDA memory allocator env, allocator preload, and CPU thread tuning."""

from __future__ import annotations

import ctypes.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, MutableMapping

from ptopt.config import OptimizeConfig, detect_runtime_cache_key, validate_or_scope_cache_dir

if TYPE_CHECKING:
    from ptopt.hooks.manager import HookManager


def _resolve_preload_lib(names: tuple[str, ...], explicit_paths: tuple[str, ...]) -> str | None:
    """Return an existing absolute library path for LD_PRELOAD, or None."""
    for p in explicit_paths:
        if Path(p).is_file():
            return p
    for name in names:
        found = ctypes.util.find_library(name)
        if found and Path(found).is_file():
            return found
    return None


def apply_prelaunch_env(cfg: OptimizeConfig, env: MutableMapping[str, str]) -> list[str]:
    """Populate pre-import env vars (alloc_conf, OMP/MKL threads, jemalloc) without clobbering user vars."""
    notes: list[str] = []

    if cfg.runtime.alloc_conf and "PYTORCH_CUDA_ALLOC_CONF" not in env:
        env["PYTORCH_CUDA_ALLOC_CONF"] = cfg.runtime.alloc_conf
        notes.append(f"PYTORCH_CUDA_ALLOC_CONF={cfg.runtime.alloc_conf}")

    if cfg.runtime.omp_threads > 0:
        threads_str = str(cfg.runtime.omp_threads)
        if "OMP_NUM_THREADS" not in env:
            env["OMP_NUM_THREADS"] = threads_str
            notes.append(f"OMP_NUM_THREADS={threads_str}")
        if "MKL_NUM_THREADS" not in env:
            env["MKL_NUM_THREADS"] = threads_str
            notes.append(f"MKL_NUM_THREADS={threads_str}")

    if cfg.runtime.jemalloc and "LD_PRELOAD" not in env:
        lib_path = _resolve_preload_lib(
            ("jemalloc",),
            (
                "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2",
                "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2",
            ),
        )
        if lib_path:
            env["LD_PRELOAD"] = lib_path
            notes.append(f"LD_PRELOAD={lib_path}")

    if (cfg.compile.enabled or cfg.kernels.enabled or cfg.inference.cuda_graphs) and cfg.compile.cache_dir:
        key_info = detect_runtime_cache_key()
        existing_inductor_dir = env.get("TORCHINDUCTOR_CACHE_DIR")
        same_as_cfg_cache = False
        if existing_inductor_dir is not None:
            try:
                same_as_cfg_cache = (
                    Path(existing_inductor_dir).resolve() == Path(cfg.compile.cache_dir).resolve()
                )
            except OSError:
                same_as_cfg_cache = existing_inductor_dir == cfg.compile.cache_dir
        if existing_inductor_dir is None or same_as_cfg_cache:
            validated = validate_or_scope_cache_dir(
                cfg.compile.cache_dir,
                key_info,
                write_meta=False,
                emit_warnings=False,
            )
            cache_dir = str(validated.cache_dir)
            if existing_inductor_dir is None or validated.rotated:
                env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
                notes.append(f"TORCHINDUCTOR_CACHE_DIR={cache_dir}")
            triton_dir = str(validated.triton_dir)
            if "TRITON_CACHE_DIR" not in env or validated.rotated:
                env["TRITON_CACHE_DIR"] = triton_dir
                notes.append(f"TRITON_CACHE_DIR={triton_dir}")
            if validated.rotated:
                notes.append(f"cache_rotated={key_info.cache_key}")
        env.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
        env.setdefault("PTOPT_CACHE_KEY", key_info.cache_key)

    return notes


def apply_memory_allocator_pass(
    cfg: OptimizeConfig,
    _manager: "HookManager",
    torch_mod: Any = None,
) -> str:
    """Apply runtime CPU thread count and report active memory allocator configuration."""
    tmod = torch_mod if torch_mod is not None else sys.modules.get("torch")
    parts: list[str] = []

    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if alloc_conf:
        parts.append(f"alloc_conf={alloc_conf}")

    if cfg.runtime.omp_threads > 0 and tmod is not None:
        set_threads = getattr(tmod, "set_num_threads", None)
        if callable(set_threads):
            try:
                set_threads(int(cfg.runtime.omp_threads))
                parts.append(f"torch_threads={cfg.runtime.omp_threads}")
            except RuntimeError:
                pass

    if os.environ.get("LD_PRELOAD"):
        parts.append(f"ld_preload={os.environ['LD_PRELOAD']}")

    return ", ".join(parts) if parts else "default allocator/threads"


def probe_cuda_vram_headroom(torch_mod: Any) -> dict[str, float] | None:
    """Return live CUDA VRAM usage/headroom metrics in MB, or None when unavailable."""
    if torch_mod is None:
        return None
    cuda_api = getattr(torch_mod, "cuda", None)
    if cuda_api is None:
        return None
    try:
        if not cuda_api.is_available():
            return None
        idx = int(cuda_api.current_device()) if hasattr(cuda_api, "current_device") else 0
        total_cache = getattr(cuda_api, "_ptopt_total_vram_cache", None)
        total_bytes = total_cache.get(idx) if isinstance(total_cache, dict) else None
        if total_bytes is None:
            props_fn = getattr(cuda_api, "get_device_properties", None)
            if not callable(props_fn):
                return None
            total_bytes = float(getattr(props_fn(idx), "total_memory", 0.0) or 0.0)
            if total_bytes > 0.0:
                if not isinstance(total_cache, dict):
                    total_cache = {}
                    try:
                        cuda_api._ptopt_total_vram_cache = total_cache
                    except Exception:  # noqa: BLE001
                        total_cache = None
                if isinstance(total_cache, dict):
                    total_cache[idx] = total_bytes
        if total_bytes <= 0.0:
            return None
        alloc_fn = getattr(cuda_api, "memory_allocated", None)
        resv_fn = getattr(cuda_api, "memory_reserved", None)
        if not callable(alloc_fn) and not callable(resv_fn):
            return None
        alloc_bytes = float(alloc_fn(idx)) if callable(alloc_fn) else 0.0
        resv_bytes = float(resv_fn(idx)) if callable(resv_fn) else alloc_bytes
        used_bytes = max(alloc_bytes, resv_bytes)
        mb = 1024.0 * 1024.0
        return {
            "total_mb": round(total_bytes / mb, 2),
            "allocated_mb": round(alloc_bytes / mb, 2),
            "reserved_mb": round(resv_bytes / mb, 2),
            "free_mb": round(max(0.0, total_bytes - used_bytes) / mb, 2),
            "used_ratio": round(min(1.0, max(0.0, used_bytes / total_bytes)), 4),
        }
    except Exception:  # noqa: BLE001
        return None
