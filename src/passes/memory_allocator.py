"""Pass 3.1b: CUDA memory allocator env, allocator preload, and CPU thread tuning."""

from __future__ import annotations

import ctypes.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, MutableMapping

from src.config import OptimizeConfig

if TYPE_CHECKING:
    from src.hooks.manager import HookManager


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
        cache_path = Path(cfg.compile.cache_dir).expanduser().resolve()
        cache_dir = str(cache_path)
        if "TORCHINDUCTOR_CACHE_DIR" not in env:
            env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
            notes.append(f"TORCHINDUCTOR_CACHE_DIR={cache_dir}")
        env.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
        triton_dir = str(cache_path / "triton")
        if "TRITON_CACHE_DIR" not in env:
            env["TRITON_CACHE_DIR"] = triton_dir
            notes.append(f"TRITON_CACHE_DIR={triton_dir}")

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
