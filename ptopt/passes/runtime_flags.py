"""Pass 3.1a: CUDA math flags (TF32), cuDNN benchmark, and SDPA backend configuration."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from ptopt.config import OptimizeConfig

if TYPE_CHECKING:
    from ptopt.hooks.manager import HookManager


def _detect_cuda_sm(torch_mod: Any) -> int:
    try:
        idx = int(torch_mod.cuda.current_device()) if hasattr(torch_mod.cuda, "current_device") else 0
        props = torch_mod.cuda.get_device_properties(idx)
        return int(props.major) * 10 + int(props.minor)
    except Exception:  # noqa: BLE001
        return 0


def apply_runtime_flags_pass(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any = None,
) -> str:
    """Apply TF32, float32 matmul precision, cuDNN benchmark, and SDPA flags safely."""
    tmod = torch_mod if torch_mod is not None else sys.modules.get("torch")
    if tmod is None:
        return "SKIPPED: torch module not loaded"

    cuda_api = getattr(tmod, "cuda", None)
    try:
        cuda_ok = bool(cuda_api and cuda_api.is_available())
    except Exception:  # noqa: BLE001
        cuda_ok = False

    if not cuda_ok:
        return "SKIPPED: CUDA unavailable (CPU/non-CUDA backend)"

    sm_int = _detect_cuda_sm(tmod)
    backends = getattr(tmod, "backends", None)
    cuda_backend = getattr(backends, "cuda", None)
    cudnn_backend = getattr(backends, "cudnn", None)

    applied_parts: list[str] = []

    # TF32 only on sm >= 80 (Ampere+)
    if cfg.runtime.tf32:
        if sm_int >= 80:
            matmul = getattr(cuda_backend, "matmul", None)
            if matmul is not None and hasattr(matmul, "allow_tf32"):
                matmul.allow_tf32 = True
            if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
                cudnn_backend.allow_tf32 = True
            set_prec = getattr(tmod, "set_float32_matmul_precision", None)
            if callable(set_prec):
                set_prec("high")
            applied_parts.append("tf32=on")
        else:
            manager.record_event("runtime.tf32", "SKIPPED", f"sm_{sm_int} < 80")
            applied_parts.append("tf32=skipped(sm<80)")

    # cuDNN benchmark
    if cfg.runtime.cudnn_benchmark and cudnn_backend is not None and hasattr(cudnn_backend, "benchmark"):
        cudnn_backend.benchmark = True
        applied_parts.append("cudnn_benchmark=on")

    # Attention SDPA backends
    if cuda_backend is not None:
        flash_fn = getattr(cuda_backend, "enable_flash_sdp", None)
        mem_fn = getattr(cuda_backend, "enable_mem_efficient_sdp", None)
        math_fn = getattr(cuda_backend, "enable_math_sdp", None)

        if callable(flash_fn):
            allow_flash = bool(cfg.attention.flash_sdp and sm_int >= 80)
            flash_fn(allow_flash)
            applied_parts.append(f"flash_sdp={allow_flash}")
        if callable(mem_fn):
            mem_fn(bool(cfg.attention.mem_efficient_sdp))
            applied_parts.append(f"mem_sdp={cfg.attention.mem_efficient_sdp}")
        if callable(math_fn):
            math_fn(bool(cfg.attention.math_sdp))

    return ", ".join(applied_parts) if applied_parts else "no runtime flags changed"
