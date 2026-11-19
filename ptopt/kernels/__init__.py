"""Custom Triton kernels and capability-gated auto-patching (Phase 3.7, lazy-import safe)."""

from __future__ import annotations

import importlib
from typing import Any

from ptopt.kernels.registry import (
    HAS_TRITON,
    FallbackRegistry,
    KernelRegistry,
    KernelSpec,
    apply_custom_kernels_pass,
    check_kernel_capability,
    install_custom_kernels_hooks,
    uninstall_custom_kernels_hooks,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ref_cross_entropy": ("ptopt.kernels.cross_entropy", "ref_cross_entropy"),
    "triton_cross_entropy": ("ptopt.kernels.cross_entropy", "triton_cross_entropy"),
    "ref_layernorm": ("ptopt.kernels.layernorm", "ref_layernorm"),
    "triton_layernorm": ("ptopt.kernels.layernorm", "triton_layernorm"),
    "ref_rmsnorm": ("ptopt.kernels.rmsnorm", "ref_rmsnorm"),
    "triton_rmsnorm": ("ptopt.kernels.rmsnorm", "triton_rmsnorm"),
    "ref_rope": ("ptopt.kernels.rope_or_swiglu", "ref_rope"),
    "triton_rope": ("ptopt.kernels.rope_or_swiglu", "triton_rope"),
    "ref_swiglu": ("ptopt.kernels.rope_or_swiglu", "ref_swiglu"),
    "triton_swiglu": ("ptopt.kernels.rope_or_swiglu", "triton_swiglu"),
    "triton_fused_linear_cross_entropy": ("ptopt.kernels.fused_linear_cross_entropy", "triton_fused_linear_cross_entropy"),
    "ref_linear_cross_entropy": ("ptopt.kernels.fused_linear_cross_entropy", "ref_linear_cross_entropy"),
    "LazyLinearLogits": ("ptopt.kernels.fused_linear_cross_entropy", "LazyLinearLogits"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_name, attr_name = _LAZY_EXPORTS[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FallbackRegistry",
    "HAS_TRITON",
    "KernelRegistry",
    "KernelSpec",
    "LazyLinearLogits",
    "apply_custom_kernels_pass",
    "check_kernel_capability",
    "install_custom_kernels_hooks",
    "ref_cross_entropy",
    "ref_layernorm",
    "ref_linear_cross_entropy",
    "ref_rmsnorm",
    "ref_rope",
    "ref_swiglu",
    "triton_cross_entropy",
    "triton_fused_linear_cross_entropy",
    "triton_layernorm",
    "triton_rmsnorm",
    "triton_rope",
    "triton_swiglu",
    "uninstall_custom_kernels_hooks",
]
