"""Kernel capability specs, FallbackRegistry tracking, and zero-code PyTorch auto-patcher."""

from __future__ import annotations

import functools
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from src.config import OptimizeConfig

if TYPE_CHECKING:
    from src.hooks.manager import HookManager

HAS_TRITON = importlib.util.find_spec("triton") is not None

_KERNELS_INSTALLED = False


@dataclass(frozen=True)
class KernelSpec:
    """Hardware & dtype capability requirements for a custom kernel."""

    name: str
    min_sm: int = 80
    supported_dtypes: tuple[str, ...] = ("fp32", "fp16", "bf16")
    triton_required: bool = True
    fallback_on_error: bool = True


@dataclass
class KernelStatusEntry:
    name: str
    status: str  # APPLIED | SKIPPED | FALLBACK
    reason: str


class FallbackRegistry:
    """Tracks custom kernel eligibility decisions and runtime fallback events."""

    def __init__(self) -> None:
        self.entries: dict[str, KernelStatusEntry] = {}

    def record(self, name: str, status: str, reason: str) -> None:
        self.entries[name] = KernelStatusEntry(name=name, status=status, reason=reason)

    def clear(self) -> None:
        self.entries.clear()


def _detect_cuda_sm(torch_mod: Any) -> int:
    try:
        if not getattr(torch_mod, "cuda", None) or not torch_mod.cuda.is_available():
            return 0
        idx = int(torch_mod.cuda.current_device()) if hasattr(torch_mod.cuda, "current_device") else 0
        props = torch_mod.cuda.get_device_properties(idx)
        return int(props.major) * 10 + int(props.minor)
    except Exception:  # noqa: BLE001
        return 0


def _dtype_name(dtype: Any, torch_mod: Any) -> str:
    if dtype == getattr(torch_mod, "float32", None):
        return "fp32"
    if dtype == getattr(torch_mod, "float16", None):
        return "fp16"
    if dtype == getattr(torch_mod, "bfloat16", None):
        return "bf16"
    if dtype == getattr(torch_mod, "float64", None):
        return "fp64"
    return str(dtype)


def check_kernel_capability(
    spec: KernelSpec,
    torch_mod: Any,
    *,
    triton_available: bool | None = None,
    sm_override: int | None = None,
    dtype_name: str | None = None,
    allow_list: list[str] | None = None,
) -> tuple[bool, str]:
    """Evaluate whether `spec` is eligible on the current (or mocked) runtime."""
    if allow_list and spec.name not in allow_list:
        return False, f"{spec.name} excluded by kernels.allow_list"

    has_triton = HAS_TRITON if triton_available is None else triton_available
    if spec.triton_required and not has_triton:
        return False, "Triton unavailable; falling back to PyTorch native op"

    cuda_api = getattr(torch_mod, "cuda", None) if torch_mod is not None else None
    if sm_override is not None:
        cuda_ok = sm_override > 0
        sm_int = sm_override
    else:
        try:
            cuda_ok = bool(cuda_api and cuda_api.is_available())
        except Exception:  # noqa: BLE001
            cuda_ok = False
        sm_int = _detect_cuda_sm(torch_mod) if cuda_ok else 0

    if not cuda_ok:
        return False, "CUDA unavailable; falling back to PyTorch native op"

    if sm_int < spec.min_sm:
        return False, f"GPU sm_{sm_int} < min_sm {spec.min_sm}; falling back to PyTorch native op"

    if dtype_name is not None and dtype_name not in spec.supported_dtypes:
        return False, f"dtype {dtype_name} unsupported by {spec.name} ({spec.supported_dtypes})"

    return True, f"sm_{sm_int}, triton={has_triton}"


class KernelRegistry:
    """Registry holding available `KernelSpec` implementations and patching PyTorch targets."""

    def __init__(self) -> None:
        self.specs: dict[str, KernelSpec] = {}
        self.fallback_registry = FallbackRegistry()

    def register(self, spec: KernelSpec) -> None:
        self.specs[spec.name] = spec


def _builtin_specs() -> dict[str, KernelSpec]:
    return {
        "rmsnorm": KernelSpec(name="rmsnorm", min_sm=80, supported_dtypes=("fp32", "fp16", "bf16")),
        "layernorm": KernelSpec(name="layernorm", min_sm=80, supported_dtypes=("fp32", "fp16", "bf16")),
        "cross_entropy": KernelSpec(name="cross_entropy", min_sm=80, supported_dtypes=("fp32", "fp16", "bf16")),
        "swiglu": KernelSpec(name="swiglu", min_sm=80, supported_dtypes=("fp32", "fp16", "bf16")),
    }


def install_custom_kernels_hooks(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any,
    *,
    triton_available: bool | None = None,
    sm_override: int | None = None,
) -> KernelRegistry | None:
    """Install gated custom kernel wrappers on eligible models & `F.cross_entropy`."""
    global _KERNELS_INSTALLED  # noqa: PLW0603
    if _KERNELS_INSTALLED or torch_mod is None or not hasattr(torch_mod, "nn"):
        return None
    if not cfg.kernels.enabled and os.environ.get("PTOPT_FORCE_KERNELS") != "1":
        return None

    from src.kernels.cross_entropy import triton_cross_entropy
    from src.kernels.layernorm import triton_layernorm
    from src.kernels.rmsnorm import triton_rmsnorm
    from src.kernels.rope_or_swiglu import triton_swiglu

    reg = KernelRegistry()
    specs = _builtin_specs()
    eligible: dict[str, bool] = {}

    for name, spec in specs.items():
        reg.register(spec)
        ok, reason = check_kernel_capability(
            spec,
            torch_mod,
            triton_available=triton_available,
            sm_override=sm_override,
            allow_list=cfg.kernels.allow_list,
        )
        eligible[name] = ok
        status = "APPLIED" if ok else "SKIPPED"
        reg.fallback_registry.record(name, status, reason)
        manager.record_event(f"kernels.{name}", status, reason)

    if not any(eligible.values()):
        return reg

    # 1. Patch F.cross_entropy when eligible
    func_mod = getattr(torch_mod.nn, "functional", None)
    if eligible.get("cross_entropy") and func_mod is not None and hasattr(func_mod, "cross_entropy"):
        orig_ce = func_mod.cross_entropy

        @functools.wraps(orig_ce)
        def _custom_cross_entropy(
            inp: Any,
            target: Any,
            weight: Any = None,
            size_average: Any = None,
            ignore_index: int = -100,
            reduce: Any = None,
            reduction: str = "mean",
            label_smoothing: float = 0.0,
        ) -> Any:
            if (
                weight is None
                and size_average is None
                and reduce is None
                and label_smoothing == 0.0
                and getattr(inp, "is_cuda", False)
                and getattr(target, "is_cuda", False)
                and inp.dim() == 2
                and target.dim() == 1
                and _dtype_name(inp.dtype, torch_mod) in ("fp32", "fp16", "bf16")
            ):
                try:
                    return triton_cross_entropy(inp, target, ignore_index=ignore_index, reduction=reduction)
                except Exception as exc:  # noqa: BLE001
                    if cfg.kernels.fallback_on_error and not manager.strict:
                        reg.fallback_registry.record("cross_entropy", "FALLBACK", f"Runtime fallback: {exc}")
                        manager.record_event("kernels.cross_entropy", "FALLBACK", "Runtime fallback to native F.cross_entropy", str(exc))
                    else:
                        raise
            return orig_ce(
                inp,
                target,
                weight=weight,
                size_average=size_average,
                ignore_index=ignore_index,
                reduce=reduce,
                reduction=reduction,
                label_smoothing=label_smoothing,
            )

        func_mod._ptopt_kern_orig_cross_entropy = orig_ce
        func_mod.cross_entropy = _custom_cross_entropy

    # 2. Patch RMSNorm, LayerNorm, SwiGLUMLP submodules lazily on top-level forward
    nn_mod = torch_mod.nn.Module
    call_attr = "__call__"
    orig_mod_call = getattr(nn_mod, call_attr)
    depth = [0]
    ln_cls = getattr(torch_mod.nn, "LayerNorm", None)
    rms_cls = getattr(torch_mod.nn, "RMSNorm", None)

    def _patch_submodules(root_mod: Any) -> int:
        patched = 0
        if not hasattr(root_mod, "modules"):
            return 0
        for sub in root_mod.modules():
            if getattr(sub, "_ptopt_kern_patched", False):
                continue
            cls_lower = type(sub).__name__.lower()

            # RMSNorm (torch.nn.RMSNorm or custom RMSNorm with .weight & .eps)
            if eligible.get("rmsnorm") and (
                (rms_cls is not None and isinstance(sub, rms_cls)) or "rmsnorm" in cls_lower
            ):
                weight = getattr(sub, "weight", None)
                if weight is not None and getattr(weight, "is_cuda", False) and weight.dim() == 1:
                    orig_fwd = sub.forward

                    def _make_rms_fwd(m: Any, base_f: Any) -> Callable[..., Any]:
                        @functools.wraps(base_f)
                        def _rms_fwd(x: Any) -> Any:
                            if getattr(x, "is_cuda", False) and _dtype_name(x.dtype, torch_mod) in ("fp32", "fp16", "bf16"):
                                try:
                                    return triton_rmsnorm(x, m.weight, eps=float(getattr(m, "eps", 1e-6)))
                                except Exception:  # noqa: BLE001
                                    if not cfg.kernels.fallback_on_error or manager.strict:
                                        raise
                            return base_f(x)

                        return _rms_fwd

                    sub._ptopt_kern_orig_forward = orig_fwd
                    sub.forward = _make_rms_fwd(sub, orig_fwd)
                    sub._ptopt_kern_patched = True
                    patched += 1
                    continue

            # LayerNorm (standard 1D normalized_shape)
            if eligible.get("layernorm") and ln_cls is not None and isinstance(sub, ln_cls):
                norm_shape = getattr(sub, "normalized_shape", ())
                weight = getattr(sub, "weight", None)
                if len(norm_shape) == 1 and weight is not None and getattr(weight, "is_cuda", False):
                    orig_fwd = sub.forward

                    def _make_ln_fwd(m: Any, base_f: Any) -> Callable[..., Any]:
                        @functools.wraps(base_f)
                        def _ln_fwd(x: Any) -> Any:
                            if getattr(x, "is_cuda", False) and _dtype_name(x.dtype, torch_mod) in ("fp32", "fp16", "bf16"):
                                try:
                                    return triton_layernorm(x, m.weight, getattr(m, "bias", None), eps=float(m.eps))
                                except Exception:  # noqa: BLE001
                                    if not cfg.kernels.fallback_on_error or manager.strict:
                                        raise
                            return base_f(x)

                        return _ln_fwd

                    sub._ptopt_kern_orig_forward = orig_fwd
                    sub.forward = _make_ln_fwd(sub, orig_fwd)
                    sub._ptopt_kern_patched = True
                    patched += 1
                    continue

            # SwiGLUMLP (`w_gate`, `w_up`, `w_down` matching LLaMA / TinyGPT MLP)
            if (
                eligible.get("swiglu")
                and hasattr(sub, "w_gate")
                and hasattr(sub, "w_up")
                and hasattr(sub, "w_down")
            ):
                gate_w = getattr(sub.w_gate, "weight", None)
                if gate_w is not None and getattr(gate_w, "is_cuda", False):
                    orig_fwd = sub.forward

                    def _make_swi_fwd(m: Any, base_f: Any) -> Callable[..., Any]:
                        @functools.wraps(base_f)
                        def _swi_fwd(x: Any) -> Any:
                            if getattr(x, "is_cuda", False) and _dtype_name(x.dtype, torch_mod) in ("fp32", "fp16", "bf16"):
                                try:
                                    return m.w_down(triton_swiglu(m.w_gate(x), m.w_up(x)))
                                except Exception:  # noqa: BLE001
                                    if not cfg.kernels.fallback_on_error or manager.strict:
                                        raise
                            return base_f(x)

                        return _swi_fwd

                    sub._ptopt_kern_orig_forward = orig_fwd
                    sub.forward = _make_swi_fwd(sub, orig_fwd)
                    sub._ptopt_kern_patched = True
                    patched += 1

        return patched

    @functools.wraps(orig_mod_call)
    def _kernels_mod_call(mod_self: Any, *args: Any, **kwargs: Any) -> Any:
        depth[0] += 1
        try:
            if depth[0] == 1 and not getattr(mod_self, "_ptopt_kern_scanned", False):
                mod_self._ptopt_kern_scanned = True
                n = _patch_submodules(mod_self)
                if n > 0:
                    manager.record_event(
                        "kernels.patch",
                        "APPLIED",
                        f"Patched {n} submodules in {type(mod_self).__name__} with custom Triton kernels",
                    )
            return orig_mod_call(mod_self, *args, **kwargs)
        finally:
            depth[0] -= 1

    nn_mod._ptopt_kern_call_attr = call_attr
    nn_mod._ptopt_kern_orig_call = orig_mod_call
    setattr(nn_mod, call_attr, _kernels_mod_call)
    _KERNELS_INSTALLED = True
    torch_mod._ptopt_kernel_registry = reg
    return reg


def uninstall_custom_kernels_hooks(torch_mod: Any) -> None:
    """Restore patched `nn.Module.__call__` and `F.cross_entropy` hooks (for unit tests)."""
    global _KERNELS_INSTALLED  # noqa: PLW0603
    if not _KERNELS_INSTALLED or torch_mod is None:
        return
    nn_mod = getattr(torch_mod, "nn", None) and torch_mod.nn.Module
    if nn_mod is not None and hasattr(nn_mod, "_ptopt_kern_orig_call"):
        setattr(nn_mod, getattr(nn_mod, "_ptopt_kern_call_attr"), getattr(nn_mod, "_ptopt_kern_orig_call"))
        delattr(nn_mod, "_ptopt_kern_call_attr")
        delattr(nn_mod, "_ptopt_kern_orig_call")
    func_mod = getattr(getattr(torch_mod, "nn", None), "functional", None)
    if func_mod is not None and hasattr(func_mod, "_ptopt_kern_orig_cross_entropy"):
        func_mod.cross_entropy = func_mod._ptopt_kern_orig_cross_entropy
        delattr(func_mod, "_ptopt_kern_orig_cross_entropy")
    if hasattr(torch_mod, "_ptopt_kernel_registry"):
        delattr(torch_mod, "_ptopt_kernel_registry")
    _KERNELS_INSTALLED = False


def apply_custom_kernels_pass(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any = None,
) -> str:
    """Enable gated custom Triton kernels when `cfg.kernels.enabled` is True."""
    if not cfg.kernels.enabled and os.environ.get("PTOPT_FORCE_KERNELS") != "1":
        return "SKIPPED: kernels.enabled is False"
    tmod = torch_mod if torch_mod is not None else sys.modules.get("torch")
    if tmod is None:
        return "SKIPPED: torch not loaded"
    reg = install_custom_kernels_hooks(cfg, manager, tmod)
    if reg is None:
        return "SKIPPED: custom kernels already active"
    applied = [k for k, v in reg.fallback_registry.entries.items() if v.status == "APPLIED"]
    if not applied:
        return "SKIPPED: no kernels passed capability gate (fallback to PyTorch native)"
    return f"Custom kernels active: {', '.join(applied)}"
