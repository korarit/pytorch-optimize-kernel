"""Kernel capability specs, FallbackRegistry tracking, and zero-code PyTorch auto-patcher."""

from __future__ import annotations

import functools
import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from src.config import OptimizeConfig
from src.hooks.module_dispatch import register_module_call_handler, unregister_module_call_handler

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
        "rope": KernelSpec(name="rope", min_sm=80, supported_dtypes=("fp32", "fp16", "bf16")),
        "fused_linear_cross_entropy": KernelSpec(
            name="fused_linear_cross_entropy", min_sm=80, supported_dtypes=("fp32", "fp16", "bf16")
        ),
    }


def _match_swiglu_mlp(sub: Any, torch_mod: Any) -> tuple[Any, Any, Any] | None:
    """Strictly verify SwiGLU MLP semantics before patching (Guardrail #1).

    Supports `(w_gate, w_up, w_down)` and HuggingFace `(gate_proj, up_proj, down_proj)`,
    and rejects GeGLU/ReGLU (`GELU`/`ReLU`) or MLPs with unexpected child layers (e.g. `Dropout`).
    """
    nn_ns = getattr(torch_mod, "nn", None)
    lin_cls = getattr(nn_ns, "Linear", None)
    silu_cls = getattr(nn_ns, "SiLU", None)
    if lin_cls is None:
        return None

    if hasattr(sub, "w_gate") and hasattr(sub, "w_up") and hasattr(sub, "w_down"):
        names = ("w_gate", "w_up", "w_down")
    elif hasattr(sub, "gate_proj") and hasattr(sub, "up_proj") and hasattr(sub, "down_proj"):
        names = ("gate_proj", "up_proj", "down_proj")
    else:
        return None

    gate_m, up_m, down_m = getattr(sub, names[0]), getattr(sub, names[1]), getattr(sub, names[2])
    if not (isinstance(gate_m, lin_cls) and isinstance(up_m, lin_cls) and isinstance(down_m, lin_cls)):
        return None

    if (
        getattr(gate_m, "in_features", -1) != getattr(up_m, "in_features", -2)
        or getattr(gate_m, "out_features", -1) != getattr(up_m, "out_features", -2)
        or getattr(gate_m, "out_features", -1) != getattr(down_m, "in_features", -2)
        or getattr(gate_m, "in_features", -1) != getattr(down_m, "out_features", -2)
    ):
        return None

    if hasattr(sub, "named_children"):
        allowed_children = set(names) | {"act_fn", "act", "activation"}
        for child_name, _child_mod in sub.named_children():
            if child_name not in allowed_children:
                return None

    act = getattr(sub, "act_fn", getattr(sub, "act", getattr(sub, "activation", None)))
    if act is not None:
        is_silu = (silu_cls is not None and isinstance(act, silu_cls)) or (
            type(act).__name__.lower() in ("silu", "siluactivation", "swish")
        ) or (getattr(act, "__name__", "").lower() == "silu")
        if not is_silu:
            return None
    else:
        cls_lower = type(sub).__name__.lower()
        if any(bad in cls_lower for bad in ("geglu", "gelu", "reglu", "relu")):
            return None
        has_swiglu_hint = "swiglu" in cls_lower
        if not has_swiglu_hint:
            try:
                src = inspect.getsource(sub.forward)
                has_swiglu_hint = "silu" in src.lower()
            except Exception:  # noqa: BLE001
                has_swiglu_hint = False
        if not has_swiglu_hint:
            return None

    return gate_m, up_m, down_m


def _is_verified_rope_inputs(q: Any, k: Any, cos: Any, sin: Any) -> bool:
    """Strictly validate half-split RoPE tensor layout & shapes (Guardrail #1)."""
    if not (
        getattr(q, "is_cuda", False)
        and getattr(k, "is_cuda", False)
        and getattr(cos, "is_cuda", False)
        and getattr(sin, "is_cuda", False)
    ):
        return False
    if q.dim() not in (3, 4) or k.dim() != q.dim():
        return False
    d = int(q.shape[-1])
    if d <= 0 or d % 2 != 0 or int(k.shape[-1]) != d or int(k.shape[-2]) != int(q.shape[-2]):
        return False
    half = d // 2
    if int(cos.shape[-1]) not in (d, half) or int(sin.shape[-1]) != int(cos.shape[-1]):
        return False
    # Verify half-split layout ([c1, c1], NOT interleaved [c0, c0, c1, c1]) when full D is passed
    if int(cos.shape[-1]) == d and not bool((cos[..., :half] == cos[..., half:]).all()):
        return False
    return True


def _is_checkpoint_control_flow(exc: BaseException) -> bool:
    """Propagate PyTorch `checkpoint(..., early_stop=True)` internal stop signals untouched."""
    return type(exc).__name__ == "_StopRecomputationError"


def _patch_functional_cross_entropy(
    torch_mod: Any,
    reg: KernelRegistry,
    manager: "HookManager",
    *,
    enable_ce: bool,
    enable_fused_lce: bool,
    compile_active: bool,
    fallback_on_error: bool,
) -> bool:
    """Install idempotent `F.cross_entropy` hook for Triton CE and LazyLinearLogits."""
    func_mod = getattr(getattr(torch_mod, "nn", None), "functional", None)
    if func_mod is None or not hasattr(func_mod, "cross_entropy"):
        return False
    if hasattr(func_mod, "_ptopt_kern_orig_cross_entropy"):
        return False

    from src.kernels.cross_entropy import triton_cross_entropy
    from src.kernels.fused_linear_cross_entropy import fused_linear_cross_entropy

    orig_ce = func_mod.cross_entropy
    ce_min_elems = 8_388_608 if compile_active else 0

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
        if hasattr(inp, "_ptopt_lazy_linear"):
            if (
                weight is None
                and size_average is None
                and reduce is None
                and label_smoothing == 0.0
                and reduction in ("mean", "sum", "none")
            ):
                try:
                    h, lin_w, lin_b, _out_shape, mm_dt = inp._ptopt_lazy_linear
                    return fused_linear_cross_entropy(
                        h,
                        lin_w,
                        target,
                        bias=lin_b,
                        ignore_index=ignore_index,
                        reduction=reduction,
                        matmul_dtype=mm_dt,
                    )
                except Exception as exc:  # noqa: BLE001
                    if _is_checkpoint_control_flow(exc):
                        raise
                    if fallback_on_error and not manager.strict:
                        reg.fallback_registry.record("fused_linear_cross_entropy", "FALLBACK", f"Runtime fallback: {exc}")
                        manager.record_event("kernels.fused_linear_cross_entropy", "FALLBACK", "Fallback to native F.cross_entropy", str(exc))
                        inp = inp.materialize()
                    else:
                        raise
            else:
                inp = inp.materialize()

        if (
            enable_ce
            and weight is None
            and size_average is None
            and reduce is None
            and label_smoothing == 0.0
            and getattr(inp, "is_cuda", False)
            and getattr(target, "is_cuda", False)
            and inp.dim() == 2
            and target.dim() == 1
            and _dtype_name(inp.dtype, torch_mod) in ("fp32", "fp16", "bf16")
            and (not compile_active or inp.numel() >= ce_min_elems or int(inp.shape[-1]) >= 4096)
        ):
            try:
                return triton_cross_entropy(inp, target, ignore_index=ignore_index, reduction=reduction)
            except Exception as exc:  # noqa: BLE001
                if _is_checkpoint_control_flow(exc):
                    raise
                if fallback_on_error and not manager.strict:
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
    return True


def _patch_module_submodules(
    root_mod: Any,
    torch_mod: Any,
    eligible: dict[str, bool],
    manager: "HookManager",
    *,
    min_lce_vocab: int,
    fallback_on_error: bool,
    hybrid_compile_lce: bool = False,
) -> int:
    """Patch eligible RMSNorm, LayerNorm, SwiGLU, and terminal large-vocab Linear head submodules."""
    if not hasattr(root_mod, "modules"):
        return 0

    from src.kernels.fused_linear_cross_entropy import create_lazy_linear_logits
    from src.kernels.layernorm import triton_layernorm
    from src.kernels.rmsnorm import triton_rmsnorm
    from src.kernels.rope_or_swiglu import triton_rope, triton_swiglu

    nn_ns = getattr(torch_mod, "nn", None)
    ln_cls = getattr(nn_ns, "LayerNorm", None)
    rms_cls = getattr(nn_ns, "RMSNorm", None)
    lin_cls = getattr(nn_ns, "Linear", None)
    compiler_ns = getattr(torch_mod, "compiler", None)
    is_compiling_fn = getattr(compiler_ns, "is_compiling", None)
    compiler_disable = getattr(compiler_ns, "disable", None)

    all_mods = list(root_mod.modules())
    last_linear = None
    if lin_cls is not None:
        linear_mods = [m for m in all_mods if isinstance(m, lin_cls)]
        last_linear = linear_mods[-1] if linear_mods else None

    patched = 0
    for sub in all_mods:
        if getattr(sub, "_ptopt_kern_patched", False):
            continue
        cls_lower = type(sub).__name__.lower()

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
                            except Exception as exc:  # noqa: BLE001
                                if _is_checkpoint_control_flow(exc) or not fallback_on_error or manager.strict:
                                    raise
                        return base_f(x)

                    return _rms_fwd

                sub._ptopt_kern_orig_forward = orig_fwd
                sub.forward = _make_rms_fwd(sub, orig_fwd)
                sub._ptopt_kern_patched = True
                patched += 1
                continue

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
                            except Exception as exc:  # noqa: BLE001
                                if _is_checkpoint_control_flow(exc) or not fallback_on_error or manager.strict:
                                    raise
                        return base_f(x)

                    return _ln_fwd

                sub._ptopt_kern_orig_forward = orig_fwd
                sub.forward = _make_ln_fwd(sub, orig_fwd)
                sub._ptopt_kern_patched = True
                patched += 1
                continue

        swiglu_projs = _match_swiglu_mlp(sub, torch_mod) if eligible.get("swiglu") else None
        if swiglu_projs is not None:
            gate_m, up_m, down_m = swiglu_projs
            gate_w = getattr(gate_m, "weight", None)
            if gate_w is not None and getattr(gate_w, "is_cuda", False):
                orig_fwd = sub.forward

                def _make_swi_fwd(
                    m: Any, g_proj: Any, u_proj: Any, d_proj: Any, base_f: Any
                ) -> Callable[..., Any]:
                    @functools.wraps(base_f)
                    def _swi_fwd(x: Any) -> Any:
                        if getattr(x, "is_cuda", False) and _dtype_name(x.dtype, torch_mod) in ("fp32", "fp16", "bf16"):
                            try:
                                return d_proj(triton_swiglu(g_proj(x), u_proj(x)))
                            except Exception as exc:  # noqa: BLE001
                                if _is_checkpoint_control_flow(exc) or not fallback_on_error or manager.strict:
                                    raise
                        return base_f(x)

                    return _swi_fwd

                sub._ptopt_kern_orig_forward = orig_fwd
                sub.forward = _make_swi_fwd(sub, gate_m, up_m, down_m, orig_fwd)
                sub._ptopt_kern_patched = True
                patched += 1
                continue

        # Verified RoPE method hook (only when module explicitly exposes a 4-arg apply_rotary_pos_emb)
        if eligible.get("rope") and callable(getattr(sub, "apply_rotary_pos_emb", None)):
            orig_rope = sub.apply_rotary_pos_emb

            def _make_rope_fn(m: Any, base_rope: Any) -> Callable[..., Any]:
                @functools.wraps(base_rope)
                def _rope_wrap(q: Any, k: Any, cos: Any, sin: Any, *extra_args: Any, **extra_kw: Any) -> Any:
                    if not extra_args and not extra_kw:
                        verified = getattr(m, "_ptopt_rope_layout_ok", None)
                        if verified is None:
                            verified = _is_verified_rope_inputs(q, k, cos, sin)
                            m._ptopt_rope_layout_ok = verified
                        if verified and getattr(q, "is_cuda", False) and getattr(k, "is_cuda", False):
                            try:
                                return triton_rope(q, k, cos, sin)
                            except Exception as exc:  # noqa: BLE001
                                if _is_checkpoint_control_flow(exc) or not fallback_on_error or manager.strict:
                                    raise
                    return base_rope(q, k, cos, sin, *extra_args, **extra_kw)

                return _rope_wrap

            sub._ptopt_kern_orig_rope = orig_rope
            sub.apply_rotary_pos_emb = _make_rope_fn(sub, orig_rope)
            sub._ptopt_kern_patched = True
            patched += 1
            continue

        # Large-vocab terminal/head `nn.Linear` -> deferred LazyLinearLogits for FusedLinearCrossEntropy
        out_feat = int(getattr(sub, "out_features", 0))
        in_feat = int(getattr(sub, "in_features", 0))
        is_head_shape = (sub is last_linear) or (in_feat > 0 and out_feat > in_feat * 4)
        if (
            eligible.get("fused_linear_cross_entropy")
            and lin_cls is not None
            and isinstance(sub, lin_cls)
            and out_feat >= min_lce_vocab
            and is_head_shape
        ):
            weight = getattr(sub, "weight", None)
            if weight is not None and getattr(weight, "is_cuda", False):
                orig_fwd = sub.forward

                def _make_lce_head_fwd(m: Any, base_f: Any) -> Callable[..., Any]:
                    @functools.wraps(base_f)
                    def _lce_head_fwd(x: Any) -> Any:
                        compiling = False if hybrid_compile_lce else (
                            bool(is_compiling_fn()) if callable(is_compiling_fn) else False
                        )
                        if (
                            not compiling
                            and getattr(m, "training", False)
                            and getattr(torch_mod, "is_grad_enabled", lambda: False)()
                            and getattr(x, "is_cuda", False)
                            and x.dim() >= 2
                            and x.numel() > 0
                            and _dtype_name(x.dtype, torch_mod) in ("fp32", "fp16", "bf16")
                        ):
                            try:
                                return create_lazy_linear_logits(x, m.weight, getattr(m, "bias", None))
                            except Exception as exc:  # noqa: BLE001
                                if _is_checkpoint_control_flow(exc) or not fallback_on_error or manager.strict:
                                    raise
                        return base_f(x)

                    if hybrid_compile_lce and callable(compiler_disable):
                        return compiler_disable(_lce_head_fwd)
                    return _lce_head_fwd

                sub._ptopt_kern_orig_forward = orig_fwd
                sub.forward = _make_lce_head_fwd(sub, orig_fwd)
                sub._ptopt_kern_patched = True
                patched += 1

    return patched


def enable_custom_kernels_on_module(
    root_mod: Any,
    torch_mod: Any,
    manager: "HookManager",
    *,
    triton_available: bool | None = None,
    sm_override: int | None = None,
) -> int:
    """Activate custom Triton kernels + FusedLinearCE directly on an eager/act-ckpt module."""
    if root_mod is None or torch_mod is None or not hasattr(torch_mod, "nn"):
        return 0
    specs = _builtin_specs()
    eligible: dict[str, bool] = {}
    for name, spec in specs.items():
        ok, _ = check_kernel_capability(
            spec,
            torch_mod,
            triton_available=triton_available,
            sm_override=sm_override,
        )
        eligible[name] = ok

    if not any(eligible.values()):
        return 0

    reg = getattr(torch_mod, "_ptopt_kernel_registry", None)
    if reg is None:
        reg = KernelRegistry()
        for spec in specs.values():
            reg.register(spec)
        torch_mod._ptopt_kernel_registry = reg

    if eligible.get("cross_entropy") or eligible.get("fused_linear_cross_entropy"):
        _patch_functional_cross_entropy(
            torch_mod,
            reg,
            manager,
            enable_ce=bool(eligible.get("cross_entropy")),
            enable_fused_lce=bool(eligible.get("fused_linear_cross_entropy")),
            compile_active=False,
            fallback_on_error=True,
        )

    try:
        min_lce_vocab = int(os.environ.get("PTOPT_FUSED_LCE_MIN_VOCAB", "4096"))
    except ValueError:
        min_lce_vocab = 4096

    n = _patch_module_submodules(
        root_mod,
        torch_mod,
        eligible,
        manager,
        min_lce_vocab=min_lce_vocab,
        fallback_on_error=True,
        hybrid_compile_lce=False,
    )
    if n > 0:
        manager.record_event(
            "kernels.act_ckpt_fallback",
            "APPLIED",
            f"Auto-enabled {n} custom Triton kernels on {type(root_mod).__name__} after compile bypass",
        )
    return n


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
    hybrid_lce_env = os.environ.get("PTOPT_HYBRID_LCE", "1") == "1"
    hybrid_balanced_only = (
        not cfg.kernels.enabled
        and os.environ.get("PTOPT_FORCE_KERNELS") != "1"
        and cfg.profile == "balanced"
        and cfg.compile.enabled
        and hybrid_lce_env
    )
    if not cfg.kernels.enabled and os.environ.get("PTOPT_FORCE_KERNELS") != "1" and not hybrid_balanced_only:
        return None

    reg = KernelRegistry()
    specs = _builtin_specs()
    eligible: dict[str, bool] = {}
    force_all = os.environ.get("PTOPT_FORCE_ALL_KERNELS") == "1"
    compile_active = bool(cfg.compile.enabled) and not force_all and not cfg.kernels.allow_list
    hybrid_lce_active = compile_active and hybrid_lce_env

    for name, spec in specs.items():
        reg.register(spec)
        if hybrid_balanced_only and name != "fused_linear_cross_entropy":
            eligible[name] = False
            reg.fallback_registry.record(name, "SKIPPED", "balanced hybrid mode enables fused_linear_cross_entropy only")
            continue

        ok, reason = check_kernel_capability(
            spec,
            torch_mod,
            triton_available=triton_available,
            sm_override=sm_override,
            allow_list=cfg.kernels.allow_list,
        )
        # Defer elementwise/norm modules to TorchInductor when compile is active,
        # but keep cross_entropy and Hybrid FusedLinearCrossEntropy active for large-vocab heads.
        if ok and compile_active and name in ("rmsnorm", "layernorm", "swiglu", "rope"):
            ok = False
            reason = f"{reason}; deferred to torch.compile Inductor fusion"
        elif ok and compile_active and name == "fused_linear_cross_entropy" and not hybrid_lce_active:
            ok = False
            reason = f"{reason}; deferred to torch.compile Inductor fusion"
        elif ok and hybrid_lce_active and name == "fused_linear_cross_entropy":
            reason = f"{reason}; hybrid compile+FusedLCE head enabled"
        eligible[name] = ok
        status = "APPLIED" if ok else "SKIPPED"
        reg.fallback_registry.record(name, status, reason)
        manager.record_event(f"kernels.{name}", status, reason)

    if not any(eligible.values()):
        return reg

    # 1. Patch F.cross_entropy when eligible (or when fused_linear_cross_entropy defers linear logits)
    if eligible.get("cross_entropy") or eligible.get("fused_linear_cross_entropy"):
        _patch_functional_cross_entropy(
            torch_mod,
            reg,
            manager,
            enable_ce=bool(eligible.get("cross_entropy")),
            enable_fused_lce=bool(eligible.get("fused_linear_cross_entropy")),
            compile_active=compile_active,
            fallback_on_error=cfg.kernels.fallback_on_error,
        )

    default_min_vocab = "8192" if hybrid_balanced_only else "4096"
    try:
        min_lce_vocab = int(os.environ.get("PTOPT_FUSED_LCE_MIN_VOCAB", default_min_vocab))
    except ValueError:
        min_lce_vocab = int(default_min_vocab)

    def _kernels_mod_call(mod_self: Any, args: tuple[Any, ...], kwargs: dict[str, Any], next_fn: Any) -> Any:
        if not getattr(mod_self, "_ptopt_kern_scanned", False):
            mod_self._ptopt_kern_scanned = True
            n = _patch_module_submodules(
                mod_self,
                torch_mod,
                eligible,
                manager,
                min_lce_vocab=min_lce_vocab,
                fallback_on_error=cfg.kernels.fallback_on_error,
                hybrid_compile_lce=hybrid_lce_active,
            )
            if n > 0:
                manager.record_event(
                    "kernels.patch",
                    "APPLIED",
                    f"Patched {n} submodules in {type(mod_self).__name__} with custom Triton kernels",
                )
        return next_fn(mod_self, *args, **kwargs)

    register_module_call_handler(torch_mod, "kernels.auto", 10, _kernels_mod_call)
    _KERNELS_INSTALLED = True
    torch_mod._ptopt_kernel_registry = reg
    return reg


def uninstall_custom_kernels_hooks(torch_mod: Any) -> None:
    """Restore patched `nn.Module.__call__` and `F.cross_entropy` hooks (for unit tests)."""
    global _KERNELS_INSTALLED  # noqa: PLW0603
    if not _KERNELS_INSTALLED or torch_mod is None:
        return
    unregister_module_call_handler(torch_mod, "kernels.auto")
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
    """Enable gated custom Triton kernels or balanced Hybrid FusedLinearCE when eligible."""
    tmod = torch_mod if torch_mod is not None else sys.modules.get("torch")
    hybrid_lce_env = os.environ.get("PTOPT_HYBRID_LCE", "1") == "1"
    hybrid_balanced_eligible = (
        not cfg.kernels.enabled
        and os.environ.get("PTOPT_FORCE_KERNELS") != "1"
        and cfg.profile == "balanced"
        and cfg.compile.enabled
        and hybrid_lce_env
        and tmod is not None
        and check_kernel_capability(_builtin_specs()["fused_linear_cross_entropy"], tmod)[0]
    )
    if not cfg.kernels.enabled and os.environ.get("PTOPT_FORCE_KERNELS") != "1" and not hybrid_balanced_eligible:
        return "SKIPPED: kernels.enabled is False"
    if tmod is None:
        return "SKIPPED: torch not loaded"
    reg = install_custom_kernels_hooks(cfg, manager, tmod)
    if reg is None:
        return "SKIPPED: custom kernels already active"
    applied = [k for k, v in reg.fallback_registry.entries.items() if v.status == "APPLIED"]
    if not applied:
        return "SKIPPED: no kernels passed capability gate (fallback to PyTorch native)"
    return f"Custom kernels active: {', '.join(applied)}"
