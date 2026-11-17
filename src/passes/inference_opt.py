"""Pass 3.6: Safe Inference Optimization (`inference_mode()`, CUDA Graphs / reduce-overhead, INT8 & FP8)."""

from __future__ import annotations

import functools
import os
import sys
import warnings
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from src.config import OptimizeConfig

if TYPE_CHECKING:
    from src.hooks.manager import HookManager

_INFERENCE_OPT_INSTALLED = False


def _detect_cuda_sm(torch_mod: Any) -> int:
    try:
        if not getattr(torch_mod, "cuda", None) or not torch_mod.cuda.is_available():
            return 0
        idx = int(torch_mod.cuda.current_device()) if hasattr(torch_mod.cuda, "current_device") else 0
        props = torch_mod.cuda.get_device_properties(idx)
        return int(props.major) * 10 + int(props.minor)
    except Exception:  # noqa: BLE001
        return 0


def _has_cuda_context(mod_self: Any, args: tuple[Any, ...], kwargs: dict[str, Any], torch_mod: Any) -> bool:
    tensor_cls = getattr(torch_mod, "Tensor", None)
    if tensor_cls is not None:
        for item in (*args, *kwargs.values()):
            if isinstance(item, tensor_cls) and getattr(item, "is_cuda", False):
                return True
    if hasattr(mod_self, "parameters"):
        try:
            first_param = next(mod_self.parameters(), None)
            if first_param is not None and getattr(first_param, "is_cuda", False):
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def _has_requires_grad_tensor(args: tuple[Any, ...], kwargs: dict[str, Any], torch_mod: Any) -> bool:
    tensor_cls = getattr(torch_mod, "Tensor", None)
    if tensor_cls is None:
        return False
    for item in (*args, *kwargs.values()):
        if isinstance(item, tensor_cls) and getattr(item, "requires_grad", False):
            return True
    return False


def _shape_signature(args: tuple[Any, ...], kwargs: dict[str, Any], torch_mod: Any) -> tuple[Any, ...]:
    tensor_cls = getattr(torch_mod, "Tensor", None)
    sig: list[Any] = []
    for item in (*args, *kwargs.values()):
        if tensor_cls is not None and isinstance(item, tensor_cls):
            sig.append((tuple(item.shape), str(item.dtype)))
    return tuple(sig)


def check_fp8_eligibility(torch_mod: Any) -> tuple[bool, str]:
    """Return `(eligible, reason)` for Ada/Hopper FP8 `_scaled_mm` inference path."""
    if torch_mod is None:
        return False, "torch module unavailable"
    cuda_api = getattr(torch_mod, "cuda", None)
    try:
        cuda_ok = bool(cuda_api and cuda_api.is_available())
    except Exception:  # noqa: BLE001
        cuda_ok = False
    if not cuda_ok:
        return False, "CUDA unavailable (FP8 requires Ada/Hopper GPU sm>=89)"
    sm_int = _detect_cuda_sm(torch_mod)
    if sm_int < 89:
        return False, f"sm_{sm_int} < 89 (FP8 requires Ada/Hopper sm>=89)"
    if not hasattr(torch_mod, "float8_e4m3fn") or not callable(getattr(torch_mod, "_scaled_mm", None)):
        return False, "torch.float8_e4m3fn or torch._scaled_mm missing in this PyTorch build"
    return True, f"sm_{sm_int} eligible for FP8 inference"


def check_dynamic_quant_eligibility(torch_mod: Any) -> tuple[bool, str]:
    """Return `(eligible, reason)` for CPU INT8 dynamic quantization."""
    if torch_mod is None:
        return False, "torch module unavailable"
    ao = getattr(torch_mod, "ao", None)
    qmod = getattr(ao, "quantization", None)
    if qmod is None or not callable(getattr(qmod, "quantize_dynamic", None)):
        return False, "torch.ao.quantization.quantize_dynamic unavailable"
    return True, "CPU INT8 dynamic quantization available"


_TRITON_FP8_QUANTIZER: Any = None
_TRITON_FP8_INIT_DONE = False


def _get_triton_fp8_quantizer() -> Any:
    """Lazily compile fused Triton FP8 activation amax + clamp/cast quantizer (zero top-level imports)."""
    global _TRITON_FP8_QUANTIZER, _TRITON_FP8_INIT_DONE  # noqa: PLW0603
    if _TRITON_FP8_INIT_DONE:
        return _TRITON_FP8_QUANTIZER
    _TRITON_FP8_INIT_DONE = True
    try:
        import importlib.util

        if importlib.util.find_spec("triton") is None:
            return None
        import torch as _torch
        import triton  # type: ignore[import-not-found]
        import triton.language as tl  # type: ignore[import-not-found]

        @triton.jit
        def _fp8_partial_amax_kernel(X_ptr, Partial_ptr, N_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N_elements
            x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            tl.store(Partial_ptr + pid, tl.max(tl.abs(x), axis=0))

        @triton.jit
        def _fp8_quant_scale_kernel(X_ptr, InvScale_ptr, Out_ptr, N_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N_elements
            inv_scale = tl.load(InvScale_ptr).to(tl.float32)
            x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            q = tl.minimum(tl.maximum(x * inv_scale, -448.0), 448.0)
            tl.store(Out_ptr + offs, q.to(Out_ptr.dtype.element_ty), mask=mask)

        def _quantize_fp8_2d(flat: Any, fp8_dtype: Any) -> tuple[Any, Any]:
            flat_c = flat.contiguous()
            n_elems = flat_c.numel()
            block_size = 4096
            n_blocks = triton.cdiv(n_elems, block_size)
            partial = _torch.empty((n_blocks,), dtype=_torch.float32, device=flat_c.device)
            _fp8_partial_amax_kernel[(n_blocks,)](
                flat_c, partial, n_elems, BLOCK_SIZE=block_size, num_warps=8
            )
            amax = partial.max().clamp(min=1e-12)
            x_scale = amax / 448.0
            inv_scale = 448.0 / amax
            x_q = _torch.empty(flat_c.shape, dtype=fp8_dtype, device=flat_c.device)
            _fp8_quant_scale_kernel[(n_blocks,)](
                flat_c, inv_scale, x_q, n_elems, BLOCK_SIZE=block_size, num_warps=8
            )
            return x_q, x_scale

        _TRITON_FP8_QUANTIZER = _quantize_fp8_2d
    except Exception:  # noqa: BLE001
        _TRITON_FP8_QUANTIZER = None
    return _TRITON_FP8_QUANTIZER


def apply_fp8_linear_eval_hooks(module: Any, torch_mod: Any) -> int:
    """Wrap eligible `nn.Linear` submodules on CUDA sm>=89 to use FP8 `_scaled_mm` during `.eval()`."""
    nn_ns = getattr(torch_mod, "nn", None)
    linear_cls = getattr(nn_ns, "Linear", None)
    func_ns = getattr(nn_ns, "functional", None)
    if linear_cls is None or func_ns is None or not hasattr(module, "modules"):
        return 0

    fp8_dtype = getattr(torch_mod, "float8_e4m3fn", None)
    bf16_dtype = getattr(torch_mod, "bfloat16", None)
    fp16_dtype = getattr(torch_mod, "float16", bf16_dtype)
    fp32_dtype = getattr(torch_mod, "float32", None)
    if fp8_dtype is None or bf16_dtype is None or fp32_dtype is None:
        return 0

    triton_quant = _get_triton_fp8_quantizer()

    def _build_weight_cache(lin: Any) -> tuple[Any, ...]:
        w_src = lin.weight.detach()
        w_amax = w_src.abs().amax().clamp(min=1e-12)
        w_scale = (w_amax / 448.0).to(dtype=fp32_dtype, device=w_src.device)
        w_inv_scale = (448.0 / w_amax).to(dtype=fp32_dtype, device=w_src.device)
        w_q = (w_src.to(fp32_dtype) * w_inv_scale).clamp(-448.0, 448.0).to(fp8_dtype)
        w_t = w_q.t()
        bias_bf16 = lin.bias.detach().to(bf16_dtype) if lin.bias is not None else None
        bias_fp16 = lin.bias.detach().to(fp16_dtype) if (lin.bias is not None and fp16_dtype != bf16_dtype) else bias_bf16
        return (lin.weight._version, w_t, w_scale, bias_bf16, bias_fp16)

    patched = 0
    for sub in module.modules():
        if not isinstance(sub, linear_cls) or getattr(sub, "_ptopt_fp8_patched", False):
            continue
        if sub.in_features < 128 or sub.in_features % 16 != 0 or sub.out_features % 16 != 0:
            continue
        if not getattr(sub.weight, "is_cuda", False):
            continue

        # Pre-quantize FP8 weights at patch time while keeping `sub.weight` & `state_dict()` untouched
        try:
            sub._ptopt_fp8_w_cache = _build_weight_cache(sub)
        except Exception:  # noqa: BLE001
            pass

        orig_fwd = sub.forward

        def _make_fp8_fwd(lin: Any, base_fwd: Any) -> Any:
            @functools.wraps(base_fwd)
            def _fp8_forward(inp: Any) -> Any:
                if getattr(lin, "training", True) or not getattr(inp, "is_cuda", False):
                    return base_fwd(inp)
                if inp.dim() < 2 or inp.shape[-1] != lin.in_features:
                    return base_fwd(inp)

                flat = inp.reshape(-1, lin.in_features)
                m_rows = int(flat.shape[0])
                # Dynamic per-tensor FP8 quantize overhead hurts small M (<512) and loses precision (E4M3)
                if m_rows < 512 or m_rows % 16 != 0:
                    return base_fwd(inp)

                try:
                    w_cache = getattr(lin, "_ptopt_fp8_w_cache", None)
                    if w_cache is None or w_cache[0] != lin.weight._version:
                        w_cache = _build_weight_cache(lin)
                        lin._ptopt_fp8_w_cache = w_cache
                    _, w_t, w_scale, bias_bf16, bias_fp16 = w_cache

                    nonlocal triton_quant
                    if triton_quant is not None:
                        try:
                            x_q, x_scale = triton_quant(flat, fp8_dtype)
                        except Exception:  # noqa: BLE001
                            triton_quant = None
                            x_amax = flat.abs().amax().clamp(min=1e-12)
                            x_scale = (x_amax / 448.0).to(dtype=fp32_dtype, device=flat.device)
                            x_q = (flat * (448.0 / x_amax)).clamp(-448.0, 448.0).to(fp8_dtype).contiguous()
                    else:
                        x_amax = flat.abs().amax().clamp(min=1e-12)
                        x_scale = (x_amax / 448.0).to(dtype=fp32_dtype, device=flat.device)
                        x_q = (flat * (448.0 / x_amax)).clamp(-448.0, 448.0).to(fp8_dtype).contiguous()

                    is_fp16 = inp.dtype == fp16_dtype and fp16_dtype != bf16_dtype
                    out_dtype = fp16_dtype if is_fp16 else bf16_dtype
                    bias = bias_fp16 if is_fp16 else bias_bf16
                    out_2d = torch_mod._scaled_mm(
                        x_q,
                        w_t,
                        scale_a=x_scale,
                        scale_b=w_scale,
                        out_dtype=out_dtype,
                        bias=bias,
                    )
                    out = out_2d.reshape(*inp.shape[:-1], lin.out_features)
                    return out if out.dtype == inp.dtype else out.to(inp.dtype)
                except Exception:  # noqa: BLE001
                    return base_fwd(inp)

            return _fp8_forward

        sub.forward = _make_fp8_fwd(sub, orig_fwd)
        sub._ptopt_fp8_orig_forward = orig_fwd
        sub._ptopt_fp8_patched = True
        patched += 1
    return patched


class InferenceOptRuntime:
    """Tracks inference_mode, CUDA-graph compile mode, enable_grad guards, and quant/FP8 state."""

    def __init__(self, cfg: OptimizeConfig, manager: "HookManager", torch_mod: Any) -> None:
        self.cfg = cfg
        self.manager = manager
        self.torch = torch_mod
        self.inference_mode_enabled = bool(cfg.inference.inference_mode)
        self.cuda_graphs_enabled = bool(cfg.inference.cuda_graphs)
        self.quant_mode = os.environ.get("PTOPT_INFERENCE_QUANT", "").strip().lower()
        self.fp8_requested = os.environ.get("PTOPT_INFERENCE_FP8") == "1"
        self.fp8_eligible, self.fp8_reason = check_fp8_eligibility(torch_mod)
        self.enable_grad_depth = 0
        self.inference_calls = 0
        self.grad_guard_skips = 0
        self.cuda_graph_compiles = 0
        self.cuda_graph_fallbacks = 0


def install_inference_opt_hook(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any,
) -> InferenceOptRuntime | None:
    """Install idempotent Inference / Running optimization hooks on `enable_grad` & `nn.Module.__call__`."""
    global _INFERENCE_OPT_INSTALLED  # noqa: PLW0603
    if _INFERENCE_OPT_INSTALLED or torch_mod is None or not hasattr(torch_mod, "nn"):
        return None
    if not (
        cfg.inference.inference_mode
        or cfg.inference.cuda_graphs
        or os.environ.get("PTOPT_INFERENCE_QUANT")
        or os.environ.get("PTOPT_INFERENCE_FP8") == "1"
    ):
        return None

    rt = InferenceOptRuntime(cfg, manager, torch_mod)

    if rt.fp8_requested:
        if rt.fp8_eligible:
            manager.record_event("inference.fp8", "APPLIED", f"FP8 inference enabled ({rt.fp8_reason})")
        else:
            manager.record_event("inference.fp8", "SKIPPED", f"FP8 skipped: {rt.fp8_reason}")

    # 1. Track explicit `torch.enable_grad()` regions so eval modules never lose intentional grads
    orig_enable_grad = getattr(torch_mod, "enable_grad", None)
    if orig_enable_grad is not None:
        class _TrackedEnableGrad:
            def __init__(self) -> None:
                self._inner = orig_enable_grad()

            def __enter__(self) -> Any:
                rt.enable_grad_depth += 1
                return self._inner.__enter__()

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
                rt.enable_grad_depth = max(0, rt.enable_grad_depth - 1)
                return self._inner.__exit__(exc_type, exc_val, exc_tb)

            def __call__(self, func: Any) -> Any:
                @functools.wraps(func)
                def _wrapped(*args: Any, **kwargs: Any) -> Any:
                    with self:
                        return func(*args, **kwargs)
                return _wrapped

        torch_mod._ptopt_orig_enable_grad = orig_enable_grad
        torch_mod.enable_grad = _TrackedEnableGrad  # type: ignore[assignment]

    # 2. Wrap top-level nn.Module.__call__ for safe eval optimization
    nn_mod = torch_mod.nn.Module
    call_attr = "__call__"
    orig_mod_call = getattr(nn_mod, call_attr)
    depth = [0]

    @functools.wraps(orig_mod_call)
    def _inference_mod_call(mod_self: Any, *args: Any, **kwargs: Any) -> Any:
        depth[0] += 1
        is_top = depth[0] == 1
        try:
            if not is_top or getattr(mod_self, "training", True):
                return orig_mod_call(mod_self, *args, **kwargs)

            # Safety guard: skip inference_mode if explicit enable_grad or requires_grad tensor exists
            needs_grad = (
                rt.enable_grad_depth > 0
                or _has_requires_grad_tensor(args, kwargs, torch_mod)
            )
            if needs_grad:
                rt.grad_guard_skips += 1
                if not getattr(mod_self, "_ptopt_grad_guard_logged", False):
                    mod_self._ptopt_grad_guard_logged = True
                    manager.record_event(
                        "inference.mode",
                        "SKIPPED",
                        "Requires-grad or enable_grad() detected during eval(); preserving autograd",
                    )
                return orig_mod_call(mod_self, *args, **kwargs)

            # Optional CPU dynamic int8 quantization on first pure-eval forward
            if (
                rt.quant_mode == "int8"
                and not getattr(mod_self, "_ptopt_quant_checked", False)
                and not _has_cuda_context(mod_self, args, kwargs, torch_mod)
            ):
                mod_self._ptopt_quant_checked = True
                ok_q, reason_q = check_dynamic_quant_eligibility(torch_mod)
                if ok_q:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", DeprecationWarning)
                            q_mod = torch_mod.ao.quantization.quantize_dynamic(
                                mod_self, {torch_mod.nn.Linear}, dtype=torch_mod.qint8
                            )
                        mod_self._ptopt_quantized_mod = q_mod
                        manager.record_event("inference.quant", "APPLIED", "CPU INT8 dynamic quantization applied")
                    except Exception as exc:  # noqa: BLE001
                        manager.record_event("inference.quant", "FALLBACK", "Dynamic INT8 quantization failed", str(exc))
                else:
                    manager.record_event("inference.quant", "SKIPPED", reason_q)

            # Optional FP8 Linear patching on sm >= 89 CUDA
            if rt.fp8_requested and rt.fp8_eligible and not getattr(mod_self, "_ptopt_fp8_checked", False):
                mod_self._ptopt_fp8_checked = True
                n_patched = apply_fp8_linear_eval_hooks(mod_self, torch_mod)
                if n_patched > 0:
                    manager.record_event(
                        "inference.fp8",
                        "APPLIED",
                        f"Patched {n_patched} Linear layers on {type(mod_self).__name__} with FP8 _scaled_mm",
                    )

            # Optional CUDA Graphs / reduce-overhead compile for static-shape CUDA eval
            on_cuda = _has_cuda_context(mod_self, args, kwargs, torch_mod)
            if (
                rt.cuda_graphs_enabled
                and on_cuda
                and callable(getattr(torch_mod, "compile", None))
                and not getattr(mod_self, "_orig_mod", None)
            ):
                if not getattr(mod_self, "_ptopt_infer_cg_checked", False):
                    mod_self._ptopt_infer_cg_checked = True
                    mod_self._ptopt_infer_cg_shapes = {_shape_signature(args, kwargs, torch_mod)}
                    if getattr(mod_self, "_compiled_call_impl", None) is None and callable(getattr(mod_self, "_call_impl", None)):
                        try:
                            mod_self._compiled_call_impl = torch_mod.compile(
                                mod_self._call_impl,
                                mode="reduce-overhead",
                            )
                            mod_self._ptopt_compile_checked = True
                            mod_self._ptopt_infer_cg_active = True
                            rt.cuda_graph_compiles += 1
                            manager.record_event(
                                "inference.cuda_graphs",
                                "APPLIED",
                                f"Compiled eval {type(mod_self).__name__} with mode=reduce-overhead (CUDA Graphs)",
                            )
                        except Exception as exc:  # noqa: BLE001
                            mod_self._compiled_call_impl = None
                            rt.cuda_graph_fallbacks += 1
                            manager.record_event(
                                "inference.cuda_graphs",
                                "FALLBACK",
                                f"reduce-overhead compile failed on {type(mod_self).__name__}; using standard eval",
                                str(exc),
                            )
                elif getattr(mod_self, "_ptopt_infer_cg_active", False):
                    seen = getattr(mod_self, "_ptopt_infer_cg_shapes", None)
                    sig = _shape_signature(args, kwargs, torch_mod)
                    if seen is not None and sig and sig not in seen:
                        mod_self._compiled_call_impl = None
                        mod_self._ptopt_infer_cg_active = False
                        mod_self._ptopt_compile_checked = False
                        rt.cuda_graph_fallbacks += 1
                        manager.record_event(
                            "inference.cuda_graphs",
                            "FALLBACK",
                            f"Dynamic eval input shape detected on {type(mod_self).__name__}; reverted reduce-overhead/CUDA Graphs",
                        )

            use_infer_ctx = (
                rt.inference_mode_enabled
                and hasattr(torch_mod, "inference_mode")
                and os.environ.get("PTOPT_INFERENCE_MODE") != "0"
            )
            ctx = torch_mod.inference_mode() if use_infer_ctx else nullcontext()
            with ctx:
                if use_infer_ctx:
                    rt.inference_calls += 1
                q_mod = getattr(mod_self, "_ptopt_quantized_mod", None)
                if q_mod is not None:
                    try:
                        return q_mod(*args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        mod_self._ptopt_quantized_mod = None
                        manager.record_event(
                            "inference.quant",
                            "FALLBACK",
                            f"Dynamic INT8 forward unsupported on {type(mod_self).__name__}; reverted to fp32 eval",
                            str(exc),
                        )
                        if manager.strict:
                            raise
                try:
                    return orig_mod_call(mod_self, *args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if getattr(mod_self, "_ptopt_infer_cg_active", False):
                        mod_self._compiled_call_impl = None
                        mod_self._ptopt_infer_cg_active = False
                        rt.cuda_graph_fallbacks += 1
                        manager.record_event(
                            "inference.cuda_graphs",
                            "FALLBACK",
                            f"reduce-overhead eval failed on {type(mod_self).__name__}; reverted to eager",
                            str(exc),
                        )
                        if manager.strict:
                            raise
                        return orig_mod_call(mod_self, *args, **kwargs)
                    raise
        finally:
            depth[0] -= 1

    nn_mod._ptopt_infer_call_attr = call_attr
    nn_mod._ptopt_infer_orig_call = orig_mod_call
    setattr(nn_mod, call_attr, _inference_mod_call)
    _INFERENCE_OPT_INSTALLED = True
    torch_mod._ptopt_inference_runtime = rt
    return rt


def uninstall_inference_opt_hook(torch_mod: Any) -> None:
    """Restore monkey-patched inference hooks (for unit tests)."""
    global _INFERENCE_OPT_INSTALLED  # noqa: PLW0603
    if not _INFERENCE_OPT_INSTALLED or torch_mod is None:
        return

    if hasattr(torch_mod, "_ptopt_orig_enable_grad"):
        torch_mod.enable_grad = torch_mod._ptopt_orig_enable_grad
        delattr(torch_mod, "_ptopt_orig_enable_grad")

    nn_mod = getattr(torch_mod, "nn", None) and torch_mod.nn.Module
    if nn_mod is not None and hasattr(nn_mod, "_ptopt_infer_orig_call"):
        setattr(nn_mod, getattr(nn_mod, "_ptopt_infer_call_attr"), getattr(nn_mod, "_ptopt_infer_orig_call"))
        delattr(nn_mod, "_ptopt_infer_call_attr")
        delattr(nn_mod, "_ptopt_infer_orig_call")

    if hasattr(torch_mod, "_ptopt_inference_runtime"):
        delattr(torch_mod, "_ptopt_inference_runtime")
    _INFERENCE_OPT_INSTALLED = False


def apply_inference_opt_pass(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any = None,
) -> str:
    """Enable safe inference/running optimization hooks."""
    tmod = torch_mod if torch_mod is not None else sys.modules.get("torch")
    if tmod is None:
        return "SKIPPED: torch not loaded"
    rt = install_inference_opt_hook(cfg, manager, tmod)
    if rt is None:
        return "SKIPPED: inference optimization disabled"
    return (
        f"Inference opt hook installed (inference_mode={rt.inference_mode_enabled}, "
        f"cuda_graphs={rt.cuda_graphs_enabled}, fp8_req={rt.fp8_requested})"
    )
