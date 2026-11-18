"""Pass 3.6: Safe Inference Optimization (`inference_mode()`, CUDA Graphs / reduce-overhead, INT8 & FP8)."""

from __future__ import annotations

import functools
import os
import sys
import warnings
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from src.config import OptimizeConfig
from src.hooks.module_dispatch import (
    build_shape_signature,
    has_cuda_context,
    has_requires_grad_tensor,
    iter_nested_tensors,
    register_module_call_handler,
    unregister_module_call_handler,
)

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
    return has_cuda_context(mod_self, args, kwargs, torch_mod)


def _has_requires_grad_tensor(args: tuple[Any, ...], kwargs: dict[str, Any], torch_mod: Any) -> bool:
    return has_requires_grad_tensor(args, kwargs, torch_mod)


def _shape_signature(args: tuple[Any, ...], kwargs: dict[str, Any], torch_mod: Any) -> tuple[Any, ...]:
    return build_shape_signature(args, kwargs, torch_mod, include_device=False)


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


def _is_cuda_tensor(t: Any) -> bool:
    """Return True when tensor resides on CUDA (helper kept clean for CPU unit mocking)."""
    return bool(getattr(t, "is_cuda", False))


def _is_dynamo_compiling(torch_mod: Any) -> bool:
    """Return True when executing inside a TorchDynamo/torch.compile trace."""
    compiler = getattr(torch_mod, "compiler", None)
    is_comp_fn = getattr(compiler, "is_compiling", None)
    if callable(is_comp_fn):
        try:
            return bool(is_comp_fn())
        except Exception:  # noqa: BLE001
            return False
    return False


def _quantize_fp8_traceable(flat: Any, fp8_dtype: Any, fp32_dtype: Any) -> tuple[Any, Any]:
    """Pure-ATen dynamic per-tensor FP8 quantizer traceable by TorchDynamo/Inductor (0 graph breaks)."""
    x_f32 = flat if flat.dtype == fp32_dtype else flat.to(fp32_dtype)
    x_amax = x_f32.abs().amax().clamp(min=1e-12)
    x_scale = x_amax / 448.0
    inv_scale = 448.0 / x_amax
    x_q = (x_f32 * inv_scale).clamp(-448.0, 448.0).to(fp8_dtype)
    if not x_q.is_contiguous():
        x_q = x_q.contiguous()
    return x_q, x_scale


def _quantize_fp8_static_traceable(
    flat: Any,
    x_inv_scale: Any,
    fp8_dtype: Any,
    fp32_dtype: Any,
) -> Any:
    """Single-pass static FP8 activation quantizer (no `amax()` HBM reduction pass)."""
    x_f32 = flat if flat.dtype == fp32_dtype else flat.to(fp32_dtype)
    x_q = (x_f32 * x_inv_scale).clamp(-448.0, 448.0).to(fp8_dtype)
    return x_q if x_q.is_contiguous() else x_q.contiguous()


def _update_fp8_static_scale(
    mod: Any,
    x_scale: Any,
    fp32_dtype: Any,
    *,
    headroom: float = 1.08,
    calib_steps: int = 2,
) -> None:
    """Update EMA FP8 activation scale + inv_scale on `mod` and flip `_ptopt_fp8_static_ready`."""
    cur_scale = (x_scale.detach().to(fp32_dtype).reshape(()) * headroom).clone()
    prev = getattr(mod, "_ptopt_fp8_x_scale", None)
    if prev is None:
        new_scale = cur_scale
    else:
        # Take conservative smoothed upper envelope to guard against outlier clipping
        ema = 0.5 * prev + 0.5 * cur_scale
        new_scale = ema if float(ema) >= float(cur_scale) else (0.25 * prev + 0.75 * cur_scale)
    mod._ptopt_fp8_x_scale = new_scale.clone()
    mod._ptopt_fp8_x_inv_scale = (1.0 / new_scale.clamp(min=1e-12)).reshape(()).clone()
    seen = int(getattr(mod, "_ptopt_fp8_calib_seen", 0)) + 1
    mod._ptopt_fp8_calib_seen = seen
    if seen >= calib_steps:
        mod._ptopt_fp8_static_ready = True


def _estimate_token_rows(tensor: Any) -> int:
    """Estimate effective GEMM `M` rows, handling 2D integer token IDs `(B, S)` properly."""
    if not hasattr(tensor, "dim") or tensor.dim() < 2:
        return 0
    is_fp = getattr(tensor, "is_floating_point", None)
    is_float = bool(is_fp()) if callable(is_fp) else True
    if tensor.dim() == 2 and not is_float:
        return int(tensor.numel())
    return int(tensor.numel()) // max(1, int(tensor.shape[-1]))


def _quantize_fp8_weight_traceable(
    lin: Any,
    fp8_dtype: Any,
    bf16_dtype: Any,
    fp16_dtype: Any,
    fp32_dtype: Any,
) -> tuple[Any, Any, Any, Any]:
    """Traceable pure-ATen weight FP8 quantization fallback (when pre-quant cache is absent)."""
    w_src = lin.weight
    w_f32 = w_src if w_src.dtype == fp32_dtype else w_src.to(fp32_dtype)
    w_amax = w_f32.abs().amax().clamp(min=1e-12)
    w_scale = w_amax / 448.0
    w_inv_scale = 448.0 / w_amax
    w_q = (w_f32 * w_inv_scale).clamp(-448.0, 448.0).to(fp8_dtype)
    if not w_q.is_contiguous():
        w_q = w_q.contiguous()
    w_t = w_q.t()
    bias_bf16 = lin.bias.to(bf16_dtype) if lin.bias is not None else None
    bias_fp16 = lin.bias.to(fp16_dtype) if (lin.bias is not None and fp16_dtype != bf16_dtype) else bias_bf16
    return w_t, w_scale, bias_bf16, bias_fp16


def _build_fp8_tensor_cache(
    weight: Any,
    bias: Any,
    fp8_dtype: Any,
    bf16_dtype: Any,
    fp16_dtype: Any,
    fp32_dtype: Any,
) -> tuple[Any, ...]:
    """Pre-quantize any `[out, in]` weight tensor into strict col-major `(1, K)` FP8 layout (`w_t`)."""
    w_src = weight.detach()
    w_f32 = w_src if w_src.dtype == fp32_dtype else w_src.to(fp32_dtype)
    w_amax = w_f32.abs().amax().clamp(min=1e-12)
    w_scale = (w_amax / 448.0).to(dtype=fp32_dtype, device=w_src.device)
    w_inv_scale = (448.0 / w_amax).to(dtype=fp32_dtype, device=w_src.device)
    w_q = (w_f32 * w_inv_scale).clamp(-448.0, 448.0).to(fp8_dtype)
    if not w_q.is_contiguous():
        w_q = w_q.contiguous()
    del w_f32
    w_t = w_q.t()
    bias_bf16 = bias.detach().to(bf16_dtype) if bias is not None else None
    bias_fp16 = bias.detach().to(fp16_dtype) if (bias is not None and fp16_dtype != bf16_dtype) else bias_bf16
    return (weight._version, w_t, w_scale, bias_bf16, bias_fp16)


def _build_fp8_weight_cache(
    lin: Any,
    torch_mod: Any,
    fp8_dtype: Any,
    bf16_dtype: Any,
    fp16_dtype: Any,
    fp32_dtype: Any,
    *,
    drop_source_weight: bool = False,
) -> tuple[Any, ...]:
    """Pre-quantize linear weight once into strict col-major `(1, K)` FP8 layout (`w_t`)."""
    ver, w_t, w_scale, bias_bf16, bias_fp16 = _build_fp8_tensor_cache(
        lin.weight, getattr(lin, "bias", None), fp8_dtype, bf16_dtype, fp16_dtype, fp32_dtype
    )
    # Opt-in only (PTOPT_FP8_DROP_BF16_EVAL=1): keep default state_dict() 100% untouched.
    if drop_source_weight and not getattr(lin, "training", True):
        w_src = lin.weight
        lin.weight.data = torch_mod.empty((0,), dtype=w_src.dtype, device=w_src.device)
        ver = lin.weight._version
    return (ver, w_t, w_scale, bias_bf16, bias_fp16)


def prewarm_fp8_weight_caches(module: Any, torch_mod: Any, *, min_work: int | None = None) -> int:
    """Pre-build FP8 weight caches on eligible patched CUDA Linear modules before `torch.compile`."""
    if module is None or torch_mod is None or not hasattr(module, "modules"):
        return 0
    fp8_dtype = getattr(torch_mod, "float8_e4m3fn", None)
    bf16_dtype = getattr(torch_mod, "bfloat16", None)
    fp16_dtype = getattr(torch_mod, "float16", bf16_dtype)
    fp32_dtype = getattr(torch_mod, "float32", None)
    if fp8_dtype is None or bf16_dtype is None or fp32_dtype is None:
        return 0
    work_gate = int(os.environ.get("PTOPT_FP8_MIN_WORK", str(512 * 512))) if min_work is None else int(min_work)
    drop_w = os.environ.get("PTOPT_FP8_DROP_BF16_EVAL", "0") == "1"
    built = 0
    for sub in module.modules():
        if getattr(sub, "training", True):
            continue
        if getattr(sub, "_ptopt_fp8_mha_patched", False) and _is_cuda_tensor(getattr(sub, "in_proj_weight", None)):
            if getattr(sub, "_ptopt_fp8_inproj_cache", None) is None:
                sub._ptopt_fp8_inproj_cache = _build_fp8_tensor_cache(
                    sub.in_proj_weight, getattr(sub, "in_proj_bias", None), fp8_dtype, bf16_dtype, fp16_dtype, fp32_dtype
                )
                built += 1
            out_p = getattr(sub, "out_proj", None)
            if out_p is not None and _is_cuda_tensor(getattr(out_p, "weight", None)):
                if getattr(out_p, "_ptopt_fp8_w_cache", None) is None:
                    out_p._ptopt_fp8_w_cache = _build_fp8_tensor_cache(
                        out_p.weight, getattr(out_p, "bias", None), fp8_dtype, bf16_dtype, fp16_dtype, fp32_dtype
                    )
                    built += 1
            continue
        if not getattr(sub, "_ptopt_fp8_patched", False):
            continue
        if not (hasattr(sub, "in_features") and hasattr(sub, "out_features")):
            continue
        if (int(sub.in_features) * int(sub.out_features)) < work_gate:
            continue
        w_cache = getattr(sub, "_ptopt_fp8_w_cache", None)
        if w_cache is None or (not drop_w and w_cache[0] != sub.weight._version):
            if getattr(sub.weight, "numel", lambda: 0)() > 0:
                sub._ptopt_fp8_w_cache = _build_fp8_weight_cache(
                    sub, torch_mod, fp8_dtype, bf16_dtype, fp16_dtype, fp32_dtype, drop_source_weight=drop_w
                )
                built += 1
    return built


def apply_fp8_linear_eval_hooks(module: Any, torch_mod: Any, *, min_work: int | None = None) -> int:
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

    work_gate = int(os.environ.get("PTOPT_FP8_MIN_WORK", str(512 * 512))) if min_work is None else int(min_work)
    drop_w = os.environ.get("PTOPT_FP8_DROP_BF16_EVAL", "0") == "1"
    static_scale_enabled = os.environ.get("PTOPT_FP8_STATIC_SCALE") == "1"
    try:
        calib_steps = max(1, int(os.environ.get("PTOPT_FP8_CALIB_STEPS", "2")))
    except ValueError:
        calib_steps = 2
    mha_opt = os.environ.get("PTOPT_FP8_MHA", "").strip()
    allow_mha_fp8 = mha_opt == "1" or (mha_opt != "0" and (min_work is not None))
    triton_quant_box: list[Any] = [_get_triton_fp8_quantizer()]

    # nn.MultiheadAttention passes out_proj.weight directly into F.multi_head_attention_forward
    # instead of invoking out_proj.forward(); never drop source weights on out_proj.
    mha_cls = getattr(nn_ns, "MultiheadAttention", None)
    skip_linear_ids: set[int] = set()
    for mod in module.modules():
        if mha_cls is not None and isinstance(mod, mha_cls) and hasattr(mod, "out_proj"):
            skip_linear_ids.add(id(mod.out_proj))

    def _run_fp8_eager_cached(
        flat: Any,
        cache_owner: Any,
        cache_attr: str,
        src_weight: Any,
        src_bias: Any,
        is_fp16: bool,
        out_dtype: Any,
        allow_drop: bool,
    ) -> Any:
        w_cache = getattr(cache_owner, cache_attr, None)
        if w_cache is None or (not allow_drop and w_cache[0] != src_weight._version):
            if getattr(src_weight, "numel", lambda: 0)() == 0:
                return None
            if allow_drop and hasattr(cache_owner, "weight") and cache_owner.weight is src_weight:
                w_cache = _build_fp8_weight_cache(
                    cache_owner, torch_mod, fp8_dtype, bf16_dtype, fp16_dtype, fp32_dtype, drop_source_weight=True
                )
            else:
                w_cache = _build_fp8_tensor_cache(
                    src_weight, src_bias, fp8_dtype, bf16_dtype, fp16_dtype, fp32_dtype
                )
            setattr(cache_owner, cache_attr, w_cache)
        _, w_t, w_scale, bias_bf16, bias_fp16 = w_cache

        if (
            static_scale_enabled
            and getattr(cache_owner, "_ptopt_fp8_static_ready", False)
            and getattr(cache_owner, "_ptopt_fp8_x_scale", None) is not None
        ):
            x_scale = cache_owner._ptopt_fp8_x_scale
            x_q = _quantize_fp8_static_traceable(flat, cache_owner._ptopt_fp8_x_inv_scale, fp8_dtype, fp32_dtype)
        else:
            tq = triton_quant_box[0]
            if tq is not None:
                try:
                    x_q, x_scale = tq(flat, fp8_dtype)
                except Exception:  # noqa: BLE001
                    triton_quant_box[0] = None
                    x_q, x_scale = _quantize_fp8_traceable(flat, fp8_dtype, fp32_dtype)
            else:
                x_q, x_scale = _quantize_fp8_traceable(flat, fp8_dtype, fp32_dtype)
            if static_scale_enabled:
                _update_fp8_static_scale(cache_owner, x_scale, fp32_dtype, calib_steps=calib_steps)

        bias = bias_fp16 if is_fp16 else bias_bf16
        return torch_mod._scaled_mm(
            x_q,
            w_t,
            scale_a=x_scale,
            scale_b=w_scale,
            out_dtype=out_dtype,
            bias=bias,
        )

    patched = 0
    for sub in module.modules():
        if (
            mha_cls is not None
            and isinstance(sub, mha_cls)
            and not getattr(sub, "_ptopt_fp8_mha_patched", False)
            and (allow_mha_fp8 or (mha_opt != "0" and int(getattr(sub, "embed_dim", 0)) >= 1024))
            and getattr(sub, "_qkv_same_embed_dim", False)
            and getattr(sub, "in_proj_weight", None) is not None
            and _is_cuda_tensor(sub.in_proj_weight)
            and int(getattr(sub, "embed_dim", 0)) >= 128
            and int(getattr(sub, "embed_dim", 0)) % 16 == 0
            and (int(sub.embed_dim) * 3 * int(sub.embed_dim)) >= work_gate
            and getattr(sub, "bias_k", None) is None
            and getattr(sub, "bias_v", None) is None
            and not getattr(sub, "add_zero_attn", False)
            and hasattr(sub, "out_proj")
            and callable(getattr(func_ns, "scaled_dot_product_attention", None))
        ):
            orig_mha_fwd = sub.forward

            def _make_mha_fp8_fwd(mha_mod: Any, base_fwd: Any) -> Any:
                embed_dim = int(mha_mod.embed_dim)
                num_heads = int(mha_mod.num_heads)
                head_dim = embed_dim // num_heads
                batch_first = bool(getattr(mha_mod, "batch_first", False))

                @functools.wraps(base_fwd)
                def _mha_fp8_forward(
                    query: Any,
                    key: Any,
                    value: Any,
                    key_padding_mask: Any = None,
                    need_weights: bool = True,
                    attn_mask: Any = None,
                    average_attn_weights: bool = True,
                    is_causal: bool = False,
                ) -> Any:
                    if (
                        getattr(mha_mod, "training", True)
                        or need_weights
                        or attn_mask is not None
                        or key_padding_mask is not None
                        or query is not key
                        or key is not value
                        or not _is_cuda_tensor(query)
                        or query.dim() != 3
                        or int(query.shape[-1]) != embed_dim
                    ):
                        return base_fwd(
                            query,
                            key,
                            value,
                            key_padding_mask=key_padding_mask,
                            need_weights=need_weights,
                            attn_mask=attn_mask,
                            average_attn_weights=average_attn_weights,
                            is_causal=is_causal,
                        )
                    q_3d = query if batch_first else query.transpose(0, 1)
                    bsz, seq_len, _ = q_3d.shape
                    m_rows = int(bsz) * int(seq_len)
                    if m_rows < 512 or m_rows % 16 != 0:
                        return base_fwd(
                            query,
                            key,
                            value,
                            key_padding_mask=key_padding_mask,
                            need_weights=need_weights,
                            attn_mask=attn_mask,
                            average_attn_weights=average_attn_weights,
                            is_causal=is_causal,
                        )
                    is_fp16 = query.dtype == fp16_dtype and fp16_dtype != bf16_dtype
                    out_dtype = fp16_dtype if is_fp16 else bf16_dtype
                    if _is_dynamo_compiling(torch_mod):
                        in_cache = getattr(mha_mod, "_ptopt_fp8_inproj_cache", None)
                        out_cache = getattr(mha_mod.out_proj, "_ptopt_fp8_w_cache", None)
                        if in_cache is None or out_cache is None:
                            return base_fwd(
                                query,
                                key,
                                value,
                                key_padding_mask=key_padding_mask,
                                need_weights=need_weights,
                                attn_mask=attn_mask,
                                average_attn_weights=average_attn_weights,
                                is_causal=is_causal,
                            )
                        _, in_wt, in_ws, in_b16, in_f16 = in_cache
                        flat_q = q_3d.reshape(m_rows, embed_dim)
                        if static_scale_enabled and getattr(mha_mod, "_ptopt_fp8_static_ready", False):
                            xq_scale = mha_mod._ptopt_fp8_x_scale
                            xq = _quantize_fp8_static_traceable(flat_q, mha_mod._ptopt_fp8_x_inv_scale, fp8_dtype, fp32_dtype)
                        else:
                            xq, xq_scale = _quantize_fp8_traceable(flat_q, fp8_dtype, fp32_dtype)
                        qkv_2d = torch_mod._scaled_mm(
                            xq, in_wt, scale_a=xq_scale, scale_b=in_ws, out_dtype=out_dtype, bias=(in_f16 if is_fp16 else in_b16)
                        )
                        q_h, k_h, v_h = qkv_2d.reshape(bsz, seq_len, 3, num_heads, head_dim).unbind(dim=2)
                        attn = func_ns.scaled_dot_product_attention(
                            q_h.transpose(1, 2), k_h.transpose(1, 2), v_h.transpose(1, 2), dropout_p=0.0, is_causal=bool(is_causal)
                        )
                        attn_2d = attn.transpose(1, 2).contiguous().reshape(m_rows, embed_dim)
                        _, o_wt, o_ws, o_b16, o_f16 = out_cache
                        if static_scale_enabled and getattr(mha_mod.out_proj, "_ptopt_fp8_static_ready", False):
                            ao_scale = mha_mod.out_proj._ptopt_fp8_x_scale
                            aq = _quantize_fp8_static_traceable(attn_2d, mha_mod.out_proj._ptopt_fp8_x_inv_scale, fp8_dtype, fp32_dtype)
                        else:
                            aq, ao_scale = _quantize_fp8_traceable(attn_2d, fp8_dtype, fp32_dtype)
                        out_2d = torch_mod._scaled_mm(
                            aq, o_wt, scale_a=ao_scale, scale_b=o_ws, out_dtype=out_dtype, bias=(o_f16 if is_fp16 else o_b16)
                        )
                        out_3d = out_2d.reshape(bsz, seq_len, embed_dim)
                        if not batch_first:
                            out_3d = out_3d.transpose(0, 1)
                        return (out_3d if out_3d.dtype == query.dtype else out_3d.to(query.dtype)), None

                    try:
                        qkv_2d = _run_fp8_eager_cached(
                            q_3d.reshape(m_rows, embed_dim),
                            mha_mod,
                            "_ptopt_fp8_inproj_cache",
                            mha_mod.in_proj_weight,
                            getattr(mha_mod, "in_proj_bias", None),
                            is_fp16,
                            out_dtype,
                            False,
                        )
                        if qkv_2d is None:
                            return base_fwd(
                                query,
                                key,
                                value,
                                key_padding_mask=key_padding_mask,
                                need_weights=need_weights,
                                attn_mask=attn_mask,
                                average_attn_weights=average_attn_weights,
                                is_causal=is_causal,
                            )
                        q_h, k_h, v_h = qkv_2d.reshape(bsz, seq_len, 3, num_heads, head_dim).unbind(dim=2)
                        attn = func_ns.scaled_dot_product_attention(
                            q_h.transpose(1, 2),
                            k_h.transpose(1, 2),
                            v_h.transpose(1, 2),
                            dropout_p=0.0,
                            is_causal=bool(is_causal),
                        )
                        attn_2d = attn.transpose(1, 2).contiguous().reshape(m_rows, embed_dim)
                        out_2d = _run_fp8_eager_cached(
                            attn_2d,
                            mha_mod.out_proj,
                            "_ptopt_fp8_w_cache",
                            mha_mod.out_proj.weight,
                            getattr(mha_mod.out_proj, "bias", None),
                            is_fp16,
                            out_dtype,
                            False,
                        )
                        if out_2d is None:
                            return base_fwd(
                                query,
                                key,
                                value,
                                key_padding_mask=key_padding_mask,
                                need_weights=need_weights,
                                attn_mask=attn_mask,
                                average_attn_weights=average_attn_weights,
                                is_causal=is_causal,
                            )
                        out_3d = out_2d.reshape(bsz, seq_len, embed_dim)
                        if not batch_first:
                            out_3d = out_3d.transpose(0, 1)
                        return (out_3d if out_3d.dtype == query.dtype else out_3d.to(query.dtype)), None
                    except Exception:  # noqa: BLE001
                        return base_fwd(
                            query,
                            key,
                            value,
                            key_padding_mask=key_padding_mask,
                            need_weights=need_weights,
                            attn_mask=attn_mask,
                            average_attn_weights=average_attn_weights,
                            is_causal=is_causal,
                        )

                return _mha_fp8_forward

            sub.forward = _make_mha_fp8_fwd(sub, orig_mha_fwd)
            sub._ptopt_fp8_mha_orig_forward = orig_mha_fwd
            sub._ptopt_fp8_mha_patched = True
            patched += 1
            continue

        if not isinstance(sub, linear_cls) or getattr(sub, "_ptopt_fp8_patched", False):
            continue
        if id(sub) in skip_linear_ids or type(sub).__name__ == "NonDynamicallyQuantizableLinear":
            continue
        if sub.in_features < 128 or sub.in_features % 16 != 0 or sub.out_features % 16 != 0:
            continue
        if not _is_cuda_tensor(sub.weight):
            continue

        orig_fwd = sub.forward

        def _make_fp8_fwd(lin: Any, base_fwd: Any) -> Any:
            lin_work = int(lin.in_features) * int(lin.out_features)

            @functools.wraps(base_fwd)
            def _fp8_forward(inp: Any) -> Any:
                if getattr(lin, "training", True) or not _is_cuda_tensor(inp):
                    return base_fwd(inp)
                if inp.dim() < 2 or inp.shape[-1] != lin.in_features:
                    return base_fwd(inp)
                if lin_work < work_gate:
                    return base_fwd(inp)

                flat = inp.reshape(-1, lin.in_features)
                m_rows = int(flat.shape[0])
                # Dynamic per-tensor FP8 quantize overhead hurts small M (<512) and loses precision (E4M3)
                if m_rows < 512 or m_rows % 16 != 0:
                    return base_fwd(inp)

                is_fp16 = inp.dtype == fp16_dtype and fp16_dtype != bf16_dtype
                out_dtype = fp16_dtype if is_fp16 else bf16_dtype

                # Path 1: TorchDynamo / TorchInductor compile path (0 graph breaks; fuses quantize with neighbors)
                if _is_dynamo_compiling(torch_mod):
                    w_cache = getattr(lin, "_ptopt_fp8_w_cache", None)
                    if w_cache is not None:
                        _, w_t, w_scale, bias_bf16, bias_fp16 = w_cache
                    else:
                        w_t, w_scale, bias_bf16, bias_fp16 = _quantize_fp8_weight_traceable(
                            lin, fp8_dtype, bf16_dtype, fp16_dtype, fp32_dtype
                        )
                    if static_scale_enabled and getattr(lin, "_ptopt_fp8_static_ready", False):
                        x_scale = lin._ptopt_fp8_x_scale
                        x_q = _quantize_fp8_static_traceable(flat, lin._ptopt_fp8_x_inv_scale, fp8_dtype, fp32_dtype)
                    else:
                        x_q, x_scale = _quantize_fp8_traceable(flat, fp8_dtype, fp32_dtype)
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

                # Path 2: Eager mode (2-pass fused Triton activation quantizer or calibrated 1-pass static scale)
                try:
                    out_2d = _run_fp8_eager_cached(
                        flat,
                        lin,
                        "_ptopt_fp8_w_cache",
                        lin.weight,
                        getattr(lin, "bias", None),
                        is_fp16,
                        out_dtype,
                        drop_w,
                    )
                    if out_2d is None:
                        return base_fwd(inp)
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

    # 2. Register top-level nn.Module.__call__ handler on unified dispatcher
    def _inference_mod_call(mod_self: Any, args: tuple[Any, ...], kwargs: dict[str, Any], next_fn: Any) -> Any:
        if getattr(mod_self, "training", True):
            return next_fn(mod_self, *args, **kwargs)

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
            return next_fn(mod_self, *args, **kwargs)

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

        # When compiling eval module with FP8 enabled and large-M inputs (M >= 512), pre-build weight
        # caches once before torch.compile traces so Dynamo lifts pre-quantized FP8 weights statically.
        if (
            rt.fp8_requested
            and rt.fp8_eligible
            and (rt.cfg.compile.enabled or rt.cuda_graphs_enabled)
            and not getattr(mod_self, "_ptopt_fp8_prewarmed", False)
        ):
            tensor_cls = getattr(torch_mod, "Tensor", None)
            has_large_m = any(
                _estimate_token_rows(tensor) >= 512
                for item in (*args, *kwargs.values())
                for tensor in iter_nested_tensors(item, tensor_cls)
            )
            if has_large_m:
                mod_self._ptopt_fp8_prewarmed = True
                prewarm_fp8_weight_caches(mod_self, torch_mod)
                if os.environ.get("PTOPT_FP8_STATIC_SCALE") == "1" and not getattr(
                    mod_self, "_ptopt_fp8_static_calibrated", False
                ):
                    mod_self._ptopt_fp8_static_calibrated = True
                    no_grad_fn = getattr(torch_mod, "no_grad", None)
                    amp_ns = getattr(torch_mod, "amp", None)
                    autocast_cls = getattr(amp_ns, "autocast", None) if amp_ns is not None else None
                    amp_dtype = getattr(torch_mod, "bfloat16", None) if str(getattr(rt.cfg.amp, "dtype", "")) == "bf16" else getattr(torch_mod, "float16", None)
                    call_impl = getattr(mod_self, "_call_impl", None)
                    if callable(call_impl):
                        try:
                            from contextlib import nullcontext

                            g_ctx = no_grad_fn() if callable(no_grad_fn) else nullcontext()
                            a_ctx = (
                                autocast_cls("cuda", dtype=amp_dtype)
                                if (callable(autocast_cls) and getattr(rt.cfg.amp, "enabled", False) and amp_dtype is not None)
                                else nullcontext()
                            )
                            with g_ctx, a_ctx:
                                call_impl(*args, **kwargs)
                            for sub_m in mod_self.modules():
                                if getattr(sub_m, "_ptopt_fp8_x_scale", None) is not None:
                                    sub_m._ptopt_fp8_static_ready = True
                            manager.record_event(
                                "inference.fp8_static",
                                "APPLIED",
                                f"Calibrated static FP8 activation scales on {type(mod_self).__name__}",
                            )
                        except Exception:  # noqa: BLE001
                            pass

        # Optional CUDA Graphs / reduce-overhead compile for static-shape CUDA eval (isolated from train handle)
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
                existing_eval = getattr(mod_self, "_ptopt_eval_compiled_impl", None)
                existing_cur = getattr(mod_self, "_compiled_call_impl", None)
                # Compile dedicated eval CUDA Graph handle if not already user-precompiled
                is_user_precompiled = existing_cur is not None and existing_cur is not getattr(
                    mod_self, "_ptopt_train_compiled_impl", None
                )
                if existing_eval is None and not is_user_precompiled and callable(getattr(mod_self, "_call_impl", None)):
                    try:
                        eval_impl = torch_mod.compile(
                            mod_self._call_impl,
                            mode="reduce-overhead",
                        )
                        mod_self._ptopt_eval_compiled_impl = eval_impl
                        mod_self._compiled_call_impl = eval_impl
                        mod_self._ptopt_compile_checked = True
                        mod_self._ptopt_infer_cg_active = True
                        rt.cuda_graph_compiles += 1
                        manager.record_event(
                            "inference.cuda_graphs",
                            "APPLIED",
                            f"Compiled eval {type(mod_self).__name__} with mode=reduce-overhead (CUDA Graphs)",
                        )
                    except Exception as exc:  # noqa: BLE001
                        mod_self._ptopt_eval_compiled_impl = None
                        mod_self._compiled_call_impl = None
                        rt.cuda_graph_fallbacks += 1
                        manager.record_event(
                            "inference.cuda_graphs",
                            "FALLBACK",
                            f"reduce-overhead compile failed on {type(mod_self).__name__}; using standard eval",
                            str(exc),
                        )
            elif getattr(mod_self, "_ptopt_infer_cg_active", False):
                eval_impl = getattr(mod_self, "_ptopt_eval_compiled_impl", None)
                if eval_impl is not None:
                    mod_self._compiled_call_impl = eval_impl
                seen = getattr(mod_self, "_ptopt_infer_cg_shapes", None)
                sig = _shape_signature(args, kwargs, torch_mod)
                if seen is not None and sig and sig not in seen:
                    mod_self._ptopt_eval_compiled_impl = None
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
                return next_fn(mod_self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if getattr(mod_self, "_ptopt_infer_cg_active", False):
                    mod_self._ptopt_eval_compiled_impl = None
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
                    return next_fn(mod_self, *args, **kwargs)
                raise

    register_module_call_handler(torch_mod, "inference.opt", 30, _inference_mod_call)
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

    unregister_module_call_handler(torch_mod, "inference.opt")

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
