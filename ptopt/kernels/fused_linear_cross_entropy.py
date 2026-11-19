"""Fused Chunked Linear + CrossEntropy kernel avoiding full `[N, V]` logits allocation in VRAM."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

import ptopt.kernels.cross_entropy as _ce_mod
from ptopt.kernels.cross_entropy import HAS_TRITON, triton

_cross_entropy_fwd_kernel = getattr(_ce_mod, "_cross_entropy_fwd_kernel", None)
_cross_entropy_bwd_kernel = getattr(_ce_mod, "_cross_entropy_bwd_kernel", None)

_TARGET_CHUNK_BYTES = 64 * 1024 * 1024


def compute_optimal_chunk_size(num_rows: int, vocab_size: int, dtype: torch.dtype, explicit_chunk: int | None = None) -> int:
    """Choose row chunk size keeping temporary `[chunk, vocab_size]` logits within ~64 MB."""
    if num_rows <= 0:
        return 1
    if explicit_chunk is not None and explicit_chunk > 0:
        return min(num_rows, int(explicit_chunk))
    elem_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    max_elems = _TARGET_CHUNK_BYTES // elem_bytes
    raw = max(128, min(2048, max_elems // max(1, vocab_size)))
    # Round down to multiple of 64 (or keep >=128) for clean GEMM tile alignment
    aligned = max(128, (raw // 64) * 64)
    return min(num_rows, aligned)


def ref_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Reference PyTorch `F.cross_entropy(F.linear(hidden, weight, bias).float(), targets)`."""
    logits = F.linear(hidden, weight, bias)
    return F.cross_entropy(logits.reshape(-1, weight.shape[0]).float(), targets.reshape(-1), ignore_index=ignore_index, reduction=reduction)


def _active_linear_dtype(hidden: torch.Tensor, weight: torch.Tensor) -> torch.dtype:
    if hidden.is_cuda and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    if weight.dtype in (torch.float16, torch.bfloat16) and hidden.dtype == weight.dtype:
        return weight.dtype
    return hidden.dtype if hidden.dtype in (torch.float16, torch.bfloat16) and weight.dtype in (torch.float16, torch.bfloat16) else torch.promote_types(hidden.dtype, weight.dtype)


def _chunk_ce_fwd(
    logits_c: torch.Tensor,
    targets_c: torch.Tensor,
    losses_out: torch.Tensor,
    lse_out: torch.Tensor,
    *,
    ignore_index: int,
    block_v: int,
    warps: int,
) -> None:
    m_c, v_dim = logits_c.shape
    if HAS_TRITON and _cross_entropy_fwd_kernel is not None and logits_c.is_cuda:
        dummy_valid_count = torch.empty((), dtype=torch.int32, device=logits_c.device)
        _cross_entropy_fwd_kernel[(m_c,)](
            logits_c,
            targets_c,
            losses_out,
            lse_out,
            dummy_valid_count,
            logits_c.stride(0),
            targets_c.stride(0),
            v_dim,
            ignore_index,
            COUNT_VALID=False,
            BLOCK_V=block_v,
            num_warps=warps,
        )
        return

    logits_f = logits_c.float()
    lse_c = torch.logsumexp(logits_f, dim=-1)
    valid = (targets_c != ignore_index) & (targets_c >= 0) & (targets_c < v_dim)
    safe_tgt = torch.where(valid, targets_c, 0)
    tgt_vals = logits_f.gather(1, safe_tgt.unsqueeze(1)).squeeze(1)
    loss_c = torch.where(valid, lse_c - tgt_vals, 0.0)
    losses_out.copy_(loss_c)
    lse_out.copy_(torch.where(valid, lse_c, 0.0))


def _chunk_ce_bwd(
    logits_c: torch.Tensor,
    targets_c: torch.Tensor,
    lse_c: torch.Tensor,
    grad_in_c: torch.Tensor,
    *,
    is_scalar_grad: bool,
    ignore_index: int,
    block_v: int,
    warps: int,
) -> torch.Tensor:
    m_c, v_dim = logits_c.shape
    if HAS_TRITON and _cross_entropy_bwd_kernel is not None and logits_c.is_cuda:
        # In-place update into `logits_c` avoids allocating a second [chunk, V] dx buffer.
        _cross_entropy_bwd_kernel[(m_c,)](
            logits_c,
            logits_c,
            targets_c,
            lse_c,
            grad_in_c,
            logits_c.stride(0),
            logits_c.stride(0),
            targets_c.stride(0),
            v_dim,
            ignore_index,
            IS_SCALAR_GRAD=is_scalar_grad,
            BLOCK_V=block_v,
            num_warps=warps,
        )
        return logits_c

    probs = torch.exp(logits_c.float() - lse_c.unsqueeze(1))
    valid = (targets_c != ignore_index) & (targets_c >= 0) & (targets_c < v_dim)
    safe_tgt = torch.where(valid, targets_c, 0)
    probs.scatter_add_(1, safe_tgt.unsqueeze(1), torch.full((m_c, 1), -1.0, dtype=torch.float32, device=logits_c.device))
    g = grad_in_c.reshape(()) if is_scalar_grad else grad_in_c.unsqueeze(1)
    dx = torch.where(valid.unsqueeze(1), probs * g, 0.0)
    return dx.to(logits_c.dtype)


class FusedLinearCrossEntropyFn(torch.autograd.Function):
    """Exact chunked `Linear + CrossEntropy` without full `[N, V]` logits in VRAM."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden_2d: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        targets_1d: torch.Tensor,
        ignore_index: int,
        reduction: str,
        chunk_size: int | None,
        matmul_dtype_code: int,
    ) -> torch.Tensor:
        dtype_map = {0: hidden_2d.dtype, 1: torch.float32, 2: torch.float16, 3: torch.bfloat16}
        mm_dtype = dtype_map.get(matmul_dtype_code, _active_linear_dtype(hidden_2d, weight))

        h_2d = hidden_2d.contiguous()
        h_in_dtype = h_2d.dtype
        h_mm = h_2d if h_in_dtype == mm_dtype else h_2d.to(mm_dtype)
        tgt_1d = targets_1d.contiguous()
        n_rows, _h_dim = h_mm.shape
        vocab = int(weight.shape[0])

        chunk = compute_optimal_chunk_size(n_rows, vocab, mm_dtype, chunk_size)
        losses = torch.empty((n_rows,), dtype=torch.float32, device=h_mm.device)
        lse = torch.empty((n_rows,), dtype=torch.float32, device=h_mm.device)

        if HAS_TRITON and triton is not None and h_mm.is_cuda:
            block_v = min(4096, max(128, triton.next_power_of_2(vocab)))
            warps = 8 if block_v >= 2048 else 4
        else:
            block_v, warps = 1024, 4

        w_mm = weight.to(mm_dtype) if weight.dtype != mm_dtype else weight
        w_t = w_mm.t()
        b_mm = (bias.to(mm_dtype) if bias.dtype != mm_dtype else bias) if bias is not None else None

        for start in range(0, n_rows, chunk):
            end = min(start + chunk, n_rows)
            logits_c = torch.mm(h_mm[start:end], w_t)
            if b_mm is not None:
                logits_c.add_(b_mm)
            _chunk_ce_fwd(
                logits_c,
                tgt_1d[start:end],
                losses[start:end],
                lse[start:end],
                ignore_index=ignore_index,
                block_v=block_v,
                warps=warps,
            )
            del logits_c

        valid_count_t: torch.Tensor | None = None
        if reduction == "mean":
            valid_mask = (tgt_1d != ignore_index) & (tgt_1d >= 0) & (tgt_1d < vocab)
            valid_count_t = valid_mask.sum().to(torch.float32).clamp(min=1.0)

        ctx.save_for_backward(h_mm, weight, bias, tgt_1d, lse, valid_count_t)
        ctx.h_in_dtype = h_in_dtype
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.chunk = chunk
        ctx.mm_dtype = mm_dtype
        ctx.block_v = block_v
        ctx.warps = warps

        if reduction == "none":
            return losses
        if reduction == "sum":
            return losses.sum()
        assert valid_count_t is not None
        return losses.sum() / valid_count_t

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:
        h_mm, weight, bias, tgt_1d, lse, valid_count_t = ctx.saved_tensors
        n_rows, h_dim = h_mm.shape
        mm_dtype = ctx.mm_dtype
        h_in_dtype = ctx.h_in_dtype

        need_h = bool(ctx.needs_input_grad[0])
        need_w = bool(ctx.needs_input_grad[1])
        need_b = bias is not None and bool(ctx.needs_input_grad[2])

        n_chunks = (n_rows + ctx.chunk - 1) // ctx.chunk
        # Use direct cuBLAS addmm_ accumulation in mm_dtype when safe (fp32 always, or bf16 <= 16 chunks)
        # to eliminate per-chunk [V, H] temp allocations, fp32 casts, and separate add_ kernels.
        use_addmm_w = mm_dtype == torch.float32 or (mm_dtype == torch.bfloat16 and n_chunks <= 16)
        d_weight_dtype = mm_dtype if use_addmm_w else torch.float32

        d_hidden = torch.empty((n_rows, h_dim), dtype=h_in_dtype, device=h_mm.device) if need_h else None
        d_weight = torch.empty(weight.shape, dtype=d_weight_dtype, device=weight.device) if need_w else None
        dw_buf = torch.empty(weight.shape, dtype=mm_dtype, device=weight.device) if (need_w and not use_addmm_w) else None
        d_bias = torch.zeros(bias.shape, dtype=torch.float32, device=bias.device) if need_b else None

        is_scalar = ctx.reduction in ("mean", "sum")
        if ctx.reduction == "mean":
            assert valid_count_t is not None
            grad_in_all = grad_out.float().reshape(()) / valid_count_t
        elif ctx.reduction == "sum":
            grad_in_all = grad_out.float().reshape(())
        else:
            grad_in_all = grad_out.float().reshape(-1).contiguous()

        w_mm = weight.to(mm_dtype) if weight.dtype != mm_dtype else weight
        w_t = w_mm.t()
        b_mm = (bias.to(mm_dtype) if bias.dtype != mm_dtype else bias) if bias is not None else None

        for start in range(0, n_rows, ctx.chunk):
            end = min(start + ctx.chunk, n_rows)
            h_c = h_mm[start:end]

            logits_c = torch.mm(h_c, w_t)
            if b_mm is not None:
                logits_c.add_(b_mm)

            g_c = grad_in_all if is_scalar else grad_in_all[start:end]
            dlogits_c = _chunk_ce_bwd(
                logits_c,
                tgt_1d[start:end],
                lse[start:end],
                g_c,
                is_scalar_grad=is_scalar,
                ignore_index=ctx.ignore_index,
                block_v=ctx.block_v,
                warps=ctx.warps,
            )
            dl_mm = dlogits_c.to(mm_dtype) if dlogits_c.dtype != mm_dtype else dlogits_c

            if d_hidden is not None:
                if h_in_dtype == mm_dtype:
                    torch.mm(dl_mm, w_mm, out=d_hidden[start:end])
                else:
                    d_hidden[start:end].copy_(torch.mm(dl_mm, w_mm))
            if d_weight is not None:
                if use_addmm_w:
                    if start == 0:
                        torch.mm(dl_mm.t(), h_c, out=d_weight)
                    else:
                        d_weight.addmm_(dl_mm.t(), h_c)
                else:
                    assert dw_buf is not None
                    torch.mm(dl_mm.t(), h_c, out=dw_buf)
                    if start == 0:
                        d_weight.copy_(dw_buf)
                    else:
                        d_weight.add_(dw_buf)
            if d_bias is not None:
                d_bias.add_(dl_mm.sum(dim=0, dtype=torch.float32))
            del logits_c, dlogits_c

        dw_out = (d_weight if d_weight.dtype == weight.dtype else d_weight.to(weight.dtype)) if d_weight is not None else None
        db_out = (d_bias if d_bias.dtype == bias.dtype else d_bias.to(bias.dtype)) if (d_bias is not None and bias is not None) else None
        return d_hidden, dw_out, db_out, None, None, None, None, None


def _dtype_to_code(dt: torch.dtype) -> int:
    if dt == torch.float32:
        return 1
    if dt == torch.float16:
        return 2
    if dt == torch.bfloat16:
        return 3
    return 0


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    chunk_size: int | None = None,
    matmul_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute exact `F.cross_entropy(F.linear(hidden, weight, bias), targets)` in chunks saving logits VRAM."""
    if reduction not in ("mean", "sum", "none") or weight.dim() != 2 or hidden.numel() == 0:
        return ref_linear_cross_entropy(
            hidden,
            weight,
            bias,
            targets,
            ignore_index=ignore_index,
            reduction=reduction,
        )
    hidden_shape = hidden.shape
    h_2d = hidden.reshape(-1, int(hidden_shape[-1]))
    tgt_1d = targets.reshape(-1)
    if h_2d.shape[0] != tgt_1d.shape[0]:
        return ref_linear_cross_entropy(
            hidden,
            weight,
            bias,
            targets,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    mm_dt = matmul_dtype if matmul_dtype is not None else _active_linear_dtype(hidden, weight)
    out = FusedLinearCrossEntropyFn.apply(
        h_2d,
        weight,
        bias,
        tgt_1d,
        int(ignore_index),
        reduction,
        chunk_size,
        _dtype_to_code(mm_dt),
    )
    if reduction == "none":
        return out.reshape(hidden_shape[:-1])
    return out


class LazyLinearLogits(torch.Tensor):
    """Zero-VRAM `torch.Tensor` subclass deferring `F.linear(hidden, weight, bias)` until needed or fused."""

    _ptopt_lazy_linear: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, ...], torch.dtype]
    _ptopt_lazy_out_dtype: torch.dtype

    @staticmethod
    def __new__(
        cls,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        out_shape: tuple[int, ...],
        matmul_dtype: torch.dtype,
        out_dtype: torch.dtype | None = None,
    ) -> "LazyLinearLogits":
        eff_out_dtype = out_dtype if out_dtype is not None else matmul_dtype
        req = bool(hidden.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad))
        with torch.no_grad():
            if hidden.device.type == "meta":
                base = torch.empty(out_shape, dtype=eff_out_dtype, device="meta")
            else:
                try:
                    base = torch.empty_strided(
                        out_shape,
                        (0,) * len(out_shape),
                        dtype=eff_out_dtype,
                        device=hidden.device,
                    )
                except Exception:  # noqa: BLE001
                    base = torch.empty(out_shape, dtype=eff_out_dtype, device="meta")
            obj = torch.Tensor._make_subclass(cls, base, require_grad=False)
        obj._ptopt_lazy_linear = (hidden, weight, bias, tuple(out_shape), matmul_dtype)
        obj._ptopt_lazy_out_dtype = eff_out_dtype
        if req:
            with torch._C.DisableTorchFunctionSubclass():
                obj.requires_grad_(True)
        return obj

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        return self._ptopt_lazy_linear[0].device

    @property
    def is_cuda(self) -> bool:  # type: ignore[override]
        return self._ptopt_lazy_linear[0].is_cuda

    def materialize(self) -> torch.Tensor:
        """Compute full `F.linear(hidden, weight, bias)` if a non-fused consumer touches logits."""
        hidden, weight, bias, out_shape, mm_dtype = self._ptopt_lazy_linear
        out_dt = getattr(self, "_ptopt_lazy_out_dtype", mm_dtype)
        h_m = hidden.to(mm_dtype) if hidden.dtype != mm_dtype else hidden
        w_m = weight.to(mm_dtype) if weight.dtype != mm_dtype else weight
        b_m = (bias.to(mm_dtype) if bias.dtype != mm_dtype else bias) if bias is not None else None
        out = F.linear(h_m, w_m, b_m).reshape(out_shape)
        if out_dt != out.dtype:
            out = out.to(out_dt)
        return out

    @classmethod
    def __torch_function__(cls, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> Any:
        kw = kwargs or {}
        fname = getattr(func, "__name__", "")
        if (
            func
            in (
                torch.Tensor.shape.__get__,
                torch.Tensor.ndim.__get__,
                torch.Tensor.dtype.__get__,
                torch.Tensor.layout.__get__,
                torch.Tensor.requires_grad.__get__,
                torch.Tensor.size,
                torch.Tensor.dim,
                torch.Tensor.numel,
                torch.Tensor.stride,
                torch.Tensor.storage_offset,
                torch.Tensor.untyped_storage,
                torch.Tensor.is_conj,
                torch.Tensor.is_neg,
                torch.Tensor.is_pinned,
                torch.Tensor.is_contiguous,
            )
            or fname
            in (
                "_is_view",
                "is_floating_point",
                "is_complex",
                "data_ptr",
                "get_device",
                "element_size",
                "nbytes",
            )
        ):
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kw)

        first = args[0] if args else None
        if isinstance(first, LazyLinearLogits):
            hidden, weight, bias, out_shape, mm_dtype = first._ptopt_lazy_linear
            cur_out_dt = getattr(first, "_ptopt_lazy_out_dtype", mm_dtype)
            vocab = int(weight.shape[0])

            if func in (torch.Tensor.float, torch.Tensor.half, torch.Tensor.bfloat16) and len(args) == 1 and not kw:
                dt_map = {
                    torch.Tensor.float: torch.float32,
                    torch.Tensor.half: torch.float16,
                    torch.Tensor.bfloat16: torch.bfloat16,
                }
                return LazyLinearLogits(hidden, weight, bias, out_shape, mm_dtype, out_dtype=dt_map[func])

            if func is torch.Tensor.to and not kw.get("copy", False) and kw.get("memory_format") is None:
                target_dt: torch.dtype | None = kw.get("dtype")
                target_dev: Any = kw.get("device")
                other_args = args[1:]
                if len(other_args) == 1:
                    if isinstance(other_args[0], torch.dtype):
                        target_dt = other_args[0]
                    elif isinstance(other_args[0], (torch.device, str)):
                        target_dev = other_args[0]
                    elif isinstance(other_args[0], torch.Tensor):
                        target_dt = other_args[0].dtype
                        target_dev = other_args[0].device
                elif len(other_args) == 2 and isinstance(other_args[1], torch.dtype):
                    target_dev, target_dt = other_args[0], other_args[1]
                elif other_args:
                    target_dt = None
                    target_dev = "unsupported"

                same_dev = target_dev is None or (
                    target_dev != "unsupported" and torch.device(target_dev) == hidden.device
                )
                if same_dev and target_dt in (torch.float32, torch.float16, torch.bfloat16):
                    return LazyLinearLogits(hidden, weight, bias, out_shape, mm_dtype, out_dtype=target_dt)

            if func is torch.Tensor.contiguous:
                return LazyLinearLogits(hidden.contiguous(), weight, bias, out_shape, mm_dtype, out_dtype=cur_out_dt)

            if func in (torch.Tensor.view, torch.Tensor.reshape, torch.reshape):
                new_shape = args[1] if (len(args) == 2 and isinstance(args[1], (tuple, list))) else args[1:]
                with torch._C.DisableTorchFunctionSubclass():
                    resolved = tuple(torch.empty(out_shape, device="meta").reshape(*new_shape).shape)
                if len(resolved) >= 1 and resolved[-1] == vocab:
                    h_reshaped = hidden.reshape(*resolved[:-1], int(hidden.shape[-1]))
                    return LazyLinearLogits(h_reshaped, weight, bias, resolved, mm_dtype, out_dtype=cur_out_dt)

            if func is torch.Tensor.__getitem__ and len(args) == 2:
                idx = args[1]
                # Preserve lazy projection when slicing only batch/sequence dimensions (not vocab dim).
                leading_ndim = len(out_shape) - 1
                idx_tuple = idx if isinstance(idx, tuple) else (idx,)
                non_vocab_only = (
                    Ellipsis not in idx_tuple
                    and (
                        len(idx_tuple) <= leading_ndim
                        or (len(idx_tuple) == len(out_shape) and idx_tuple[-1] == slice(None))
                    )
                )
                if non_vocab_only:
                    h_idx = idx_tuple[:-1] if len(idx_tuple) == len(out_shape) else idx
                    sliced_h = hidden[h_idx]
                    new_out_shape = tuple(sliced_h.shape[:-1]) + (vocab,)
                    return LazyLinearLogits(sliced_h, weight, bias, new_out_shape, mm_dtype, out_dtype=cur_out_dt)

        new_args = tuple(x.materialize() if isinstance(x, LazyLinearLogits) else x for x in args)
        new_kw = {k: (v.materialize() if isinstance(v, LazyLinearLogits) else v) for k, v in kw.items()}
        with torch._C.DisableTorchFunctionSubclass():
            return func(*new_args, **new_kw)


def create_lazy_linear_logits(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> LazyLinearLogits:
    """Construct a `LazyLinearLogits` capturing the active autocast dtype and exact output shape."""
    mm_dtype = _active_linear_dtype(hidden, weight)
    out_shape = tuple(hidden.shape[:-1]) + (int(weight.shape[0]),)
    return LazyLinearLogits(hidden, weight, bias, out_shape, mm_dtype)


triton_fused_linear_cross_entropy = fused_linear_cross_entropy
