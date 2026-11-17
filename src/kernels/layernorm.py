"""Custom Triton Fused LayerNorm (Forward + Backward) with PyTorch reference fallback."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]
    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU dev environment
    triton = None
    tl = None
    HAS_TRITON = False

_MAX_FUSED_COLS = 16384


def ref_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """PyTorch reference LayerNorm."""
    return F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps)


if HAS_TRITON:

    @triton.jit
    def _layernorm_fwd_kernel(
        X_ptr,
        W_ptr,
        B_ptr,
        Y_ptr,
        Mean_ptr,
        Rstd_ptr,
        stride_x,
        stride_y,
        N,
        eps,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        diff = tl.where(mask, x - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / N
        rstd = tl.math.rsqrt(var + eps)

        tl.store(Mean_ptr + row, mean)
        tl.store(Rstd_ptr + row, rstd)

        x_hat = (x - mean) * rstd
        if HAS_WEIGHT:
            w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            x_hat = x_hat * w
        if HAS_BIAS:
            b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            x_hat = x_hat + b
        tl.store(Y_ptr + row * stride_y + cols, x_hat, mask=mask)

    @triton.jit
    def _layernorm_bwd_kernel(
        DX_ptr,
        PartialDW_ptr,
        PartialDB_ptr,
        DY_ptr,
        X_ptr,
        W_ptr,
        Mean_ptr,
        Rstd_ptr,
        stride_dx,
        stride_dy,
        stride_x,
        stride_dw,
        stride_db,
        M,
        N,
        NUM_WORKERS,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32) if HAS_WEIGHT else 1.0
        dw_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        db_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        row = pid
        while row < M:
            x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY_ptr + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
            mean = tl.load(Mean_ptr + row).to(tl.float32)
            rstd = tl.load(Rstd_ptr + row).to(tl.float32)

            x_hat = tl.where(mask, (x - mean) * rstd, 0.0)
            wdy = tl.where(mask, dy * w, 0.0)
            c1 = tl.sum(x_hat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (x_hat * c1 + c2)) * rstd

            tl.store(DX_ptr + row * stride_dx + cols, dx, mask=mask)
            if HAS_WEIGHT:
                dw_acc += dy * x_hat
            if HAS_BIAS:
                db_acc += dy
            row += NUM_WORKERS

        if HAS_WEIGHT:
            tl.store(PartialDW_ptr + pid * stride_dw + cols, dw_acc, mask=mask)
        if HAS_BIAS:
            tl.store(PartialDB_ptr + pid * stride_db + cols, db_acc, mask=mask)


def _select_num_warps(block_n: int) -> int:
    if block_n <= 1024:
        return 4
    if block_n <= 4096:
        return 8
    return 16


class _TritonLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        orig_shape = x.shape
        N = orig_shape[-1]
        x_2d = x.reshape(-1, N).contiguous()
        M = x_2d.shape[0]
        y_2d = torch.empty_like(x_2d)
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        has_w = weight is not None
        has_b = bias is not None
        block_n = triton.next_power_of_2(N)
        warps = _select_num_warps(block_n)

        _layernorm_fwd_kernel[(M,)](
            x_2d,
            weight if has_w else x_2d,
            bias if has_b else x_2d,
            y_2d,
            mean,
            rstd,
            x_2d.stride(0),
            y_2d.stride(0),
            N,
            eps,
            HAS_WEIGHT=has_w,
            HAS_BIAS=has_b,
            BLOCK_N=block_n,
            num_warps=warps,
        )
        ctx.save_for_backward(x_2d, weight, bias, mean, rstd)
        ctx.orig_shape = orig_shape
        ctx.block_n = block_n
        ctx.num_warps = warps
        ctx.has_w = has_w
        ctx.has_b = has_b
        return y_2d.reshape(orig_shape)

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, None]:
        x_2d, weight, bias, mean, rstd = ctx.saved_tensors
        M, N = x_2d.shape
        dy_2d = dy.reshape(M, N).contiguous()
        dx_2d = torch.empty_like(x_2d)

        num_workers = min(M, 256)
        partial_dw = torch.empty((num_workers, N), dtype=torch.float32, device=x_2d.device) if ctx.has_w else dx_2d
        partial_db = torch.empty((num_workers, N), dtype=torch.float32, device=x_2d.device) if ctx.has_b else dx_2d

        _layernorm_bwd_kernel[(num_workers,)](
            dx_2d,
            partial_dw,
            partial_db,
            dy_2d,
            x_2d,
            weight if ctx.has_w else x_2d,
            mean,
            rstd,
            dx_2d.stride(0),
            dy_2d.stride(0),
            x_2d.stride(0),
            partial_dw.stride(0) if ctx.has_w else 1,
            partial_db.stride(0) if ctx.has_b else 1,
            M,
            N,
            num_workers,
            HAS_WEIGHT=ctx.has_w,
            HAS_BIAS=ctx.has_b,
            BLOCK_N=ctx.block_n,
            num_warps=ctx.num_warps,
        )
        dw = partial_dw.sum(dim=0).to(weight.dtype) if (ctx.has_w and weight is not None) else None
        db = partial_db.sum(dim=0).to(bias.dtype) if (ctx.has_b and bias is not None) else None
        return dx_2d.reshape(ctx.orig_shape), dw, db, None


def triton_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute LayerNorm via custom Triton kernel when eligible, else PyTorch ref."""
    if (
        not HAS_TRITON
        or not x.is_cuda
        or (weight is not None and not weight.is_cuda)
        or x.shape[-1] > _MAX_FUSED_COLS
        or x.numel() == 0
        or x.dtype not in (torch.float32, torch.float16, torch.bfloat16)
    ):
        return ref_layernorm(x, weight, bias=bias, eps=eps)
    return _TritonLayerNormFn.apply(x, weight, bias, float(eps))
