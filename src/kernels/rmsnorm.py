"""Custom Triton RMSNorm (Forward + Backward) with PyTorch reference fallback."""

from __future__ import annotations

from typing import Any

import torch

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]
    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU dev environment
    triton = None
    tl = None
    HAS_TRITON = False

_MAX_FUSED_COLS = 16384


def ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically accurate PyTorch reference RMSNorm computed in fp32."""
    orig_dtype = x.dtype
    x_f = x.float()
    rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f * rstd * weight.float()).to(orig_dtype)


if HAS_TRITON:

    @triton.jit
    def _rmsnorm_fwd_kernel(
        X_ptr,
        W_ptr,
        Y_ptr,
        Rstd_ptr,
        stride_x,
        stride_y,
        N,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = tl.math.rsqrt(mean_sq + eps)
        tl.store(Rstd_ptr + row, rstd)

        y = (x * rstd) * w
        tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)

    @triton.jit
    def _rmsnorm_bwd_kernel(
        DX_ptr,
        PartialDW_ptr,
        DY_ptr,
        X_ptr,
        W_ptr,
        Rstd_ptr,
        stride_dx,
        stride_dy,
        stride_x,
        stride_dw,
        M,
        N,
        NUM_WORKERS,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dw_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        row = pid
        while row < M:
            x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY_ptr + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
            rstd = tl.load(Rstd_ptr + row).to(tl.float32)

            x_hat = x * rstd
            wdy = dy * w
            c1 = tl.sum(wdy * x_hat, axis=0) / N
            dx = rstd * (wdy - x_hat * c1)

            tl.store(DX_ptr + row * stride_dx + cols, dx, mask=mask)
            dw_acc += dy * x_hat
            row += NUM_WORKERS

        tl.store(PartialDW_ptr + pid * stride_dw + cols, dw_acc, mask=mask)


class _TritonRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        orig_shape = x.shape
        N = orig_shape[-1]
        x_2d = x.reshape(-1, N).contiguous()
        M = x_2d.shape[0]
        y_2d = torch.empty_like(x_2d)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        block_n = triton.next_power_of_2(N)
        _rmsnorm_fwd_kernel[(M,)](
            x_2d,
            weight,
            y_2d,
            rstd,
            x_2d.stride(0),
            y_2d.stride(0),
            N,
            eps,
            BLOCK_N=block_n,
        )
        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.orig_shape = orig_shape
        ctx.block_n = block_n
        return y_2d.reshape(orig_shape)

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x_2d, weight, rstd = ctx.saved_tensors
        M, N = x_2d.shape
        dy_2d = dy.reshape(M, N).contiguous()
        dx_2d = torch.empty_like(x_2d)

        num_workers = min(M, 128)
        partial_dw = torch.empty((num_workers, N), dtype=torch.float32, device=x_2d.device)
        _rmsnorm_bwd_kernel[(num_workers,)](
            dx_2d,
            partial_dw,
            dy_2d,
            x_2d,
            weight,
            rstd,
            dx_2d.stride(0),
            dy_2d.stride(0),
            x_2d.stride(0),
            partial_dw.stride(0),
            M,
            N,
            num_workers,
            BLOCK_N=ctx.block_n,
        )
        dw = partial_dw.sum(dim=0).to(weight.dtype)
        return dx_2d.reshape(ctx.orig_shape), dw, None


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute RMSNorm via custom Triton kernel when eligible, else fall back to PyTorch ref."""
    if (
        not HAS_TRITON
        or not x.is_cuda
        or not weight.is_cuda
        or x.shape[-1] > _MAX_FUSED_COLS
        or x.numel() == 0
        or x.dtype not in (torch.float32, torch.float16, torch.bfloat16)
    ):
        return ref_rmsnorm(x, weight, eps=eps)
    return _TritonRMSNormFn.apply(x, weight, float(eps))
