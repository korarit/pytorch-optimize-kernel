"""Custom Triton Fused Online-Softmax CrossEntropy (Forward + Backward) saving logits memory."""

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


def ref_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """PyTorch reference CrossEntropy computed cleanly in fp32."""
    return F.cross_entropy(logits.float(), targets, ignore_index=ignore_index, reduction=reduction)


if HAS_TRITON:

    @triton.jit
    def _cross_entropy_fwd_kernel(
        Logits_ptr,
        Targets_ptr,
        Losses_ptr,
        LSE_ptr,
        stride_logits,
        V,
        ignore_index,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        target = tl.load(Targets_ptr + row)

        if target == ignore_index or target < 0 or target >= V:
            tl.store(Losses_ptr + row, 0.0)
            tl.store(LSE_ptr + row, 0.0)
            return

        row_ptr = Logits_ptr + row * stride_logits
        m = float("-inf")
        d = 0.0

        for start in range(0, V, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            mask = cols < V
            x = tl.load(row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
            block_max = tl.max(x, axis=0)
            m_new = tl.maximum(m, block_max)
            d = d * tl.exp(m - m_new) + tl.sum(tl.where(mask, tl.exp(x - m_new), 0.0), axis=0)
            m = m_new

        lse = m + tl.log(d)
        x_target = tl.load(row_ptr + target).to(tl.float32)
        loss = lse - x_target

        tl.store(Losses_ptr + row, loss)
        tl.store(LSE_ptr + row, lse)

    @triton.jit
    def _cross_entropy_bwd_kernel(
        DX_ptr,
        Logits_ptr,
        Targets_ptr,
        LSE_ptr,
        GradOut_ptr,
        stride_dx,
        stride_logits,
        V,
        ignore_index,
        IS_SCALAR_GRAD: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        target = tl.load(Targets_ptr + row)
        dx_row = DX_ptr + row * stride_dx
        x_row = Logits_ptr + row * stride_logits

        if target == ignore_index or target < 0 or target >= V:
            for start in range(0, V, BLOCK_V):
                cols = start + tl.arange(0, BLOCK_V)
                mask = cols < V
                tl.store(dx_row + cols, 0.0, mask=mask)
            return

        lse = tl.load(LSE_ptr + row).to(tl.float32)
        g_out = tl.load(GradOut_ptr).to(tl.float32) if IS_SCALAR_GRAD else tl.load(GradOut_ptr + row).to(tl.float32)

        for start in range(0, V, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            mask = cols < V
            x = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
            probs = tl.exp(x - lse)
            grad = tl.where(cols == target, probs - 1.0, probs) * g_out
            tl.store(dx_row + cols, grad, mask=mask)


class _TritonCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int,
        reduction: str,
    ) -> torch.Tensor:
        logits_2d = logits.contiguous()
        targets_1d = targets.contiguous()
        M, V = logits_2d.shape
        losses = torch.empty((M,), dtype=torch.float32, device=logits.device)
        lse = torch.empty((M,), dtype=torch.float32, device=logits.device)

        block_v = min(4096, max(128, triton.next_power_of_2(V)))
        warps = 8 if block_v >= 2048 else 4
        _cross_entropy_fwd_kernel[(M,)](
            logits_2d,
            targets_1d,
            losses,
            lse,
            logits_2d.stride(0),
            V,
            ignore_index,
            BLOCK_V=block_v,
            num_warps=warps,
        )

        valid_count_t: torch.Tensor | None = None
        if reduction == "mean":
            valid_mask = (targets_1d != ignore_index) & (targets_1d >= 0) & (targets_1d < V)
            valid_count_t = valid_mask.sum().to(torch.float32).clamp(min=1.0)

        ctx.save_for_backward(logits_2d, targets_1d, lse, valid_count_t)
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.block_v = block_v
        ctx.num_warps = warps

        if reduction == "none":
            return losses
        if reduction == "sum":
            return losses.sum()
        assert valid_count_t is not None
        return losses.sum() / valid_count_t

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        logits_2d, targets_1d, lse, valid_count_t = ctx.saved_tensors
        M, V = logits_2d.shape
        dx = torch.empty_like(logits_2d)

        is_scalar = ctx.reduction in ("mean", "sum")
        if ctx.reduction == "mean":
            assert valid_count_t is not None
            grad_in = grad_out.float().reshape(()) / valid_count_t
        elif ctx.reduction == "sum":
            grad_in = grad_out.float().reshape(())
        else:
            grad_in = grad_out.float().reshape(-1).contiguous()

        _cross_entropy_bwd_kernel[(M,)](
            dx,
            logits_2d,
            targets_1d,
            lse,
            grad_in,
            dx.stride(0),
            logits_2d.stride(0),
            V,
            ctx.ignore_index,
            IS_SCALAR_GRAD=is_scalar,
            BLOCK_V=ctx.block_v,
            num_warps=ctx.num_warps,
        )
        return dx, None, None, None


def triton_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute fused online-softmax CrossEntropy via Triton when eligible, else PyTorch ref."""
    if (
        not HAS_TRITON
        or not logits.is_cuda
        or not targets.is_cuda
        or logits.dim() != 2
        or targets.dim() != 1
        or logits.numel() == 0
        or reduction not in ("mean", "sum", "none")
        or logits.dtype not in (torch.float32, torch.float16, torch.bfloat16)
    ):
        return ref_cross_entropy(logits, targets, ignore_index=ignore_index, reduction=reduction)
    return _TritonCrossEntropyFn.apply(logits, targets, int(ignore_index), reduction)
