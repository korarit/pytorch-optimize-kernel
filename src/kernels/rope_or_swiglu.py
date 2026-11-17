"""Custom Triton Fused SwiGLU Activation (`silu(gate) * up`) Forward + Backward."""

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


def ref_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """PyTorch reference SwiGLU elementwise activation."""
    return F.silu(gate) * up


if HAS_TRITON:

    @triton.jit
    def _swiglu_fwd_kernel(Gate_ptr, Up_ptr, Out_ptr, N_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N_elements

        g = tl.load(Gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(Up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        out = (g * sig) * u
        tl.store(Out_ptr + offs, out, mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        DOut_ptr,
        Gate_ptr,
        Up_ptr,
        DGate_ptr,
        DUp_ptr,
        N_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N_elements

        dout = tl.load(DOut_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(Gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(Up_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        sig = tl.sigmoid(g)
        silu = g * sig
        d_up = dout * silu
        d_silu_dg = sig * (1.0 + g * (1.0 - sig))
        d_gate = dout * u * d_silu_dg

        tl.store(DGate_ptr + offs, d_gate, mask=mask)
        tl.store(DUp_ptr + offs, d_up, mask=mask)


class _TritonSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        g_c = gate.contiguous()
        u_c = up.contiguous()
        out = torch.empty_like(g_c)
        n = g_c.numel()
        block_size = 1024
        grid = (triton.cdiv(n, block_size),)
        _swiglu_fwd_kernel[grid](g_c, u_c, out, n, BLOCK_SIZE=block_size)
        ctx.save_for_backward(g_c, u_c)
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g_c, u_c = ctx.saved_tensors
        dout_c = dout.contiguous()
        d_gate = torch.empty_like(g_c)
        d_up = torch.empty_like(u_c)
        n = g_c.numel()
        block_size = 1024
        grid = (triton.cdiv(n, block_size),)
        _swiglu_bwd_kernel[grid](dout_c, g_c, u_c, d_gate, d_up, n, BLOCK_SIZE=block_size)
        return d_gate, d_up


def triton_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute `F.silu(gate) * up` via fused Triton kernel when eligible, else PyTorch ref."""
    if (
        not HAS_TRITON
        or not gate.is_cuda
        or not up.is_cuda
        or gate.shape != up.shape
        or gate.numel() == 0
        or gate.dtype not in (torch.float32, torch.float16, torch.bfloat16)
    ):
        return ref_swiglu(gate, up)
    return _TritonSwiGLUFn.apply(gate, up)
