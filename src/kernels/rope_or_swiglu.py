"""Custom Triton Fused SwiGLU (`silu(gate) * up`) & Rotary Position Embedding (RoPE) Kernels."""

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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def ref_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference half-split Rotary Position Embedding (HuggingFace LLaMA/Qwen/Mistral compatible)."""
    d = q.shape[-1]
    if cos.shape[-1] == d // 2:
        cos = torch.cat((cos, cos), dim=-1)
    if sin.shape[-1] == d // 2:
        sin = torch.cat((sin, sin), dim=-1)
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    cos_f = cos.to(q.dtype)
    sin_f = sin.to(q.dtype)
    q_embed = (q * cos_f) + (_rotate_half(q) * sin_f)
    cos_kf = cos.to(k.dtype)
    sin_kf = sin.to(k.dtype)
    k_embed = (k * cos_kf) + (_rotate_half(k) * sin_kf)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def _to_row_strided_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Return a 2D row-strided view (`stride(-1) == 1`) without copying whenever possible."""
    if tensor.dim() == 2 and tensor.stride(1) == 1:
        return tensor
    last_dim = tensor.shape[-1]
    if tensor.dim() >= 1 and tensor.stride(-1) == 1:
        try:
            viewed = tensor.view(-1, last_dim)
            if viewed.stride(1) == 1:
                return viewed
        except RuntimeError:
            pass
    return tensor.reshape(-1, last_dim).contiguous()


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
    def _swiglu_fwd_strided_kernel(
        Gate_ptr,
        Up_ptr,
        Out_ptr,
        stride_g,
        stride_u,
        stride_out,
        N_cols,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_blk = tl.program_id(1)
        cols = col_blk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N_cols

        g = tl.load(Gate_ptr + row * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(Up_ptr + row * stride_u + cols, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        out = (g * sig) * u
        tl.store(Out_ptr + row * stride_out + cols, out, mask=mask)

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

    @triton.jit
    def _swiglu_bwd_strided_kernel(
        DOut_ptr,
        Gate_ptr,
        Up_ptr,
        DGate_ptr,
        DUp_ptr,
        stride_dout,
        stride_g,
        stride_u,
        stride_dg,
        stride_du,
        N_cols,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_blk = tl.program_id(1)
        cols = col_blk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N_cols

        dout = tl.load(DOut_ptr + row * stride_dout + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(Gate_ptr + row * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(Up_ptr + row * stride_u + cols, mask=mask, other=0.0).to(tl.float32)

        sig = tl.sigmoid(g)
        silu = g * sig
        d_up = dout * silu
        d_silu_dg = sig * (1.0 + g * (1.0 - sig))
        d_gate = dout * u * d_silu_dg

        tl.store(DGate_ptr + row * stride_dg + cols, d_gate, mask=mask)
        tl.store(DUp_ptr + row * stride_du + cols, d_up, mask=mask)

    @triton.jit
    def _rope_fwd_bwd_kernel(
        Q_ptr,
        K_ptr,
        Cos_ptr,
        Sin_ptr,
        QOut_ptr,
        KOut_ptr,
        stride_q,
        stride_k,
        stride_qo,
        stride_ko,
        stride_cb,
        stride_cs,
        stride_sb,
        stride_ss,
        M_q,
        M_k,
        H_q,
        H_k,
        S_len,
        HALF_D,
        IS_BWD: tl.constexpr,
        COS_FULL_D: tl.constexpr,
        EXACT_ROWS: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_HALF)
        mask = cols < HALF_D

        if row < M_q:
            if EXACT_ROWS:
                c_off_q = row * stride_cs
                s_off_q = row * stride_ss
            else:
                b_q = row // (H_q * S_len)
                s_q = row % S_len
                c_off_q = b_q * stride_cb + s_q * stride_cs
                s_off_q = b_q * stride_sb + s_q * stride_ss

            c1 = tl.load(Cos_ptr + c_off_q + cols, mask=mask, other=1.0).to(tl.float32)
            s1 = tl.load(Sin_ptr + s_off_q + cols, mask=mask, other=0.0).to(tl.float32)
            if COS_FULL_D:
                c2 = tl.load(Cos_ptr + c_off_q + HALF_D + cols, mask=mask, other=1.0).to(tl.float32)
                s2 = tl.load(Sin_ptr + s_off_q + HALF_D + cols, mask=mask, other=0.0).to(tl.float32)
            else:
                c2 = c1
                s2 = s1

            q1 = tl.load(Q_ptr + row * stride_q + cols, mask=mask, other=0.0).to(tl.float32)
            q2 = tl.load(Q_ptr + row * stride_q + HALF_D + cols, mask=mask, other=0.0).to(tl.float32)
            if IS_BWD:
                qo1 = q1 * c1 + q2 * s2
                qo2 = q2 * c2 - q1 * s1
            else:
                qo1 = q1 * c1 - q2 * s1
                qo2 = q2 * c2 + q1 * s2
            tl.store(QOut_ptr + row * stride_qo + cols, qo1, mask=mask)
            tl.store(QOut_ptr + row * stride_qo + HALF_D + cols, qo2, mask=mask)

        if row < M_k:
            if EXACT_ROWS:
                c_off_k = row * stride_cs
                s_off_k = row * stride_ss
            else:
                b_k = row // (H_k * S_len)
                s_k = row % S_len
                c_off_k = b_k * stride_cb + s_k * stride_cs
                s_off_k = b_k * stride_sb + s_k * stride_ss

            c1k = tl.load(Cos_ptr + c_off_k + cols, mask=mask, other=1.0).to(tl.float32)
            s1k = tl.load(Sin_ptr + s_off_k + cols, mask=mask, other=0.0).to(tl.float32)
            if COS_FULL_D:
                c2k = tl.load(Cos_ptr + c_off_k + HALF_D + cols, mask=mask, other=1.0).to(tl.float32)
                s2k = tl.load(Sin_ptr + s_off_k + HALF_D + cols, mask=mask, other=0.0).to(tl.float32)
            else:
                c2k = c1k
                s2k = s1k

            k1 = tl.load(K_ptr + row * stride_k + cols, mask=mask, other=0.0).to(tl.float32)
            k2 = tl.load(K_ptr + row * stride_k + HALF_D + cols, mask=mask, other=0.0).to(tl.float32)
            if IS_BWD:
                ko1 = k1 * c1k + k2 * s2k
                ko2 = k2 * c2k - k1 * s1k
            else:
                ko1 = k1 * c1k - k2 * s1k
                ko2 = k2 * c2k + k1 * s2k
            tl.store(KOut_ptr + row * stride_ko + cols, ko1, mask=mask)
            tl.store(KOut_ptr + row * stride_ko + HALF_D + cols, ko2, mask=mask)


class _TritonSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        orig_shape = gate.shape
        g_2d = _to_row_strided_2d(gate)
        u_2d = _to_row_strided_2d(up)
        M, N = g_2d.shape
        out_2d = torch.empty((M, N), dtype=gate.dtype, device=gate.device)

        if g_2d.is_contiguous() and u_2d.is_contiguous():
            n = g_2d.numel()
            block_size = 1024
            grid = (triton.cdiv(n, block_size),)
            _swiglu_fwd_kernel[grid](g_2d, u_2d, out_2d, n, BLOCK_SIZE=block_size)
        else:
            block_n = 1024
            grid = (M, triton.cdiv(N, block_n))
            _swiglu_fwd_strided_kernel[grid](
                g_2d,
                u_2d,
                out_2d,
                g_2d.stride(0),
                u_2d.stride(0),
                out_2d.stride(0),
                N,
                BLOCK_N=block_n,
            )
        ctx.save_for_backward(g_2d, u_2d)
        ctx.orig_shape = orig_shape
        return out_2d.reshape(orig_shape)

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g_2d, u_2d = ctx.saved_tensors
        M, N = g_2d.shape
        dout_2d = _to_row_strided_2d(dout)
        d_gate = torch.empty((M, N), dtype=g_2d.dtype, device=g_2d.device)
        d_up = torch.empty((M, N), dtype=u_2d.dtype, device=u_2d.device)

        if g_2d.is_contiguous() and u_2d.is_contiguous() and dout_2d.is_contiguous():
            n = g_2d.numel()
            block_size = 1024
            grid = (triton.cdiv(n, block_size),)
            _swiglu_bwd_kernel[grid](dout_2d, g_2d, u_2d, d_gate, d_up, n, BLOCK_SIZE=block_size)
        else:
            block_n = 1024
            grid = (M, triton.cdiv(N, block_n))
            _swiglu_bwd_strided_kernel[grid](
                dout_2d,
                g_2d,
                u_2d,
                d_gate,
                d_up,
                dout_2d.stride(0),
                g_2d.stride(0),
                u_2d.stride(0),
                d_gate.stride(0),
                d_up.stride(0),
                N,
                BLOCK_N=block_n,
            )
        orig_shape = ctx.orig_shape
        return d_gate.reshape(orig_shape), d_up.reshape(orig_shape)


def _resolve_rope_table_strides(
    table: torch.Tensor,
    q_shape: tuple[int, ...],
) -> tuple[torch.Tensor, int, int, bool] | None:
    """Map `cos`/`sin` table onto `(b, s)` strides for `[B, H, S, D]` / `[S, D]` without copying."""
    if table.stride(-1) != 1:
        table = table.contiguous()
    if table.shape[:-1] == q_shape[:-1]:
        t2d = _to_row_strided_2d(table)
        return t2d, 0, t2d.stride(0), True
    if len(q_shape) == 4:
        B, _H, S, _D = q_shape
        if table.dim() == 2 and table.shape[0] == S:
            return table, 0, table.stride(0), False
        if table.dim() == 3 and table.shape[1] == S and table.shape[0] in (1, B):
            return table, (0 if table.shape[0] == 1 else table.stride(0)), table.stride(1), False
        if table.dim() == 4 and table.shape[2] == S and table.shape[1] == 1 and table.shape[0] in (1, B):
            return table, (0 if table.shape[0] == 1 else table.stride(0)), table.stride(2), False
    elif len(q_shape) == 3:
        B, S, _D = q_shape
        if table.dim() == 2 and table.shape[0] == S:
            return table, 0, table.stride(0), False
        if table.dim() == 3 and table.shape[1] == S and table.shape[0] in (1, B):
            return table, (0 if table.shape[0] == 1 else table.stride(0)), table.stride(1), False
    return None


class _TritonRoPEFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_shape = tuple(q.shape)
        k_shape = tuple(k.shape)
        res_c = _resolve_rope_table_strides(cos, q_shape)
        res_s = _resolve_rope_table_strides(sin, q_shape)
        assert res_c is not None and res_s is not None
        c_tbl, sc_b, sc_s, exact_c = res_c
        s_tbl, ss_b, ss_s, exact_s = res_s
        if exact_c != exact_s:
            # Broadcast mismatch between cos and sin exact vs strided -> align both to q_shape
            c_tbl = _to_row_strided_2d(cos.expand(*q_shape[:-1], cos.shape[-1]).contiguous())
            s_tbl = _to_row_strided_2d(sin.expand(*q_shape[:-1], sin.shape[-1]).contiguous())
            sc_b, sc_s = 0, c_tbl.stride(0)
            ss_b, ss_s = 0, s_tbl.stride(0)
            exact_rows = True
        else:
            exact_rows = exact_c and (q_shape == k_shape)

        q_2d = _to_row_strided_2d(q)
        k_2d = _to_row_strided_2d(k)
        M_q, D = q_2d.shape
        M_k = k_2d.shape[0]
        half_d = D // 2
        q_out = torch.empty((M_q, D), dtype=q.dtype, device=q.device)
        k_out = torch.empty((M_k, D), dtype=k.dtype, device=k.device)

        h_q = q_shape[1] if len(q_shape) == 4 else 1
        h_k = k_shape[1] if len(k_shape) == 4 else 1
        s_len = q_shape[-2] if len(q_shape) >= 2 else 1
        cos_full_d = cos.shape[-1] == D and sin.shape[-1] == D
        block_half = max(16, triton.next_power_of_2(half_d))
        warps = 4 if block_half <= 128 else 8
        M_max = max(M_q, M_k)

        _rope_fwd_bwd_kernel[(M_max,)](
            q_2d,
            k_2d,
            c_tbl,
            s_tbl,
            q_out,
            k_out,
            q_2d.stride(0),
            k_2d.stride(0),
            q_out.stride(0),
            k_out.stride(0),
            sc_b,
            sc_s,
            ss_b,
            ss_s,
            M_q,
            M_k,
            h_q,
            h_k,
            s_len,
            half_d,
            IS_BWD=False,
            COS_FULL_D=cos_full_d,
            EXACT_ROWS=exact_rows,
            BLOCK_HALF=block_half,
            num_warps=warps,
        )
        ctx.save_for_backward(c_tbl, s_tbl)
        ctx.meta = (q_shape, k_shape, sc_b, sc_s, ss_b, ss_s, cos_full_d, exact_rows, block_half, warps)
        return q_out.reshape(q_shape), k_out.reshape(k_shape)

    @staticmethod
    def backward(ctx: Any, dq_out: torch.Tensor, dk_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        c_tbl, s_tbl = ctx.saved_tensors
        q_shape, k_shape, sc_b, sc_s, ss_b, ss_s, cos_full_d, exact_rows, block_half, warps = ctx.meta
        dq_2d = _to_row_strided_2d(dq_out)
        dk_2d = _to_row_strided_2d(dk_out)
        M_q, D = dq_2d.shape
        M_k = dk_2d.shape[0]
        half_d = D // 2
        dq_in = torch.empty((M_q, D), dtype=dq_out.dtype, device=dq_out.device)
        dk_in = torch.empty((M_k, D), dtype=dk_out.dtype, device=dk_out.device)
        h_q = q_shape[1] if len(q_shape) == 4 else 1
        h_k = k_shape[1] if len(k_shape) == 4 else 1
        s_len = q_shape[-2] if len(q_shape) >= 2 else 1

        _rope_fwd_bwd_kernel[(max(M_q, M_k),)](
            dq_2d,
            dk_2d,
            c_tbl,
            s_tbl,
            dq_in,
            dk_in,
            dq_2d.stride(0),
            dk_2d.stride(0),
            dq_in.stride(0),
            dk_in.stride(0),
            sc_b,
            sc_s,
            ss_b,
            ss_s,
            M_q,
            M_k,
            h_q,
            h_k,
            s_len,
            half_d,
            IS_BWD=True,
            COS_FULL_D=cos_full_d,
            EXACT_ROWS=exact_rows,
            BLOCK_HALF=block_half,
            num_warps=warps,
        )
        return dq_in.reshape(q_shape), dk_in.reshape(k_shape), None, None


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


def triton_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fused Rotary Position Embedding on `(q, k)` via Triton when eligible, else PyTorch ref."""
    d = q.shape[-1] if q.dim() >= 2 else 0
    eligible = (
        HAS_TRITON
        and q.is_cuda
        and k.is_cuda
        and cos.is_cuda
        and sin.is_cuda
        and q.dim() >= 2
        and k.dim() >= 2
        and q.shape[-1] == k.shape[-1]
        and q.shape[-2] == k.shape[-2]
        and q.shape[0] == k.shape[0]
        and d > 0
        and d % 2 == 0
        and d <= 1024
        and q.numel() > 0
        and k.numel() > 0
        and cos.shape[-1] in (d, d // 2)
        and sin.shape[-1] == cos.shape[-1]
        and q.dtype == k.dtype
        and q.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and (_resolve_rope_table_strides(cos, tuple(q.shape)) is not None)
        and (_resolve_rope_table_strides(sin, tuple(q.shape)) is not None)
    )
    if not eligible:
        return ref_rope(q, k, cos, sin)
    return _TritonRoPEFn.apply(q, k, cos, sin)
