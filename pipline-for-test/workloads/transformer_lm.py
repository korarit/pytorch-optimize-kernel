#!/usr/bin/env python3
"""Workload 2: Decoder-only Tiny Transformer LM (SDPA, RMSNorm, SwiGLU, CrossEntropy)."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import (
    build_base_parser,
    emit_summary,
    load_checkpoint,
    read_peak_memory_mb,
    reset_peak_memory,
    resolve_device,
    save_checkpoint,
    seed_everything,
    sync_device,
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.out_proj(attn)


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_model * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int = 2048, d_model: int = 256, n_heads: int = 4, n_layers: int = 4, max_seq: int = 256) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = idx.shape
        pos = torch.arange(seq_len, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm_f(x))


def main() -> int:
    parser = build_base_parser("Workload 2: Tiny Transformer LM Training", default_steps=20, default_batch_size=16)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    device = resolve_device(args.device)
    seed_everything(args.seed)

    model = TinyGPT(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq=max(256, args.seq_len),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    start_step = 0
    if args.load_ckpt is not None:
        start_step = load_checkpoint(args.load_ckpt, model, optimizer, device=device)

    data_gen = torch.Generator().manual_seed(args.seed + 1000)
    batches = [
        (
            torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), generator=data_gen),
            torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), generator=data_gen),
        )
        for _ in range(args.steps)
    ]

    model.train()
    reset_peak_memory(device)
    sync_device(device)
    t0 = time.perf_counter()

    losses: list[float] = []
    step_times_ms: list[float] = []
    for x, y in batches:
        step_t0 = time.perf_counter()
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        sync_device(device)
        step_times_ms.append((time.perf_counter() - step_t0) * 1000.0)
        losses.append(float(loss.detach().cpu()))

    sync_device(device)
    elapsed = time.perf_counter() - t0
    alloc_mb, resv_mb = read_peak_memory_mb(device)

    if args.save_ckpt is not None:
        save_checkpoint(args.save_ckpt, model, optimizer, step=start_step + args.steps)

    emit_summary(
        "transformer_lm",
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        elapsed_sec=elapsed,
        losses=losses,
        peak_alloc_mb=alloc_mb,
        peak_reserved_mb=resv_mb,
        step_times_ms=step_times_ms,
        extra={"tokens_per_sec": round((args.steps * args.batch_size * args.seq_len) / max(1e-9, elapsed), 2)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
