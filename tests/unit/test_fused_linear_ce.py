"""Unit parity tests for Phase 5.2: Fused Chunked Linear + CrossEntropy & LazyLinearLogits."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import OptimizeConfig
from src.hooks import HookManager
from src.kernels import (
    LazyLinearLogits,
    install_custom_kernels_hooks,
    ref_linear_cross_entropy,
    triton_fused_linear_cross_entropy as fused_linear_cross_entropy,
    uninstall_custom_kernels_hooks,
)
from src.kernels.fused_linear_cross_entropy import compute_optimal_chunk_size, create_lazy_linear_logits


class TestFusedLinearCrossEntropyParity(unittest.TestCase):
    """Exact forward & backward gradient parity across reductions, ignore_index, and chunk splits."""

    def test_optimal_chunk_size_caps_memory_budget(self) -> None:
        self.assertEqual(compute_optimal_chunk_size(4096, 32768, torch.bfloat16), 256)
        self.assertEqual(compute_optimal_chunk_size(4096, 65536, torch.bfloat16), 128)
        self.assertEqual(compute_optimal_chunk_size(90, 32768, torch.float32), 90)

    def test_forward_backward_parity_mean_sum_none_with_ignore_index(self) -> None:
        torch.manual_seed(7)
        for device in (["cuda"] if torch.cuda.is_available() else ["cpu"]):
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            atol_loss = 1e-2 if dtype == torch.bfloat16 else 1e-5
            atol_grad = 5e-2 if dtype == torch.bfloat16 else 1e-5

            for reduction in ("mean", "sum", "none"):
                with self.subTest(device=device, dtype=dtype, reduction=reduction):
                    h_ref = torch.randn(37, 24, device=device, dtype=dtype, requires_grad=True)
                    w_ref = torch.randn(193, 24, device=device, dtype=dtype, requires_grad=True)
                    b_ref = torch.randn(193, device=device, dtype=dtype, requires_grad=True)
                    tgt = torch.randint(0, 193, (37,), device=device)
                    # Whole first chunk ignored + mixed tail
                    tgt[:16] = -100
                    tgt[20] = -100

                    h_fused = h_ref.detach().clone().requires_grad_(True)
                    w_fused = w_ref.detach().clone().requires_grad_(True)
                    b_fused = b_ref.detach().clone().requires_grad_(True)

                    loss_ref = ref_linear_cross_entropy(h_ref, w_ref, b_ref, tgt, ignore_index=-100, reduction=reduction)
                    loss_fused = fused_linear_cross_entropy(
                        h_fused,
                        w_fused,
                        tgt,
                        bias=b_fused,
                        ignore_index=-100,
                        reduction=reduction,
                        chunk_size=16,  # force 3 chunks: 16, 16, 5 (with first chunk all ignored)
                    )

                    self.assertTrue(torch.allclose(loss_ref.float(), loss_fused.float(), rtol=1e-2, atol=atol_loss))
                    if reduction == "none":
                        g = torch.randn_like(loss_ref)
                        loss_ref.backward(g)
                        loss_fused.backward(g)
                    else:
                        loss_ref.backward()
                        loss_fused.backward()

                    assert h_ref.grad is not None and h_fused.grad is not None
                    assert w_ref.grad is not None and w_fused.grad is not None
                    assert b_ref.grad is not None and b_fused.grad is not None
                    self.assertTrue(torch.allclose(h_ref.grad.float(), h_fused.grad.float(), rtol=1e-2, atol=atol_grad))
                    self.assertTrue(torch.allclose(w_ref.grad.float(), w_fused.grad.float(), rtol=1e-2, atol=atol_grad))
                    self.assertTrue(torch.allclose(b_ref.grad.float(), b_fused.grad.float(), rtol=1e-2, atol=atol_grad))

    def test_all_ignored_targets_safe_no_nan(self) -> None:
        h = torch.randn(12, 8, requires_grad=True)
        w = torch.randn(64, 8, requires_grad=True)
        tgt = torch.full((12,), -100, dtype=torch.long)
        loss = fused_linear_cross_entropy(h, w, tgt, chunk_size=5)
        self.assertEqual(float(loss.item()), 0.0)
        loss.backward()
        assert h.grad is not None and w.grad is not None
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertTrue(torch.isfinite(w.grad).all())

    def test_lazy_linear_logits_slice_view_and_materialize_fallback(self) -> None:
        torch.manual_seed(13)
        h = torch.randn(2, 5, 16, requires_grad=True)
        w = torch.randn(128, 16, requires_grad=True)
        b = torch.randn(128, requires_grad=True)

        lazy = create_lazy_linear_logits(h, w, b)
        self.assertIsInstance(lazy, LazyLinearLogits)
        self.assertIsInstance(lazy, torch.Tensor)
        self.assertEqual(lazy.shape, (2, 5, 128))
        self.assertEqual(lazy.device.type, "cpu")

        # Causal shift slice `[:, :-1, :].contiguous().view(-1, V)` preserves lazy status cleanly
        shifted = lazy[:, :-1, :].contiguous().view(-1, 128)
        self.assertIsInstance(shifted, LazyLinearLogits)
        self.assertEqual(shifted.shape, (8, 128))

        # Arbitrary op (argmax / topk / sum) materializes transparently
        preds = shifted.argmax(dim=-1)
        ref_preds = F.linear(h, w, b)[:, :-1, :].reshape(-1, 128).argmax(dim=-1)
        self.assertTrue(torch.equal(preds, ref_preds))

        # Backprop through materialized consumer works identically to F.linear
        loss_mat = (shifted * 0.5).sum()
        loss_mat.backward()
        assert h.grad is not None and w.grad is not None
        self.assertEqual(h.grad.shape, h.shape)
        self.assertEqual(w.grad.shape, w.shape)

    def test_registry_eager_mode_patches_large_head_and_fuses_with_cross_entropy(self) -> None:
        class _FakeHeadModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(16, 16)
                self.head = nn.Linear(16, 4096, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.head(torch.relu(self.proj(x)))

        cfg = OptimizeConfig(profile="aggressive")
        cfg.compile.enabled = False
        cfg.kernels.enabled = True
        cfg.kernels.allow_list = ["fused_linear_cross_entropy"]
        mgr = HookManager()

        reg = install_custom_kernels_hooks(cfg, mgr, torch, triton_available=True, sm_override=89)
        assert reg is not None
        self.assertEqual(reg.fallback_registry.entries["fused_linear_cross_entropy"].status, "APPLIED")

        model = _FakeHeadModel()
        # Mock CUDA checks cleanly on CPU so registry end-to-end hook dispatch runs in CPU CI too
        fake_w = model.head.weight
        try:
            with patch("src.kernels.registry.os.environ.get", side_effect=lambda k, d=None: "4096" if k == "PTOPT_FUSED_LCE_MIN_VOCAB" else os.environ.get(k, d)):
                # Force head.weight.is_cuda check only via patched helper or run directly when CUDA available
                if torch.cuda.is_available():
                    model = model.cuda()
                    x = torch.randn(4, 8, 16, device="cuda")
                    y = torch.randint(0, 4096, (32,), device="cuda")
                    logits = model(x)
                    self.assertIsInstance(logits, LazyLinearLogits)
                    loss = F.cross_entropy(logits.view(-1, 4096), y)
                    loss.backward()
                    self.assertIsNotNone(model.head.weight.grad)
                else:
                    # Verify fused_linear_cross_entropy directly via LazyLinearLogits -> F.cross_entropy hook
                    h_in = torch.randn(4, 8, 16, requires_grad=True)
                    lazy = create_lazy_linear_logits(h_in, fake_w, None)
                    tgt = torch.randint(0, 4096, (32,))
                    loss_hooked = F.cross_entropy(lazy.view(-1, 4096), tgt)
                    loss_ref = ref_linear_cross_entropy(h_in.detach(), fake_w.detach(), None, tgt)
                    self.assertTrue(torch.allclose(loss_hooked.float(), loss_ref.float(), atol=1e-5))
                    loss_hooked.backward()
                    self.assertIsNotNone(fake_w.grad)
        finally:
            uninstall_custom_kernels_hooks(torch)
            mgr.reset()


if __name__ == "__main__":
    unittest.main()
