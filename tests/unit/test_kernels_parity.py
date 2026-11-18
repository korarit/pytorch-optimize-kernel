"""Unit tests for Phase 3.7: Custom Triton kernels parity (fwd+bwd), edge shapes, and capability gates."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import OptimizeConfig
from src.hooks import HookManager
from src.kernels import (
    HAS_TRITON,
    KernelSpec,
    apply_custom_kernels_pass,
    check_kernel_capability,
    install_custom_kernels_hooks,
    ref_cross_entropy,
    ref_layernorm,
    ref_rmsnorm,
    ref_swiglu,
    triton_cross_entropy,
    triton_layernorm,
    triton_rmsnorm,
    triton_swiglu,
    uninstall_custom_kernels_hooks,
)


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-4, 1e-4
    if dtype == torch.float16:
        return 5e-3, 5e-3
    return 2e-2, 2e-2  # bfloat16


class TestKernelCapabilityGate(unittest.TestCase):
    """Tests for hardware, Triton, dtype, and allow_list capability gating."""

    def test_cpu_t4_and_missing_triton_fall_back_cleanly(self) -> None:
        spec = KernelSpec(name="rmsnorm", min_sm=80, supported_dtypes=("fp32", "fp16", "bf16"))

        ok_triton, r_triton = check_kernel_capability(spec, torch, triton_available=False, sm_override=89)
        self.assertFalse(ok_triton)
        self.assertIn("Triton unavailable", r_triton)

        ok_cpu, r_cpu = check_kernel_capability(spec, torch, triton_available=True, sm_override=0)
        self.assertFalse(ok_cpu)
        self.assertIn("CUDA unavailable", r_cpu)

        ok_t4, r_t4 = check_kernel_capability(spec, torch, triton_available=True, sm_override=75)
        self.assertFalse(ok_t4)
        self.assertIn("sm_75 < min_sm 80", r_t4)

        ok_fp64, r_fp64 = check_kernel_capability(spec, torch, triton_available=True, sm_override=89, dtype_name="fp64")
        self.assertFalse(ok_fp64)
        self.assertIn("dtype fp64", r_fp64)

        ok_l4, _ = check_kernel_capability(spec, torch, triton_available=True, sm_override=89, dtype_name="bf16")
        self.assertTrue(ok_l4)

    def test_install_fallback_registry_logs_skips_when_unsupported(self) -> None:
        cfg = OptimizeConfig(profile="aggressive")
        cfg.kernels.enabled = True
        mgr = HookManager()

        with patch.dict(os.environ, {}, clear=False):
            reg = install_custom_kernels_hooks(cfg, mgr, torch, triton_available=True, sm_override=75)
            assert reg is not None
            try:
                self.assertEqual(len(reg.fallback_registry.entries), 6)
                self.assertIn("rope", reg.fallback_registry.entries)
                self.assertTrue(all(e.status == "SKIPPED" for e in reg.fallback_registry.entries.values()))
                self.assertTrue(all(ev.status == "SKIPPED" for ev in mgr.events))
            finally:
                uninstall_custom_kernels_hooks(torch)
                mgr.reset()

        cfg_off = OptimizeConfig(profile="balanced")
        cfg_off.kernels.enabled = False
        self.assertTrue(apply_custom_kernels_pass(cfg_off, mgr, torch).startswith("SKIPPED:"))

    def test_compile_aware_kernel_gate_defers_module_wrappers_to_inductor(self) -> None:
        cfg = OptimizeConfig(profile="aggressive")
        cfg.kernels.enabled = True
        cfg.compile.enabled = True
        mgr = HookManager()

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PTOPT_FORCE_ALL_KERNELS", None)
            reg = install_custom_kernels_hooks(cfg, mgr, torch, triton_available=True, sm_override=89)
            assert reg is not None
            try:
                entries = reg.fallback_registry.entries
                self.assertEqual(entries["cross_entropy"].status, "APPLIED")
                self.assertEqual(entries["rmsnorm"].status, "SKIPPED")
                self.assertEqual(entries["layernorm"].status, "SKIPPED")
                self.assertEqual(entries["swiglu"].status, "SKIPPED")
                self.assertEqual(entries["fused_linear_cross_entropy"].status, "APPLIED")
                self.assertIn("hybrid", entries["fused_linear_cross_entropy"].reason.lower())
                self.assertIn("Inductor", entries["rmsnorm"].reason)
                self.assertIn("Inductor", entries["layernorm"].reason)
            finally:
                uninstall_custom_kernels_hooks(torch)
                mgr.reset()

        with patch.dict(os.environ, {"PTOPT_HYBRID_LCE": "0"}, clear=False):
            reg2 = install_custom_kernels_hooks(cfg, mgr, torch, triton_available=True, sm_override=89)
            assert reg2 is not None
            try:
                entries2 = reg2.fallback_registry.entries
                self.assertEqual(entries2["fused_linear_cross_entropy"].status, "SKIPPED")
            finally:
                uninstall_custom_kernels_hooks(torch)
                mgr.reset()

        with patch.dict(os.environ, {"PTOPT_FORCE_ALL_KERNELS": "1"}):
            reg_all = install_custom_kernels_hooks(cfg, mgr, torch, triton_available=True, sm_override=89)
            assert reg_all is not None
            try:
                self.assertTrue(all(e.status == "APPLIED" for e in reg_all.fallback_registry.entries.values()))
            finally:
                uninstall_custom_kernels_hooks(torch)
                mgr.reset()


class TestKernelsForwardBackwardParity(unittest.TestCase):
    """Forward + Backward numerical parity across edge shapes (non-pow2, b=1, seq=1) & dtypes."""

    def setUp(self) -> None:
        self.device = torch.device("cuda" if (HAS_TRITON and torch.cuda.is_available()) else "cpu")
        self.dtypes = (torch.float32, torch.float16, torch.bfloat16) if self.device.type == "cuda" else (torch.float32,)
        self.edge_shapes = ((1, 1, 384), (2, 5, 768), (3, 7, 1025))

    def test_rmsnorm_forward_backward_parity(self) -> None:
        torch.manual_seed(7)
        for dtype in self.dtypes:
            atol, rtol = _tolerances(dtype)
            for shape in self.edge_shapes:
                dim = shape[-1]
                x_ref = torch.randn(shape, device=self.device, dtype=torch.float64, requires_grad=True)
                w_ref = torch.randn(dim, device=self.device, dtype=torch.float64, requires_grad=True)
                dy_ref = torch.randn(shape, device=self.device, dtype=torch.float64)

                y_ref = ref_rmsnorm(x_ref, w_ref, eps=1e-6)
                y_ref.backward(dy_ref)

                x = x_ref.detach().to(dtype).requires_grad_(True)
                w = w_ref.detach().to(dtype).requires_grad_(True)
                dy = dy_ref.detach().to(dtype)

                y = triton_rmsnorm(x, w, eps=1e-6)
                y.backward(dy)

                torch.testing.assert_close(y.float(), y_ref.float(), atol=atol, rtol=rtol)
                torch.testing.assert_close(x.grad.float(), x_ref.grad.float(), atol=atol * 2, rtol=rtol * 2)
                torch.testing.assert_close(w.grad.float(), w_ref.grad.float(), atol=atol * 4, rtol=rtol * 4)

    def test_layernorm_forward_backward_parity(self) -> None:
        torch.manual_seed(11)
        for dtype in self.dtypes:
            atol, rtol = _tolerances(dtype)
            for shape in self.edge_shapes:
                dim = shape[-1]
                x_ref = torch.randn(shape, device=self.device, dtype=torch.float64, requires_grad=True)
                w_ref = torch.randn(dim, device=self.device, dtype=torch.float64, requires_grad=True)
                b_ref = torch.randn(dim, device=self.device, dtype=torch.float64, requires_grad=True)
                dy_ref = torch.randn(shape, device=self.device, dtype=torch.float64)

                y_ref = torch.nn.functional.layer_norm(x_ref, (dim,), weight=w_ref, bias=b_ref, eps=1e-5)
                y_ref.backward(dy_ref)

                x = x_ref.detach().to(dtype).requires_grad_(True)
                w = w_ref.detach().to(dtype).requires_grad_(True)
                b = b_ref.detach().to(dtype).requires_grad_(True)
                dy = dy_ref.detach().to(dtype)

                y = triton_layernorm(x, w, b, eps=1e-5)
                y.backward(dy)

                torch.testing.assert_close(y.float(), y_ref.float(), atol=atol, rtol=rtol)
                torch.testing.assert_close(x.grad.float(), x_ref.grad.float(), atol=atol * 2, rtol=rtol * 2)
                torch.testing.assert_close(w.grad.float(), w_ref.grad.float(), atol=atol * 4, rtol=rtol * 4)
                torch.testing.assert_close(b.grad.float(), b_ref.grad.float(), atol=atol * 4, rtol=rtol * 4)

    def test_cross_entropy_forward_backward_parity(self) -> None:
        torch.manual_seed(19)
        for dtype in self.dtypes:
            atol, rtol = _tolerances(dtype)
            for rows, vocab in ((1, 384), (16, 1025), (32, 2048)):
                logits_ref = torch.randn(rows, vocab, device=self.device, dtype=torch.float64, requires_grad=True)
                targets = torch.randint(0, vocab, (rows,), device=self.device)
                if rows > 1:
                    targets[0] = -100  # test ignore_index

                loss_ref = torch.nn.functional.cross_entropy(logits_ref, targets, ignore_index=-100)
                loss_ref.backward()

                logits = logits_ref.detach().to(dtype).requires_grad_(True)
                loss = triton_cross_entropy(logits, targets, ignore_index=-100, reduction="mean")
                loss.backward()

                torch.testing.assert_close(loss.float(), loss_ref.float(), atol=atol, rtol=rtol)
                torch.testing.assert_close(logits.grad.float(), logits_ref.grad.float(), atol=atol * 2, rtol=rtol * 2)

    def test_cross_entropy_all_ignored_targets_safe_without_item_sync(self) -> None:
        logits = torch.randn(4, 32, device=self.device, dtype=torch.float32, requires_grad=True)
        targets = torch.full((4,), -100, device=self.device, dtype=torch.long)
        loss = triton_cross_entropy(logits, targets, ignore_index=-100, reduction="mean")
        loss.backward()
        assert logits.grad is not None
        if self.device.type == "cuda":
            self.assertEqual(float(loss.item()), 0.0)
            self.assertTrue(bool((logits.grad == 0.0).all().item()))
        else:
            # CPU path delegates to F.cross_entropy reference (which produces NaN when 0 targets are valid)
            ref = ref_cross_entropy(logits.detach(), targets, ignore_index=-100, reduction="mean")
            torch.testing.assert_close(loss.float(), ref.float(), equal_nan=True)

    def test_swiglu_forward_backward_parity(self) -> None:
        torch.manual_seed(29)
        for dtype in self.dtypes:
            atol, rtol = _tolerances(dtype)
            for shape in self.edge_shapes:
                g_ref = torch.randn(shape, device=self.device, dtype=torch.float64, requires_grad=True)
                u_ref = torch.randn(shape, device=self.device, dtype=torch.float64, requires_grad=True)
                dy_ref = torch.randn(shape, device=self.device, dtype=torch.float64)

                y_ref = ref_swiglu(g_ref, u_ref)
                y_ref.backward(dy_ref)

                g = g_ref.detach().to(dtype).requires_grad_(True)
                u = u_ref.detach().to(dtype).requires_grad_(True)
                dy = dy_ref.detach().to(dtype)

                y = triton_swiglu(g, u)
                y.backward(dy)

                torch.testing.assert_close(y.float(), y_ref.float(), atol=atol, rtol=rtol)
                torch.testing.assert_close(g.grad.float(), g_ref.grad.float(), atol=atol * 2, rtol=rtol * 2)
                torch.testing.assert_close(u.grad.float(), u_ref.grad.float(), atol=atol * 2, rtol=rtol * 2)

    def test_rope_forward_backward_parity_and_strided_views(self) -> None:
        from src.kernels.rope_or_swiglu import _to_row_strided_2d, ref_rope, triton_rope

        torch.manual_seed(37)
        for dtype in self.dtypes:
            atol, rtol = _tolerances(dtype)
            # Test standard MHA + GQA shapes and both D//2 and D cos/sin tables
            for bsz, h_q, h_k, seq, d_head, tbl_half in (
                (2, 4, 4, 16, 64, False),
                (2, 8, 2, 17, 64, True),
            ):
                q_ref = torch.randn(bsz, h_q, seq, d_head, device=self.device, dtype=torch.float64, requires_grad=True)
                k_ref = torch.randn(bsz, h_k, seq, d_head, device=self.device, dtype=torch.float64, requires_grad=True)
                tbl_d = d_head // 2 if tbl_half else d_head
                angles = torch.randn(1, 1, seq, d_head // 2, device=self.device, dtype=torch.float64)
                cos_base = angles.cos() if tbl_half else torch.cat([angles.cos(), angles.cos()], dim=-1)
                sin_base = angles.sin() if tbl_half else torch.cat([angles.sin(), angles.sin()], dim=-1)

                qr_ref, kr_ref = ref_rope(q_ref, k_ref, cos_base, sin_base)
                dq = torch.randn_like(qr_ref)
                dk = torch.randn_like(kr_ref)
                torch.autograd.backward([qr_ref, kr_ref], [dq, dk])

                q = q_ref.detach().to(dtype).requires_grad_(True)
                k = k_ref.detach().to(dtype).requires_grad_(True)
                cos = cos_base.detach().to(dtype)
                sin = sin_base.detach().to(dtype)
                qr, kr = triton_rope(q, k, cos, sin)
                torch.autograd.backward([qr, kr], [dq.to(dtype), dk.to(dtype)])

                torch.testing.assert_close(qr.float(), qr_ref.float(), atol=atol, rtol=rtol)
                torch.testing.assert_close(kr.float(), kr_ref.float(), atol=atol, rtol=rtol)
                assert q.grad is not None and k.grad is not None
                torch.testing.assert_close(q.grad.float(), q_ref.grad.float(), atol=atol * 2, rtol=rtol * 2)
                torch.testing.assert_close(k.grad.float(), k_ref.grad.float(), atol=atol * 2, rtol=rtol * 2)

        # Verify zero-copy row-strided 2D helper on gate_up.chunk(2, dim=-1) and sliced logits
        buf = torch.randn(12, 96, device=self.device)
        g_slice, u_slice = buf.chunk(2, dim=-1)
        g_view = _to_row_strided_2d(g_slice)
        u_view = _to_row_strided_2d(u_slice)
        self.assertEqual(g_view.data_ptr(), g_slice.data_ptr())
        self.assertEqual(u_view.data_ptr(), u_slice.data_ptr())
        self.assertEqual(g_view.stride(), (96, 1))


if __name__ == "__main__":
    unittest.main()
