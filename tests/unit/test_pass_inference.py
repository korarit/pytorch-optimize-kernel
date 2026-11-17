"""Unit tests for Phase 3.6: Safe Inference Optimization (`inference_mode`, grads, FP8/INT8 gates, CUDA Graphs)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import OptimizeConfig
from src.hooks import HookManager
from src.passes.inference_opt import (
    apply_inference_opt_pass,
    check_fp8_eligibility,
    install_inference_opt_hook,
    uninstall_inference_opt_hook,
)


class TestInferenceOptPass(unittest.TestCase):
    """Tests for train/eval switching, requires_grad guards, FP8 arch gate, and CUDA Graph fallback."""

    def test_train_eval_switching_preserves_backward_and_uses_inference_mode(self) -> None:
        cfg = OptimizeConfig(profile="safe")
        cfg.inference.inference_mode = True
        mgr = HookManager()
        rt = install_inference_opt_hook(cfg, mgr, torch)
        assert rt is not None

        seen_infer_flags: list[bool] = []

        class _SwitchingNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(8, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                seen_infer_flags.append(torch.is_inference_mode_enabled())
                return self.lin(x)

        model = _SwitchingNet()
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        try:
            # Train step 1
            model.train()
            opt.zero_grad()
            loss1 = model(torch.randn(4, 8)).sum()
            loss1.backward()
            self.assertIsNotNone(model.lin.weight.grad)
            opt.step()

            # Eval step enters inference_mode automatically
            model.eval()
            with torch.no_grad():
                eval_out = model(torch.randn(2, 8))
            self.assertEqual(eval_out.shape, (2, 4))

            # Train step 2 back from eval still computes gradients cleanly
            model.train()
            opt.zero_grad()
            loss2 = model(torch.randn(4, 8)).sum()
            loss2.backward()
            self.assertIsNotNone(model.lin.weight.grad)
            opt.step()

            self.assertEqual(seen_infer_flags, [False, True, False])
            self.assertEqual(rt.inference_calls, 1)
        finally:
            uninstall_inference_opt_hook(torch)
            mgr.reset()

    def test_requires_grad_and_enable_grad_guards_skip_inference_mode(self) -> None:
        cfg = OptimizeConfig(profile="safe")
        cfg.inference.inference_mode = True
        mgr = HookManager()
        rt = install_inference_opt_hook(cfg, mgr, torch)
        assert rt is not None

        model = nn.Sequential(nn.Linear(6, 6), nn.Tanh(), nn.Linear(6, 2))
        model.eval()
        try:
            # 1) Input requires_grad=True during eval() (e.g. adversarial / PINN)
            x_adv = torch.randn(3, 6, requires_grad=True)
            out_adv = model(x_adv)
            self.assertTrue(out_adv.requires_grad)
            out_adv.sum().backward()
            self.assertIsNotNone(x_adv.grad)

            # 2) Explicit torch.enable_grad() during eval()
            x_plain = torch.randn(3, 6)
            with torch.enable_grad():
                out_en = model(x_plain)
                self.assertTrue(out_en.requires_grad)

            self.assertGreaterEqual(rt.grad_guard_skips, 2)
            self.assertEqual(rt.inference_calls, 0)
            self.assertTrue(any(ev.status == "SKIPPED" and "preserving autograd" in ev.reason for ev in mgr.events))
        finally:
            uninstall_inference_opt_hook(torch)
            mgr.reset()

    def test_fp8_hardware_gate_t4_skips_and_l4_allows(self) -> None:
        def _mock_cuda_torch(major: int, minor: int) -> SimpleNamespace:
            return SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    current_device=lambda: 0,
                    get_device_properties=lambda _i: SimpleNamespace(major=major, minor=minor),
                ),
                float8_e4m3fn=object(),
                _scaled_mm=lambda *a, **kw: None,
                nn=torch.nn,
                enable_grad=torch.enable_grad,
                inference_mode=torch.inference_mode,
                Tensor=torch.Tensor,
            )

        t4_torch = _mock_cuda_torch(7, 5)
        ok_t4, reason_t4 = check_fp8_eligibility(t4_torch)
        self.assertFalse(ok_t4)
        self.assertIn("sm_75 < 89", reason_t4)

        l4_torch = _mock_cuda_torch(8, 9)
        ok_l4, reason_l4 = check_fp8_eligibility(l4_torch)
        self.assertTrue(ok_l4)
        self.assertIn("sm_89", reason_l4)

        cfg = OptimizeConfig(profile="aggressive")
        cfg.inference.inference_mode = True
        mgr_t4 = HookManager()
        with patch.dict(os.environ, {"PTOPT_INFERENCE_FP8": "1"}):
            install_inference_opt_hook(cfg, mgr_t4, t4_torch)
            try:
                self.assertTrue(any(ev.pass_name == "inference.fp8" and ev.status == "SKIPPED" for ev in mgr_t4.events))
            finally:
                uninstall_inference_opt_hook(t4_torch)
                mgr_t4.reset()

        mgr_l4 = HookManager()
        with patch.dict(os.environ, {"PTOPT_INFERENCE_FP8": "1"}):
            install_inference_opt_hook(cfg, mgr_l4, l4_torch)
            try:
                self.assertTrue(any(ev.pass_name == "inference.fp8" and ev.status == "APPLIED" for ev in mgr_l4.events))
            finally:
                uninstall_inference_opt_hook(l4_torch)
                mgr_l4.reset()

    def test_cuda_graphs_reduce_overhead_and_dynamic_shape_fallback(self) -> None:
        cfg = OptimizeConfig(profile="aggressive")
        cfg.inference.inference_mode = True
        cfg.inference.cuda_graphs = True
        mgr = HookManager()
        compile_modes: list[object] = []

        def _spy_compile(fn: object, **kwargs: object) -> object:
            compile_modes.append(kwargs.get("mode"))
            return fn

        model = nn.Linear(8, 4)
        model.eval()
        with patch("src.passes.inference_opt._has_cuda_context", return_value=True), patch.object(
            torch, "compile", side_effect=_spy_compile
        ):
            rt = install_inference_opt_hook(cfg, mgr, torch)
            assert rt is not None
            try:
                model(torch.randn(1, 8))
                model(torch.randn(1, 8))
                self.assertEqual(compile_modes, ["reduce-overhead"])
                self.assertEqual(rt.cuda_graph_compiles, 1)

                # Changing input batch size triggers CUDA Graph dynamic-shape fallback cleanly
                out_dyn = model(torch.randn(3, 8))
                self.assertEqual(out_dyn.shape, (3, 4))
                self.assertIsNone(model._compiled_call_impl)
                self.assertEqual(rt.cuda_graph_fallbacks, 1)
                self.assertTrue(any(ev.pass_name == "inference.cuda_graphs" and ev.status == "FALLBACK" for ev in mgr.events))
            finally:
                uninstall_inference_opt_hook(torch)
                mgr.reset()

    def test_cpu_dynamic_quantization_opt_in(self) -> None:
        cfg = OptimizeConfig(profile="safe")
        cfg.inference.inference_mode = True
        mgr = HookManager()
        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
        model.eval()

        with patch.dict(os.environ, {"PTOPT_INFERENCE_QUANT": "int8"}):
            msg = apply_inference_opt_pass(cfg, mgr, torch_mod=torch)
            self.assertIn("Inference opt hook installed", msg)
            try:
                out = model(torch.randn(2, 16))
                self.assertEqual(out.shape, (2, 4))
                self.assertTrue(any(ev.pass_name == "inference.quant" and ev.status == "APPLIED" for ev in mgr.events))

                # TransformerEncoderLayer fastpath fails under dynamic Linear quantization -> falls back cleanly
                enc = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32, batch_first=True)
                enc.eval()
                enc_out = enc(torch.randn(2, 4, 16))
                self.assertEqual(enc_out.shape, (2, 4, 16))
                self.assertIsNone(getattr(enc, "_ptopt_quantized_mod", None))
                self.assertTrue(any(ev.pass_name == "inference.quant" and ev.status == "FALLBACK" for ev in mgr.events))
            finally:
                uninstall_inference_opt_hook(torch)
                mgr.reset()


if __name__ == "__main__":
    unittest.main()
