"""Unit tests for Phase 3.3: Optimizer Fusion and Safe Auto-AMP."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.explain import build_mock_fingerprint
from src.config import ConfigResolver, OptimizeConfig
from src.hooks import HookManager
from src.passes.auto_amp import (
    apply_auto_amp_pass,
    install_auto_amp_hook,
    uninstall_auto_amp_hook,
)
from src.passes.optimizer_fusion import (
    check_fused_eligibility,
    install_optimizer_fusion_hook,
    uninstall_optimizer_fusion_hook,
)


class TestOptimizerFusionPass(unittest.TestCase):
    """Tests for fused/foreach eligibility, generator materialization, and CPU guard."""

    def test_cpu_tensors_never_inject_fused_true(self) -> None:
        model = nn.Linear(8, 4)
        ok, reason = check_fused_eligibility(list(model.parameters()), torch)
        self.assertFalse(ok)
        self.assertIn("CUDA", reason)

        cfg = OptimizeConfig(profile="balanced")
        cfg.optimizer.fused = True
        cfg.optimizer.foreach = True
        mgr = HookManager()
        install_optimizer_fusion_hook(cfg, mgr, torch)
        try:
            # Passing a generator (model.parameters()) must NOT be exhausted before AdamW.__init__!
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            self.assertEqual(len(opt.param_groups[0]["params"]), 2)
            self.assertFalse(opt.param_groups[0].get("fused", False))
            self.assertTrue(opt.param_groups[0].get("foreach", False))
        finally:
            uninstall_optimizer_fusion_hook(torch)
            mgr.reset()

    def test_mock_cuda_injects_fused_for_adamw_and_foreach_for_sgd(self) -> None:
        captured: dict[str, dict[str, object]] = {}

        class _FakeAdamW:
            def __init__(self, params, lr=1e-3, fused=None, foreach=None):  # noqa: ANN001
                captured["AdamW"] = {"params": list(params), "fused": fused, "foreach": foreach}

        class _FakeSGD:
            def __init__(self, params, lr=1e-2, foreach=None):  # noqa: ANN001
                captured["SGD"] = {"params": list(params), "foreach": foreach}

        fake_param = SimpleNamespace(is_cuda=True, is_floating_point=lambda: True, is_complex=lambda: False)
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            optim=SimpleNamespace(AdamW=_FakeAdamW, SGD=_FakeSGD),
        )

        cfg = OptimizeConfig(profile="balanced")
        cfg.optimizer.fused = True
        cfg.optimizer.foreach = True
        mgr = HookManager()
        install_optimizer_fusion_hook(cfg, mgr, fake_torch)
        try:
            _FakeAdamW((p for p in [fake_param]), lr=1e-3)
            _FakeSGD([fake_param], lr=1e-2)
            self.assertTrue(captured["AdamW"]["fused"])
            self.assertIsNone(captured["AdamW"]["foreach"])
            self.assertTrue(captured["SGD"]["foreach"])
        finally:
            uninstall_optimizer_fusion_hook(fake_torch)
            mgr.reset()


class TestAutoAmpPass(unittest.TestCase):
    """Tests for T4 vs L4 config selection, pre-existing AMP yielding, and NaN watchdog."""

    def test_t4_selects_fp16_scaler_and_l4_selects_bf16(self) -> None:
        resolver = ConfigResolver()
        t4_cfg = resolver.resolve(build_mock_fingerprint("t4"), profile="balanced", env={}).config
        self.assertTrue(t4_cfg.amp.enabled)
        self.assertEqual(t4_cfg.amp.dtype, "fp16")
        self.assertTrue(t4_cfg.amp.grad_scaler)

        l4_cfg = resolver.resolve(build_mock_fingerprint("l4"), profile="balanced", env={}).config
        self.assertTrue(l4_cfg.amp.enabled)
        self.assertEqual(l4_cfg.amp.dtype, "bf16")
        self.assertFalse(l4_cfg.amp.grad_scaler)

    def test_cpu_skips_auto_amp(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.amp.enabled = True
        cfg.amp.dtype = "bf16"
        mgr = HookManager()
        msg = apply_auto_amp_pass(cfg, mgr, torch_mod=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
        self.assertTrue(msg.startswith("SKIPPED:"))

    def test_preexisting_user_autocast_and_grad_scaler_yield(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.amp.enabled = True
        cfg.amp.dtype = "fp16"
        cfg.amp.grad_scaler = True
        mgr = HookManager()

        rt = install_auto_amp_hook(cfg, mgr, torch)
        self.assertIsNotNone(rt)
        assert rt is not None
        try:
            # Simulate user creating GradScaler(enabled=True)
            _user_scaler = torch.amp.GradScaler("cpu", enabled=True)
            self.assertTrue(rt.user_amp_detected)
            self.assertTrue(any(ev.status == "SKIPPED" and ev.pass_name == "amp.auto" for ev in mgr.events))
        finally:
            uninstall_auto_amp_hook(torch)
            mgr.reset()

    def test_stability_watchdog_disables_amp_on_nan(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.amp.enabled = True
        cfg.amp.dtype = "bf16"
        mgr = HookManager()
        rt = install_auto_amp_hook(cfg, mgr, torch)
        assert rt is not None
        try:
            ok_fin = rt.check_loss_stability(torch.tensor(1.25))
            self.assertTrue(ok_fin)
            self.assertFalse(rt.disabled_by_watchdog)

            ok_nan = rt.check_loss_stability(torch.tensor(float("nan")))
            self.assertFalse(ok_nan)
            self.assertTrue(rt.disabled_by_watchdog)
            self.assertTrue(any(ev.status == "FALLBACK" and "Non-finite" in ev.reason for ev in mgr.events))
        finally:
            uninstall_auto_amp_hook(torch)
            mgr.reset()


if __name__ == "__main__":
    unittest.main()
