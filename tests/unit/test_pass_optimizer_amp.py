"""Unit tests for Phase 3.3: Optimizer Fusion and Safe Auto-AMP."""

from __future__ import annotations

import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_optimizer_load_state_dict_restores_injected_fused_flag(self) -> None:
        from src.hooks.checkpoint_compat import install_checkpoint_compat_hooks, uninstall_checkpoint_compat_hooks

        p = nn.Parameter(torch.randn(4, 4))
        opt = torch.optim.AdamW([p], lr=1e-3)
        baseline_sd = opt.state_dict()
        self.assertIsNone(baseline_sd["param_groups"][0]["fused"])

        # Simulate a CUDA optimizer whose __init__ received injected fused=True
        opt._ptopt_injected_fused = True  # type: ignore[attr-defined]
        opt.param_groups[0]["fused"] = True
        opt.defaults["foreach"] = None

        install_checkpoint_compat_hooks(torch)
        try:
            saved_sd = opt.state_dict()
            self.assertFalse(saved_sd["param_groups"][0]["fused"])
            opt.load_state_dict(baseline_sd)
            self.assertTrue(opt.param_groups[0]["fused"])
        finally:
            uninstall_checkpoint_compat_hooks(torch)


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

    def test_module_call_actually_enters_autocast_when_cuda_context_present(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.amp.enabled = True
        cfg.amp.dtype = "bf16"
        mgr = HookManager()
        rt = install_auto_amp_hook(cfg, mgr, torch)
        assert rt is not None

        entered: list[tuple[str, object]] = []

        def _fake_autocast(device_type: str = "cuda", dtype: object = None, **_kw: object):
            entered.append((device_type, dtype))
            return nullcontext()

        model = nn.Linear(4, 2)
        try:
            with patch("src.passes.auto_amp._has_cuda_context", return_value=True), patch.object(
                torch.amp, "autocast", side_effect=_fake_autocast
            ):
                out = model(torch.randn(2, 4))
            self.assertEqual(out.shape, (2, 2))
            self.assertEqual(entered, [("cuda", torch.bfloat16)])
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

    def test_module_call_actually_enters_autocast_when_cuda_context_present(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.amp.enabled = True
        cfg.amp.dtype = "bf16"
        mgr = HookManager()
        rt = install_auto_amp_hook(cfg, mgr, torch)
        assert rt is not None

        entered: list[tuple[str, object]] = []

        def _fake_autocast(device_type: str = "cuda", dtype: object = None, **_kw: object):
            entered.append((device_type, dtype))
            return nullcontext()

        model = nn.Linear(4, 2)
        try:
            with patch("src.passes.auto_amp._has_cuda_context", return_value=True), patch.object(
                torch.amp, "autocast", side_effect=_fake_autocast
            ):
                out = model(torch.randn(2, 4))
            self.assertEqual(out.shape, (2, 2))
            self.assertEqual(entered, [("cuda", torch.bfloat16)])
        finally:
            uninstall_auto_amp_hook(torch)
            mgr.reset()

    def test_functional_loss_promotes_half_logits_to_fp32(self) -> None:
        orig_ce_real = F.cross_entropy
        orig_bce_real = F.binary_cross_entropy_with_logits
        seen_ce_dtypes: list[torch.dtype] = []
        seen_bce_dtypes: list[tuple[torch.dtype, torch.dtype]] = []

        class _FakeCudaHalf:
            def __init__(self, tensor: torch.Tensor) -> None:
                self._tensor = tensor
                self.is_cuda = True
                self.dtype = tensor.dtype

            def float(self) -> torch.Tensor:
                return self._tensor.float()

        def _spy_ce(inp: object, target: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            dtype = getattr(inp, "dtype")
            seen_ce_dtypes.append(dtype)
            if isinstance(inp, _FakeCudaHalf):
                return torch.tensor(0.0, dtype=dtype)
            return orig_ce_real(inp, target, *args, **kwargs)  # type: ignore[arg-type]

        def _spy_bce(inp: object, target: object, *args: object, **kwargs: object) -> torch.Tensor:
            in_dt = getattr(inp, "dtype")
            tgt_dt = getattr(target, "dtype")
            seen_bce_dtypes.append((in_dt, tgt_dt))
            if isinstance(inp, _FakeCudaHalf) or isinstance(target, _FakeCudaHalf):
                return torch.tensor(0.0, dtype=in_dt)
            return orig_bce_real(inp, target, *args, **kwargs)  # type: ignore[arg-type]

        cfg = OptimizeConfig(profile="balanced")
        cfg.amp.enabled = True
        cfg.amp.dtype = "fp16"
        cfg.amp.grad_scaler = True
        mgr = HookManager()

        logits_fp16 = _FakeCudaHalf(torch.tensor([[12.0, -12.0]], dtype=torch.float16))
        target_cls = torch.tensor([0], dtype=torch.long)
        bce_logits_fp16 = _FakeCudaHalf(torch.tensor([[2.0, -2.0]], dtype=torch.float16))
        bce_target_fp16 = _FakeCudaHalf(torch.tensor([[1.0, 0.0]], dtype=torch.float16))

        with patch.object(F, "cross_entropy", new=_spy_ce), patch.object(
            F, "binary_cross_entropy_with_logits", new=_spy_bce
        ):
            rt = install_auto_amp_hook(cfg, mgr, torch)
            assert rt is not None
            try:
                loss_ce = F.cross_entropy(logits_fp16, target_cls)  # type: ignore[arg-type]
                self.assertEqual(loss_ce.dtype, torch.float32)
                self.assertTrue(torch.isfinite(loss_ce))

                loss_bce = F.binary_cross_entropy_with_logits(
                    bce_logits_fp16, bce_target_fp16  # type: ignore[arg-type]
                )
                self.assertEqual(loss_bce.dtype, torch.float32)
                self.assertTrue(torch.isfinite(loss_bce))

                # Inside an existing autocast region, wrapper must leave dtype untouched
                with patch("src.passes.auto_amp._is_autocast_active", return_value=True):
                    F.cross_entropy(logits_fp16, target_cls)  # type: ignore[arg-type]

                self.assertEqual(seen_ce_dtypes, [torch.float32, torch.float16])
                self.assertEqual(seen_bce_dtypes, [(torch.float32, torch.float32)])
            finally:
                uninstall_auto_amp_hook(torch)
                mgr.reset()

        self.assertIs(F.cross_entropy, orig_ce_real)
        self.assertIs(F.binary_cross_entropy_with_logits, orig_bce_real)

    def test_eager_amp_skips_cast_heavy_layernorm_mlp_without_conv_or_attn(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.amp.enabled = True
        cfg.amp.dtype = "bf16"
        cfg.compile.enabled = False
        mgr = HookManager()
        rt = install_auto_amp_hook(cfg, mgr, torch)
        assert rt is not None

        entered: list[tuple[str, object]] = []

        def _fake_autocast(device_type: str = "cuda", dtype: object = None, **_kw: object):
            entered.append((device_type, dtype))
            return nullcontext()

        deep_ln_mlp = nn.Sequential(*[m for _ in range(6) for m in (nn.Linear(8, 8), nn.LayerNorm(8))])
        conv_with_ln = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), *[nn.Identity() for _ in range(6)])
        try:
            with patch("src.passes.auto_amp._has_cuda_context", return_value=True), patch.object(
                torch.amp, "autocast", side_effect=_fake_autocast
            ):
                out_mlp = deep_ln_mlp(torch.randn(2, 8))
                out_cnn = conv_with_ln(torch.randn(1, 3, 8, 8))

            self.assertEqual(out_mlp.shape, (2, 8))
            self.assertEqual(out_cnn.shape, (1, 4, 8, 8))
            # Only the Conv2d module entered autocast; the 6-LayerNorm MLP skipped eager AMP
            self.assertEqual(len(entered), 1)
            self.assertTrue(any(ev.status == "SKIPPED" and "deep LayerNorm MLP" in ev.reason for ev in mgr.events))
        finally:
            uninstall_auto_amp_hook(torch)
            mgr.reset()


if __name__ == "__main__":
    unittest.main()
