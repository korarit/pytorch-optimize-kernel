"""Unit tests for Phase 3.5: Auto `torch.compile`, cache dir, error fallback, and `_orig_mod.` normalization."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptopt.config import OptimizeConfig
from ptopt.hooks import HookManager
from ptopt.hooks.checkpoint_compat import (
    add_orig_mod_prefix,
    install_checkpoint_compat_hooks,
    strip_orig_mod_prefix,
    uninstall_checkpoint_compat_hooks,
)
from ptopt.passes.auto_compile import (
    apply_auto_compile_pass,
    install_auto_compile_hook,
    uninstall_auto_compile_hook,
)
from ptopt.passes.memory_allocator import apply_prelaunch_env


class TestAutoCompilePass(unittest.TestCase):
    """Tests for compile guards, runtime fallback, dynamic shape adaptation, and checkpoint compat."""

    def test_compile_error_fallback_keeps_script_running(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.compile.enabled = True
        cfg.compile.mode = "default"
        mgr = HookManager()

        def _failing_compile(*_args: object, **_kwargs: object):
            raise RuntimeError("simulated inductor backend failure")

        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        with patch.dict(os.environ, {"PTOPT_COMPILE_CPU": "1"}), patch.object(
            torch, "compile", side_effect=_failing_compile
        ):
            rt = install_auto_compile_hook(cfg, mgr, torch)
            assert rt is not None
            try:
                out = model(torch.randn(2, 8))
                self.assertEqual(out.shape, (2, 4))
                self.assertIsNone(model._compiled_call_impl)
                self.assertEqual(rt.fallback_count, 1)
                self.assertTrue(
                    any(ev.status == "FALLBACK" and "setup failed" in ev.reason for ev in mgr.events)
                )
            finally:
                uninstall_auto_compile_hook(torch)
                mgr.reset()

    def test_compiled_runtime_execution_error_reverts_to_eager(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.compile.enabled = True
        mgr = HookManager()

        def _broken_compiled(*_args: object, **_kwargs: object):
            raise RuntimeError("simulated runtime graph break explosion")

        model = nn.Linear(6, 3)
        with patch.dict(os.environ, {"PTOPT_COMPILE_CPU": "1"}), patch.object(
            torch, "compile", return_value=_broken_compiled
        ):
            rt = install_auto_compile_hook(cfg, mgr, torch)
            assert rt is not None
            try:
                out = model(torch.randn(2, 6))
                self.assertEqual(out.shape, (2, 3))
                self.assertIsNone(model._compiled_call_impl)
                self.assertEqual(rt.fallback_count, 1)
                self.assertTrue(
                    any(ev.status == "FALLBACK" and "falling back to eager" in ev.reason for ev in mgr.events)
                )
            finally:
                uninstall_auto_compile_hook(torch)
                mgr.reset()

    def test_compile_idempotency_and_precompiled_yield(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.compile.enabled = True
        mgr = HookManager()
        compile_calls: list[dict[str, object]] = []

        def _spy_compile(fn: object, **kwargs: object) -> object:
            compile_calls.append(dict(kwargs))
            return fn

        model = nn.Linear(4, 2)
        precompiled = nn.Linear(4, 2)
        precompiled._orig_mod = precompiled  # type: ignore[attr-defined]

        with patch.dict(os.environ, {"PTOPT_COMPILE_CPU": "1"}), patch.object(
            torch, "compile", side_effect=_spy_compile
        ):
            rt = install_auto_compile_hook(cfg, mgr, torch)
            assert rt is not None
            try:
                x = torch.randn(2, 4)
                model(x)
                model(x)
                model(x)
                precompiled(x)
                self.assertEqual(len(compile_calls), 1)
                self.assertTrue(
                    any(ev.status == "SKIPPED" and "Pre-existing torch.compile" in ev.reason for ev in mgr.events)
                )
            finally:
                uninstall_auto_compile_hook(torch)
                mgr.reset()

    def test_tiny_dynamic_shape_reverts_to_eager_and_large_switches_to_dynamic_true(self) -> None:
        # 1. Tiny module (`nn.Linear(4, 2)` = 10 params): static compile on step 0, revert to eager on shape change
        cfg_tiny = OptimizeConfig(profile="balanced")
        cfg_tiny.compile.enabled = True
        cfg_tiny.compile.dynamic = False
        mgr_tiny = HookManager()
        tiny_flags: list[object] = []

        def _spy_tiny(fn: object, **kwargs: object) -> object:
            tiny_flags.append(kwargs.get("dynamic"))
            return lambda *a, **kw: fn(*a, **kw)

        tiny_model = nn.Linear(4, 2)
        with patch.dict(os.environ, {"PTOPT_COMPILE_CPU": "1"}), patch.object(
            torch, "compile", side_effect=_spy_tiny
        ):
            rt = install_auto_compile_hook(cfg_tiny, mgr_tiny, torch)
            assert rt is not None
            try:
                tiny_model(torch.randn(2, 4))
                tiny_model(torch.randn(3, 4))
                tiny_model(torch.randn(5, 4))
                self.assertEqual(tiny_flags, [False])
                self.assertIsNone(tiny_model._compiled_call_impl)
                self.assertEqual(rt.fallback_count, 1)
                self.assertTrue(
                    any(ev.status == "FALLBACK" and "reverted to eager" in ev.reason for ev in mgr_tiny.events)
                )
            finally:
                uninstall_auto_compile_hook(torch)
                mgr_tiny.reset()

        # 2. Larger module (>= 250k params): static compile on step 0, adapt once to dynamic=True on shape change
        cfg = OptimizeConfig(profile="balanced")
        cfg.compile.enabled = True
        cfg.compile.dynamic = False
        mgr = HookManager()
        dynamic_flags: list[object] = []

        def _spy_compile(fn: object, **kwargs: object) -> object:
            dynamic_flags.append(kwargs.get("dynamic"))
            return fn

        large_model = nn.Linear(512, 512)  # 262,656 params > 250,000 threshold
        with patch.dict(os.environ, {"PTOPT_COMPILE_CPU": "1"}), patch.object(
            torch, "compile", side_effect=_spy_compile
        ):
            rt = install_auto_compile_hook(cfg, mgr, torch)
            assert rt is not None
            try:
                large_model(torch.randn(2, 512))
                large_model(torch.randn(3, 512))
                large_model(torch.randn(5, 512))
                large_model(torch.randn(7, 512))
                # Step 0 compiles static (False); step 1 adapts once to dynamic=True; steps 2-3 reuse
                self.assertEqual(dynamic_flags, [False, True])
            finally:
                uninstall_auto_compile_hook(torch)
                mgr.reset()

    def test_cache_dir_env_and_orig_mod_checkpoint_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "inductor_cache"
            cfg = OptimizeConfig(profile="balanced")
            cfg.compile.enabled = True
            cfg.compile.cache_dir = str(cache_dir)

            env: dict[str, str] = {}
            notes = apply_prelaunch_env(cfg, env)
            self.assertEqual(env["TORCHINDUCTOR_CACHE_DIR"], str(cache_dir.resolve()))
            self.assertEqual(env["TORCHINDUCTOR_FX_GRAPH_CACHE"], "1")
            self.assertTrue(any("TORCHINDUCTOR_CACHE_DIR=" in n for n in notes))

            # Verify apply_auto_compile_pass creates cache_dir cleanly when disabled vs enabled
            mgr = HookManager()
            cfg_off = OptimizeConfig(profile="safe")
            self.assertTrue(apply_auto_compile_pass(cfg_off, mgr, torch).startswith("SKIPPED:"))

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
                msg = apply_auto_compile_pass(cfg, mgr, torch)
            try:
                self.assertIn("Auto-Compile hook installed", msg)
                self.assertTrue(cache_dir.is_dir())
            finally:
                uninstall_auto_compile_hook(torch)
                mgr.reset()

        # _orig_mod. checkpoint prefix round-trip
        base_model = nn.Linear(4, 2)
        prefixed = add_orig_mod_prefix(base_model.state_dict())
        self.assertTrue(all(k.startswith("_orig_mod.") for k in prefixed.keys()))
        stripped = strip_orig_mod_prefix(prefixed)
        self.assertEqual(list(stripped.keys()), list(base_model.state_dict().keys()))

        install_checkpoint_compat_hooks(torch)
        try:
            base_model.load_state_dict(prefixed)
        finally:
            uninstall_checkpoint_compat_hooks(torch)

    def test_step0_static_tiny_module_bypass_skips_compile_entirely(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.compile.enabled = True
        mgr = HookManager()
        compile_calls = 0

        def _spy_compile(fn: object, **_kwargs: object) -> object:
            nonlocal compile_calls
            compile_calls += 1
            return fn

        tiny_model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 8))  # 9,352 params
        with patch.dict(
            os.environ,
            {"PTOPT_COMPILE_CPU": "1", "PTOPT_COMPILE_STATIC_MIN_PARAMS": "50000"},
        ), patch.object(torch, "compile", side_effect=_spy_compile):
            rt = install_auto_compile_hook(cfg, mgr, torch)
            assert rt is not None
            try:
                out = tiny_model(torch.randn(8, 64))
                self.assertEqual(out.shape, (8, 8))
                self.assertEqual(compile_calls, 0)
                self.assertTrue(getattr(tiny_model, "_ptopt_compile_ineligible", False))
                self.assertTrue(
                    any(ev.status == "SKIPPED" and "Tiny module" in ev.reason for ev in mgr.events)
                )
            finally:
                uninstall_auto_compile_hook(torch)
                mgr.reset()

    def test_train_cuda_graphs_2step_promotion_mark_step_and_dynamic_fallback(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.compile.enabled = True
        cfg.compile.mode = "default"
        cfg.compile.dynamic = False
        cfg.compile.train_cuda_graphs = True
        mgr = HookManager()

        compile_modes: list[tuple[object, object]] = []
        mark_calls = 0

        def _spy_compile(fn: object, **kwargs: object) -> object:
            compile_modes.append((kwargs.get("mode"), kwargs.get("dynamic")))
            return fn

        def _spy_mark() -> None:
            nonlocal mark_calls
            mark_calls += 1

        model = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 16))  # >250k params
        with patch.dict(
            os.environ,
            {"PTOPT_COMPILE_CPU": "1", "PTOPT_CUDA_SM_OVERRIDE": "89", "PTOPT_TRAIN_CG_FORCE": "1"},
        ), patch.object(torch, "compile", side_effect=_spy_compile), patch.object(
            torch.compiler, "cudagraph_mark_step_begin", side_effect=_spy_mark, create=True
        ):
            rt = install_auto_compile_hook(cfg, mgr, torch)
            assert rt is not None
            try:
                x_static = torch.randn(32, 512)
                # Step 0: eager shape probe (0 compiles); Step 1: 2nd static step compiles ONCE -> reduce-overhead + mark_step_begin
                # Step 2: 3rd static step keeps reduce-overhead + calls mark_step_begin
                model(x_static)
                self.assertEqual(compile_modes, [])
                self.assertTrue(getattr(model, "_ptopt_train_cg_pending", False))
                model(x_static)
                model(x_static)
                self.assertEqual(compile_modes, [("reduce-overhead", False)])
                self.assertTrue(getattr(model, "_ptopt_train_cg_active", False))
                self.assertEqual(mark_calls, 2)
                self.assertTrue(
                    any(ev.pass_name == "compile.train_cuda_graphs" and ev.status == "APPLIED" for ev in mgr.events)
                )

                # Dynamic shape change on Step 3 demotes CUDA Graphs back to default/dynamic=True
                model(torch.randn(16, 512))
                self.assertFalse(getattr(model, "_ptopt_train_cg_active", False))
                self.assertTrue(getattr(model, "_ptopt_train_cg_failed", False))
                self.assertEqual(compile_modes[-1], ("default", True))
                self.assertTrue(
                    any(ev.pass_name == "compile.train_cuda_graphs" and ev.status == "FALLBACK" for ev in mgr.events)
                )
            finally:
                uninstall_auto_compile_hook(torch)
                mgr.reset()

    def test_train_cuda_graphs_skips_grad_scaler_dropout_and_runtime_error_fallback(self) -> None:
        # 1. GradScaler active skips train_cuda_graphs
        cfg_gs = OptimizeConfig(profile="balanced")
        cfg_gs.compile.enabled = True
        cfg_gs.compile.train_cuda_graphs = True
        cfg_gs.amp.enabled = True
        cfg_gs.amp.dtype = "fp16"
        cfg_gs.amp.grad_scaler = True
        mgr_gs = HookManager()
        modes_gs: list[object] = []

        model_gs = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 16))
        with patch.dict(
            os.environ,
            {"PTOPT_COMPILE_CPU": "1", "PTOPT_CUDA_SM_OVERRIDE": "89", "PTOPT_TRAIN_CG_FORCE": "1"},
        ), patch.object(torch, "compile", side_effect=lambda fn, **kw: (modes_gs.append(kw.get("mode")) or fn)):
            install_auto_compile_hook(cfg_gs, mgr_gs, torch)
            try:
                x = torch.randn(16, 256)
                model_gs(x)
                model_gs(x)
                self.assertEqual(modes_gs, ["default"])
                self.assertTrue(
                    any(ev.pass_name == "compile.train_cuda_graphs" and ev.status == "SKIPPED" and "GradScaler" in ev.reason for ev in mgr_gs.events)
                )
            finally:
                uninstall_auto_compile_hook(torch)
                mgr_gs.reset()

        # 2. Dropout model skips train_cuda_graphs by default
        cfg_do = OptimizeConfig(profile="balanced")
        cfg_do.compile.enabled = True
        cfg_do.compile.train_cuda_graphs = True
        mgr_do = HookManager()
        model_do = nn.Sequential(nn.Linear(256, 256), nn.Dropout(0.1), nn.Linear(256, 16))
        with patch.dict(
            os.environ,
            {"PTOPT_COMPILE_CPU": "1", "PTOPT_CUDA_SM_OVERRIDE": "89", "PTOPT_TRAIN_CG_FORCE": "1"},
        ), patch.object(torch, "compile", side_effect=lambda fn, **_kw: fn):
            install_auto_compile_hook(cfg_do, mgr_do, torch)
            try:
                x = torch.randn(16, 256)
                model_do(x)
                model_do(x)
                self.assertFalse(getattr(model_do, "_ptopt_train_cg_active", False))
                self.assertTrue(
                    any(ev.pass_name == "compile.train_cuda_graphs" and ev.status == "SKIPPED" and "Dropout" in ev.reason for ev in mgr_do.events)
                )
            finally:
                uninstall_auto_compile_hook(torch)
                mgr_do.reset()

        # 3. Simulated runtime CUDAGraph error reverts cleanly to mode="default" without crashing user script
        cfg_err = OptimizeConfig(profile="balanced")
        cfg_err.compile.enabled = True
        cfg_err.compile.train_cuda_graphs = True
        mgr_err = HookManager()
        model_err = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 16))

        def _compile_fail_on_ro(fn: object, **kwargs: object) -> object:
            if kwargs.get("mode") == "reduce-overhead":
                def _boom(*_a: object, **_kw: object) -> object:
                    raise RuntimeError("Error: CUDAGraph tensor overwritten by subsequent run")
                return _boom
            return fn

        with patch.dict(
            os.environ,
            {"PTOPT_COMPILE_CPU": "1", "PTOPT_CUDA_SM_OVERRIDE": "89", "PTOPT_TRAIN_CG_FORCE": "1"},
        ), patch.object(torch, "compile", side_effect=_compile_fail_on_ro):
            install_auto_compile_hook(cfg_err, mgr_err, torch)
            try:
                x = torch.randn(16, 256)
                model_err(x)
                out = model_err(x)
                self.assertEqual(out.shape, (16, 16))
                self.assertFalse(getattr(model_err, "_ptopt_train_cg_active", False))
                self.assertTrue(
                    any(ev.pass_name == "compile.train_cuda_graphs" and ev.status == "FALLBACK" for ev in mgr_err.events)
                )
            finally:
                uninstall_auto_compile_hook(torch)
                mgr_err.reset()

    def test_max_autotune_downgrades_to_no_cudagraphs_when_ineligible(self) -> None:
        cfg = OptimizeConfig(profile="aggressive")
        cfg.compile.enabled = True
        cfg.compile.mode = "max-autotune"
        cfg.compile.train_cuda_graphs = True
        mgr = HookManager()
        seen_modes: list[object] = []

        model = nn.Sequential(nn.Linear(256, 256), nn.Dropout(0.1), nn.Linear(256, 16))
        with patch.dict(
            os.environ,
            {"PTOPT_COMPILE_CPU": "1", "PTOPT_CUDA_SM_OVERRIDE": "89", "PTOPT_TRAIN_CG_FORCE": "1"},
        ), patch.object(torch, "compile", side_effect=lambda fn, **kw: (seen_modes.append(kw.get("mode")) or fn)):
            install_auto_compile_hook(cfg, mgr, torch)
            try:
                model(torch.randn(8, 256))
                self.assertEqual(seen_modes, ["max-autotune-no-cudagraphs"])
            finally:
                uninstall_auto_compile_hook(torch)
                mgr.reset()

    def test_nested_dict_and_cyclic_container_shape_signature(self) -> None:
        from ptopt.hooks.module_dispatch import build_shape_signature

        cyclic: dict[str, object] = {"x": torch.randn(2, 4)}
        cyclic["self"] = cyclic
        sig = build_shape_signature((), {"batch": cyclic}, torch)
        self.assertEqual(len(sig), 1)
        self.assertEqual(sig[0][0], (2, 4))

        cfg = OptimizeConfig(profile="balanced")
        cfg.compile.enabled = True
        cfg.compile.dynamic = False
        mgr = HookManager()
        dyn_flags: list[object] = []

        class _DictModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(512, 512)

            def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
                return self.lin(batch["x"])

        model = _DictModel()
        with patch.dict(os.environ, {"PTOPT_COMPILE_CPU": "1"}), patch.object(
            torch, "compile", side_effect=lambda fn, **kw: (dyn_flags.append(kw.get("dynamic")) or fn)
        ):
            install_auto_compile_hook(cfg, mgr, torch)
            try:
                model({"x": torch.randn(2, 512)})
                model({"x": torch.randn(5, 512)})
                self.assertEqual(dyn_flags, [False, True])
            finally:
                uninstall_auto_compile_hook(torch)
                mgr.reset()

    def test_unified_dispatcher_and_train_eval_compiled_handle_isolation(self) -> None:
        from ptopt.passes.inference_opt import install_inference_opt_hook, uninstall_inference_opt_hook

        orig_call = nn.Module.__call__
        cfg = OptimizeConfig(profile="aggressive")
        cfg.compile.enabled = True
        cfg.compile.mode = "default"
        cfg.compile.train_cuda_graphs = False
        cfg.inference.inference_mode = True
        cfg.inference.cuda_graphs = True
        mgr = HookManager()
        compiled_modes: list[object] = []

        def _spy_compile(fn: object, **kw: object) -> object:
            mode = kw.get("mode")
            compiled_modes.append(mode)
            return lambda *a, **k: fn(*a, **k)

        model = nn.Linear(512, 512)
        with patch.dict(os.environ, {"PTOPT_COMPILE_CPU": "1"}), patch(
            "ptopt.passes.inference_opt._has_cuda_context", return_value=True
        ), patch.object(torch, "compile", side_effect=_spy_compile):
            install_inference_opt_hook(cfg, mgr, torch)
            install_auto_compile_hook(cfg, mgr, torch)
            self.assertIsNotNone(getattr(torch, "_ptopt_module_dispatcher", None))
            try:
                x = torch.randn(4, 512)
                model.train()
                _ = model(x)
                train_impl = getattr(model, "_ptopt_train_compiled_impl", None)
                self.assertIsNotNone(train_impl)

                # Switching to .eval() must NOT be skipped by train's compiled handle; it builds reduce-overhead handle
                model.eval()
                _ = model(x)
                eval_impl = getattr(model, "_ptopt_eval_compiled_impl", None)
                self.assertIsNotNone(eval_impl)
                self.assertIsNot(train_impl, eval_impl)
                self.assertEqual(compiled_modes, ["default", "reduce-overhead"])

                # Switching back to .train() restores train_impl cleanly
                model.train()
                _ = model(x)
                self.assertIs(model._compiled_call_impl, train_impl)
            finally:
                uninstall_auto_compile_hook(torch)
                uninstall_inference_opt_hook(torch)
                mgr.reset()

        self.assertIs(nn.Module.__call__, orig_call)
        self.assertIsNone(getattr(torch, "_ptopt_module_dispatcher", None))

    def test_dataloader_remainder_preempts_dynamic_default_compile_and_respects_override(self) -> None:
        from torch.utils.data import DataLoader, TensorDataset
        from ptopt.passes.dataloader_tune import install_dataloader_hook, uninstall_dataloader_hook

        ds = TensorDataset(torch.randn(20, 512))  # 20 % 8 == 4 != 0

        # 1. Default aggressive profile: DataLoader remainder preempts max-autotune/CUDA-graphs -> single dynamic=True, mode="default"
        cfg = OptimizeConfig(profile="aggressive")
        cfg.compile.enabled = True
        cfg.compile.mode = "max-autotune"
        cfg.compile.dynamic = False
        cfg.compile.train_cuda_graphs = True
        mgr = HookManager()
        compile_calls: list[tuple[object, object]] = []

        model = nn.Linear(512, 512)
        with patch.dict(
            os.environ,
            {"PTOPT_COMPILE_CPU": "1", "PTOPT_CUDA_SM_OVERRIDE": "89"},
        ), patch.object(
            torch,
            "compile",
            side_effect=lambda fn, **kw: (compile_calls.append((kw.get("mode"), kw.get("dynamic"))) or fn),
        ):
            install_dataloader_hook(cfg, mgr, torch, shm_probe_fn=lambda: 16384.0)
            install_auto_compile_hook(cfg, mgr, torch)
            try:
                dl = DataLoader(ds, batch_size=8, drop_last=False, num_workers=0)
                self.assertTrue(getattr(torch, "_ptopt_dataloader_has_remainder", False))
                for (xb,) in dl:
                    _ = model(xb)
                self.assertEqual(compile_calls, [("default", True)])
                self.assertTrue(
                    any(
                        ev.pass_name == "compile.auto"
                        and ev.status == "APPLIED"
                        and "dl_remainder" in ev.reason
                        for ev in mgr.events
                    )
                )
            finally:
                uninstall_auto_compile_hook(torch)
                uninstall_dataloader_hook(torch)
                mgr.reset()

        # 2. Explicit cli_overrides compile.dynamic=false disables pre-emption
        cfg_exp = OptimizeConfig(profile="balanced")
        cfg_exp.compile.enabled = True
        cfg_exp.compile.mode = "default"
        cfg_exp.compile.dynamic = False
        cfg_exp.rule_trace = [{"name": "cli_overrides", "changes": {"compile.dynamic": False}}]
        mgr_exp = HookManager()
        exp_calls: list[tuple[object, object]] = []
        model_exp = nn.Linear(512, 512)
        with patch.dict(os.environ, {"PTOPT_COMPILE_CPU": "1"}), patch.object(
            torch,
            "compile",
            side_effect=lambda fn, **kw: (exp_calls.append((kw.get("mode"), kw.get("dynamic"))) or fn),
        ):
            install_dataloader_hook(cfg_exp, mgr_exp, torch, shm_probe_fn=lambda: 16384.0)
            install_auto_compile_hook(cfg_exp, mgr_exp, torch)
            try:
                dl = DataLoader(ds, batch_size=8, drop_last=False, num_workers=0)
                for (xb,) in dl:
                    _ = model_exp(xb)
                self.assertEqual(exp_calls, [("default", False), ("default", True)])
            finally:
                uninstall_auto_compile_hook(torch)
                uninstall_dataloader_hook(torch)
                mgr_exp.reset()


if __name__ == "__main__":
    unittest.main()
