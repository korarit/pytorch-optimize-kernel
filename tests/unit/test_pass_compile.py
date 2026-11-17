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

from src.config import OptimizeConfig
from src.hooks import HookManager
from src.hooks.checkpoint_compat import (
    add_orig_mod_prefix,
    install_checkpoint_compat_hooks,
    strip_orig_mod_prefix,
    uninstall_checkpoint_compat_hooks,
)
from src.passes.auto_compile import (
    apply_auto_compile_pass,
    install_auto_compile_hook,
    uninstall_auto_compile_hook,
)
from src.passes.memory_allocator import apply_prelaunch_env


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

    def test_dynamic_shape_switches_once_to_dynamic_true(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.compile.enabled = True
        cfg.compile.dynamic = False
        mgr = HookManager()
        dynamic_flags: list[object] = []

        def _spy_compile(fn: object, **kwargs: object) -> object:
            dynamic_flags.append(kwargs.get("dynamic"))
            return fn

        model = nn.Linear(4, 2)
        with patch.dict(os.environ, {"PTOPT_COMPILE_CPU": "1"}), patch.object(
            torch, "compile", side_effect=_spy_compile
        ):
            rt = install_auto_compile_hook(cfg, mgr, torch)
            assert rt is not None
            try:
                model(torch.randn(2, 4))
                model(torch.randn(3, 4))
                model(torch.randn(5, 4))
                model(torch.randn(7, 4))
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


if __name__ == "__main__":
    unittest.main()
