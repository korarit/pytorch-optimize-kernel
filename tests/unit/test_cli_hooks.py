"""Unit tests for Phase 2.1: CLI Launcher, sitecustomize chain-loader, idempotency, and worker propagation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.main import build_child_env, main as ptopt_main, parse_ptopt_argv
from src.config import OptimizeConfig
from src.hooks import HookManager, chain_load_sitecustomize


class TestCLIArgParsing(unittest.TestCase):
    """Tests for splitting ptopt flags from wrapped user commands."""

    def test_split_with_separator_preserves_duplicate_user_flags(self) -> None:
        parsed = parse_ptopt_argv([
            "--profile", "safe",
            "--no-compile",
            "--",
            "python", "train.py", "--profile", "user_prof", "--device", "cuda", "--no-compile",
        ])
        self.assertEqual(parsed.ptopt_args.profile, "safe")
        self.assertTrue(parsed.ptopt_args.no_compile)
        self.assertEqual(
            parsed.command,
            ["python", "train.py", "--profile", "user_prof", "--device", "cuda", "--no-compile"],
        )

    def test_split_without_separator_stops_at_script_or_executable(self) -> None:
        parsed = parse_ptopt_argv([
            "--profile", "aggressive",
            "--amp-dtype=fp16",
            "train.py", "--profile", "user_prof", "--lr", "1e-3",
        ])
        self.assertEqual(parsed.ptopt_args.profile, "aggressive")
        self.assertEqual(parsed.ptopt_args.amp_dtype, "fp16")
        self.assertEqual(
            parsed.command,
            [sys.executable, "train.py", "--profile", "user_prof", "--lr", "1e-3"],
        )

    def test_module_invocation_prepends_python_executable(self) -> None:
        parsed = parse_ptopt_argv(["--profile", "balanced", "-m", "pkg.train", "--epochs", "2"])
        self.assertEqual(parsed.command, [sys.executable, "-m", "pkg.train", "--epochs", "2"])


class TestHookManagerLifecycle(unittest.TestCase):
    """Tests for HookManager idempotency and error fallback isolation."""

    def test_bootstrap_idempotency_and_pass_execution(self) -> None:
        mgr = HookManager()
        calls: list[str] = []

        def _sample_pass(cfg: OptimizeConfig, _m: HookManager) -> str:
            calls.append(cfg.profile)
            return "applied sample pass"

        mgr.register_pass("sample", _sample_pass)
        mgr.register_pass("sample", _sample_pass)  # duplicate name ignored

        cfg_json = json.dumps(OptimizeConfig(profile="safe").to_dict())
        env = {"PTOPT_ACTIVE": "1", "PTOPT_CONFIG_JSON": cfg_json}

        saved_torch = sys.modules.pop("torch", None)
        try:
            first = mgr.bootstrap(env=env)
            second = mgr.bootstrap(env=env)
            self.assertTrue(first)
            self.assertFalse(second)

            # Even if torch was not yet imported when bootstrap ran, calling apply_torch_hooks twice only runs pass once.
            mgr.apply_torch_hooks(SimpleNamespace(__name__="torch"))
            mgr.apply_torch_hooks(SimpleNamespace(__name__="torch"))
            self.assertEqual(calls, ["safe"])
            self.assertEqual(len(mgr.events), 1)
            self.assertEqual(mgr.events[0].status, "APPLIED")
        finally:
            mgr.reset()
            if saved_torch is not None:
                sys.modules["torch"] = saved_torch

    def test_fallback_vs_strict_mode(self) -> None:
        mgr = HookManager()

        def _broken_pass(_cfg: OptimizeConfig, _m: HookManager) -> str:
            raise RuntimeError("boom")

        saved_torch = sys.modules.pop("torch", None)
        try:
            mgr.register_pass("broken", _broken_pass)
            mgr.bootstrap(env={"PTOPT_ACTIVE": "1", "PTOPT_STRICT": "0"})
            mgr.apply_torch_hooks(SimpleNamespace(__name__="torch"))
            self.assertEqual(len(mgr.events), 1)
            self.assertEqual(mgr.events[0].status, "FALLBACK")
            self.assertIn("boom", mgr.events[0].error or "")
            mgr.reset()

            mgr_strict = HookManager()
            mgr_strict.register_pass("broken", _broken_pass)
            mgr_strict.bootstrap(env={"PTOPT_ACTIVE": "1", "PTOPT_STRICT": "1"})
            with self.assertRaises(RuntimeError):
                mgr_strict.apply_torch_hooks(SimpleNamespace(__name__="torch"))
            mgr_strict.reset()
        finally:
            if saved_torch is not None:
                sys.modules["torch"] = saved_torch


class TestSubprocessAndColabPlumbing(unittest.TestCase):
    """End-to-end subprocess tests for exit code, chain-loading, lazy hook, and spawn propagation."""

    def test_exit_code_propagation(self) -> None:
        for code in (0, 1, 42):
            rc = ptopt_main(["--profile", "safe", "--", sys.executable, "-c", f"import sys; sys.exit({code})"])
            self.assertEqual(rc, code)

    def test_sitecustomize_chain_loading_and_bootstrapped_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            downstream_dir = Path(tmp) / "downstream"
            downstream_dir.mkdir()
            (downstream_dir / "sitecustomize.py").write_text(
                "import os\nos.environ['DOWNSTREAM_SITECUSTOMIZE_RAN'] = '1'\n",
                encoding="utf-8",
            )

            env = build_child_env(
                OptimizeConfig(profile="balanced").to_dict(),
                base_env={**os.environ, "PYTHONPATH": str(downstream_dir)},
                repo_root=REPO_ROOT,
            )
            probe_code = (
                "import os, json\n"
                "from src.hooks import get_hook_manager\n"
                "m = get_hook_manager()\n"
                "print(json.dumps({"
                "'bootstrapped': m.bootstrapped, "
                "'env_marker': os.environ.get('_PTOPT_BOOTSTRAPPED'), "
                "'downstream': os.environ.get('DOWNSTREAM_SITECUSTOMIZE_RAN')"
                "}))\n"
            )
            res = subprocess.run([sys.executable, "-c", probe_code], env=env, capture_output=True, text=True, check=True)
            data = json.loads(res.stdout.strip().splitlines()[-1])
            self.assertTrue(data["bootstrapped"])
            self.assertEqual(data["env_marker"], "1")
            self.assertEqual(data["downstream"], "1")

    def test_chain_load_helper_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_site_dir = Path(tmp) / "site_extra"
            fake_site_dir.mkdir()
            (fake_site_dir / "sitecustomize.py").write_text(
                "import os\nos.environ['DIRECT_CHAIN_TEST'] = 'ok'\n",
                encoding="utf-8",
            )
            sys.path.append(str(fake_site_dir))
            try:
                loaded = chain_load_sitecustomize(REPO_ROOT / "src" / "hooks" / "bootstrap" / "sitecustomize.py")
                self.assertTrue(any(p.parent == fake_site_dir.resolve() for p in loaded))
                self.assertEqual(os.environ.get("DIRECT_CHAIN_TEST"), "ok")
            finally:
                sys.path.remove(str(fake_site_dir))
                os.environ.pop("DIRECT_CHAIN_TEST", None)

    def test_lazy_torch_import_hook_and_spawn_worker_propagation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "user_script.py"
            script_path.write_text(
                "\n".join([
                    "import json, multiprocessing, os, subprocess, sys",
                    "from src.hooks import get_hook_manager",
                    "def _worker(q):",
                    "    mgr = get_hook_manager()",
                    "    q.put({'active': os.environ.get('PTOPT_ACTIVE'), 'bootstrapped': mgr.bootstrapped})",
                    "if __name__ == '__main__':",
                    "    mgr = get_hook_manager()",
                    "    seen = []",
                    "    mgr.register_pass('probe', lambda cfg, _m: seen.append(cfg.profile) or 'ok')",
                    "    before_import = list(seen)",
                    "    import torch  # noqa: F401",
                    "    after_import = list(seen)",
                    "    ctx = multiprocessing.get_context('spawn')",
                    "    q = ctx.Queue()",
                    "    p = ctx.Process(target=_worker, args=(q,))",
                    "    p.start(); p.join(timeout=15)",
                    "    worker_info = q.get(timeout=5)",
                    "    sub = subprocess.run([sys.executable, '-c', 'import os; print(os.environ.get(\"PTOPT_ACTIVE\"))'], capture_output=True, text=True, check=True)",
                    "    print(json.dumps({'before': before_import, 'after': after_import, 'patched': mgr.torch_patched, 'worker': worker_info, 'sub': sub.stdout.strip()}))",
                ]) + "\n",
                encoding="utf-8",
            )

            env = build_child_env(
                OptimizeConfig(profile="safe").to_dict(),
                repo_root=REPO_ROOT,
            )
            res = subprocess.run([sys.executable, str(script_path)], env=env, capture_output=True, text=True, check=True)
            data = json.loads(res.stdout.strip().splitlines()[-1])
            self.assertEqual(data["before"], [])
            self.assertEqual(data["after"], ["safe"])
            self.assertTrue(data["patched"])
            self.assertEqual(data["worker"], {"active": "1", "bootstrapped": True})
            self.assertEqual(data["sub"], "1")


if __name__ == "__main__":
    unittest.main()
