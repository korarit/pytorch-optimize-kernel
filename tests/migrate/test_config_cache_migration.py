"""Phase 4.1 Axis 3: End-to-End Config Schema v0->v1 & Inductor/Triton Cache-Key Migration Matrix."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.main import build_child_env
from src.config import (
    CACHE_META_FILENAME,
    SCHEMA_VERSION,
    build_runtime_cache_key,
    detect_runtime_cache_key,
    validate_or_scope_cache_dir,
)
from src.hooks.manager import HookManager

_TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestConfigAndCacheMigrationMatrix(unittest.TestCase):
    """Axis 3 Migrate Matrix: v0 config upgrade and cross-GPU/PyTorch/Triton cache isolation."""

    def test_v0_config_migrates_end_to_end_in_hook_manager_and_subprocess(self) -> None:
        legacy_cfg = {
            "version": 0,
            "profile": "fast",
            "tf32": False,
            "amp": True,
            "amp_dtype": "bfloat16",
            "grad_scaler": True,
            "compile": True,
            "compile_mode": "max_autotune",
            "optimizer": "fused",
            "experimental": {"legacy_flag": True},
        }
        # In-process HookManager bootstrap directly from legacy PTOPT_CONFIG_JSON
        mgr = HookManager()
        env_raw = {"PTOPT_ACTIVE": "1", "PTOPT_CONFIG_JSON": json.dumps(legacy_cfg)}
        with tempfile.TemporaryDirectory() as tmp:
            # Verify HookManager migrates without eager torch import side effects breaking
            old_active = os.environ.get("_PTOPT_BOOTSTRAPPED")
            try:
                mgr.bootstrap(env_raw)
                assert mgr.config is not None
                self.assertEqual(mgr.config.schema_version, SCHEMA_VERSION)
                self.assertEqual(mgr.config.profile, "aggressive")
                self.assertEqual(mgr.config.amp.dtype, "bf16")
                self.assertFalse(mgr.config.amp.grad_scaler)
                self.assertEqual(mgr.config.compile.mode, "max-autotune")
                self.assertTrue(any(ev.status == "MIGRATED" for ev in mgr.events))
            finally:
                mgr.reset()
                if old_active is None:
                    os.environ.pop("_PTOPT_BOOTSTRAPPED", None)
                else:
                    os.environ["_PTOPT_BOOTSTRAPPED"] = old_active

            # End-to-end child process via build_child_env(legacy_cfg)
            legacy_cfg["cache_dir"] = str(Path(tmp) / "migrated_cache")
            legacy_cfg["tf32"] = False
            legacy_cfg["fused_optimizer"] = False
            legacy_cfg["cuda_graphs"] = False
            child_env = build_child_env(legacy_cfg, verbose=False, repo_root=REPO_ROOT)
            script = textwrap.dedent(
                f"""
                import json, torch
                from src.hooks.manager import get_hook_manager
                x = torch.randn(4, 16, device={_TARGET_DEVICE!r})
                m = torch.nn.Linear(16, 4).to({_TARGET_DEVICE!r})
                y = m(x).sum().item()
                mgr = get_hook_manager()
                print("MIG_CHILD: " + json.dumps({{
                    "schema_version": mgr.config.schema_version,
                    "profile": mgr.config.profile,
                    "amp_dtype": mgr.config.amp.dtype,
                    "finite": bool(torch.isfinite(torch.tensor(y)).item()),
                }}))
                """
            )
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=REPO_ROOT,
                env=child_env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + "\n" + proc.stdout)
            payload_line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("MIG_CHILD: "))
            payload = json.loads(payload_line.split("MIG_CHILD: ", 1)[1])
            self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
            self.assertEqual(payload["profile"], "aggressive")
            self.assertEqual(payload["amp_dtype"], "bf16")
            self.assertTrue(payload["finite"])

    def test_stale_cross_arch_or_torch_triton_cache_isolated_in_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shared_cache = Path(tmp) / "inductor_shared"
            stale_key = build_runtime_cache_key("sm_70", "1.13.1+cu117", "1.0.0")
            validate_or_scope_cache_dir(shared_cache, stale_key, write_meta=True, emit_warnings=False)
            stale_sentinel = shared_cache / "stale_v100_kernel.so"
            stale_sentinel.write_bytes(b"STALE_CACHE")

            active_key = detect_runtime_cache_key(torch)
            self.assertNotEqual(active_key.cache_key, stale_key.cache_key)

            worker = Path(tmp) / "cache_probe.py"
            worker.write_text(
                textwrap.dedent(
                    """
                    import json, os, torch
                    m = torch.nn.Linear(8, 4)
                    out = m(torch.randn(2, 8)).sum().item()
                    print("CACHE_PROBE: " + json.dumps({
                        "inductor_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                        "triton_dir": os.environ.get("TRITON_CACHE_DIR"),
                        "cache_key": os.environ.get("PTOPT_CACHE_KEY"),
                        "out": float(out),
                    }))
                    """
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.cli.main",
                    "--profile",
                    "balanced",
                    "--override",
                    "compile.enabled=true",
                    "--override",
                    f"compile.cache_dir={shared_cache}",
                    "--",
                    sys.executable,
                    str(worker),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + "\n" + proc.stdout)
            probe_line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("CACHE_PROBE: "))
            probe = json.loads(probe_line.split("CACHE_PROBE: ", 1)[1])

            expected_scoped = str((shared_cache / active_key.cache_key).resolve())
            self.assertEqual(probe["cache_key"], active_key.cache_key)
            self.assertEqual(probe["inductor_dir"], expected_scoped)
            self.assertEqual(probe["triton_dir"], str((Path(expected_scoped) / "triton").resolve()))
            self.assertTrue((Path(expected_scoped) / CACHE_META_FILENAME).is_file())
            self.assertFalse((Path(expected_scoped) / "stale_v100_kernel.so").exists())
            self.assertTrue(stale_sentinel.is_file())

            # Run again pointing directly at scoped cache -> must reuse cleanly with zero nested subdir
            reuse_res = validate_or_scope_cache_dir(expected_scoped, active_key, write_meta=True, emit_warnings=False)
            self.assertTrue(reuse_res.reused)
            self.assertFalse(reuse_res.rotated)
            self.assertEqual(str(reuse_res.cache_dir), expected_scoped)


if __name__ == "__main__":
    unittest.main()
