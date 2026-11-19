"""Unit tests for Phase 1.3: --explain, --show-config, --dump-config, and --target-arch."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptopt.cli.explain import (
    build_config_snapshot_dict,
    build_mock_fingerprint,
    format_explain_text,
    run_explain_cli,
)
from ptopt.config import ConfigResolver, ConfigValidationError, OptimizeConfig


class TestExplainCLI(unittest.TestCase):
    """Tests for explain formatting, JSON snapshot dumping, and target arch presets."""

    def test_explain_contains_all_sections(self) -> None:
        fp = build_mock_fingerprint("l4")
        res = ConfigResolver().resolve(fp, profile="balanced", env={})
        text = format_explain_text(fp, res)
        for marker in ("[HARDWARE FINGERPRINT]", "[MATCHED RULES]", "[RESOLUTION TRACE]", "[FINAL RESOLVED CONFIG]"):
            self.assertIn(marker, text)
        self.assertIn("NVIDIA L4", text)
        self.assertIn("gpu_cuda_ada_sm89", text)

    def test_dump_config_roundtrip_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dump_path = Path(tmp) / "nested" / "snapshot.json"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = run_explain_cli([
                    "--mock-profile", "a100",
                    "--profile", "aggressive",
                    "--dump-config", str(dump_path),
                    "--show-config",
                ])
            self.assertEqual(rc, 0)
            self.assertTrue(dump_path.is_file())

            # --show-config stdout must parse directly as OptimizeConfig
            shown_cfg = OptimizeConfig.from_dict(json.loads(buf.getvalue()))
            self.assertEqual(shown_cfg.profile, "aggressive")
            self.assertTrue(shown_cfg.runtime.tf32)
            self.assertEqual(shown_cfg.amp.dtype, "bf16")

            # --dump-config snapshot must contain config & hardware_fingerprint
            snap = json.loads(dump_path.read_text(encoding="utf-8"))
            reloaded = OptimizeConfig.from_dict(snap["config"])
            self.assertEqual(reloaded.to_dict(), shown_cfg.to_dict())
            self.assertEqual(snap["hardware_fingerprint"]["gpu"]["devices"][0]["sm_name"], "sm_80")

    def test_target_arch_presets_and_overrides(self) -> None:
        for target, expected_tf32, expected_dtype in (
            ("sm_75", False, "fp16"),
            ("sm_80", True, "bf16"),
            ("sm_89", True, "bf16"),
            ("cpu", False, "fp32"),
        ):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = run_explain_cli(["--target-arch", target, "--profile", "balanced", "--show-config"])
            self.assertEqual(rc, 0)
            cfg = OptimizeConfig.from_dict(json.loads(buf.getvalue()))
            self.assertEqual(cfg.runtime.tf32, expected_tf32)
            self.assertEqual(cfg.amp.dtype, expected_dtype)

        # Override test
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_explain_cli([
                "--target-arch", "sm_89",
                "--profile", "balanced",
                "--override", "compile.enabled=false",
                "--show-config",
            ])
        cfg_ovr = OptimizeConfig.from_dict(json.loads(buf.getvalue()))
        self.assertFalse(cfg_ovr.compile.enabled)

    def test_invalid_target_arch_raises(self) -> None:
        with self.assertRaises(ConfigValidationError):
            build_mock_fingerprint("unknown_gpu")

    def test_snapshot_dict_helper(self) -> None:
        fp = build_mock_fingerprint("t4")
        res = ConfigResolver().resolve(fp, profile="safe", env={})
        snap = build_config_snapshot_dict(fp, res)
        self.assertEqual(snap["schema_version"], 1)
        self.assertIn("trace", snap)


if __name__ == "__main__":
    unittest.main()
