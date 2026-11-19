"""Unit tests for Phase 4.2: `ptopt --doctor` diagnostics and `--fallback-report` formatting."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptopt.cli.doctor import format_doctor_text, run_doctor
from ptopt.cli.explain import build_mock_fingerprint
from ptopt.cli.main import main as ptopt_main
from ptopt.config import OptimizeConfig
from ptopt.hooks.fallback_reporter import (
    build_run_text_report,
    classify_events,
    format_fallback_report,
)
from ptopt.hooks.manager import FallbackEvent, HookManager


class TestDoctorDiagnostics(unittest.TestCase):
    """Tests for `ptopt --doctor` across CPU-only, low-/dev/shm, missing Triton, and bad paths."""

    def test_doctor_cpu_only_and_missing_triton_reports_missing_and_warns_without_crashing(self) -> None:
        fp_cpu = build_mock_fingerprint("cpu")
        with tempfile.TemporaryDirectory() as tmp:
            report = run_doctor(fp=fp_cpu, cache_dir=Path(tmp) / "cache", report_dir=Path(tmp) / "reports")
            text = format_doctor_text(report)

        self.assertEqual(report.recommended_profile, "supersafe")
        by_name = {c.name: c for c in report.checks}
        self.assertEqual(by_name["CUDA Driver & GPU"].status, "MISSING")
        self.assertEqual(by_name["Triton Compiler"].status, "WARN")
        self.assertEqual(by_name["TORCHINDUCTOR_CACHE_DIR"].status, "OK")
        self.assertIn("[MISSING] CUDA Driver & GPU", text)
        self.assertIn("Profile : supersafe", text)

    def test_doctor_cuda_missing_triton_low_shm_and_unwritable_dir(self) -> None:
        fp_low_t4 = build_mock_fingerprint("low_shm_t4")
        fp_no_triton = replace(
            fp_low_t4,
            host=replace(fp_low_t4.host, triton_version=None, jemalloc_path=None, tcmalloc_path=None),
        )
        with tempfile.TemporaryDirectory() as tmp:
            readonly_file = Path(tmp) / "not_a_dir"
            readonly_file.write_text("file", encoding="utf-8")
            with patch("tempfile.NamedTemporaryFile", side_effect=PermissionError("read-only")):
                report = run_doctor(
                    fp=fp_no_triton,
                    cache_dir=Path(tmp) / "cache",
                    report_dir=Path(tmp) / "report",
                )
            text = format_doctor_text(report)

        self.assertEqual(report.recommended_profile, "balanced")
        by_name = {c.name: c for c in report.checks}
        self.assertEqual(by_name["CUDA Driver & GPU"].status, "OK")
        self.assertEqual(by_name["Triton Compiler"].status, "MISSING")
        self.assertEqual(by_name["/dev/shm Shared Memory"].status, "WARN")
        self.assertEqual(by_name["TORCHINDUCTOR_CACHE_DIR"].status, "MISSING")
        self.assertEqual(by_name["Report Output Directory"].status, "MISSING")
        self.assertIn("fp16 AMP", report.recommendation_reason)
        self.assertIn("pip install triton", text)
        self.assertGreaterEqual(report.missing_count, 3)

    def test_doctor_cli_exit_and_mock_target(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ptopt_main(["--target-arch", "l4", "--doctor"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("NVIDIA L4", out)
        self.assertIn("[Recommended Profile]", out)


class TestFallbackReporter(unittest.TestCase):
    """Tests for `--fallback-report` formatting, HookManager integration, and `--save-report`."""

    def test_classify_and_format_applied_skipped_fallback(self) -> None:
        events = [
            FallbackEvent("runtime.flags", "APPLIED", "tf32=on, flash_sdp=True"),
            FallbackEvent("kernels.auto", "SKIPPED", "kernels.enabled is False"),
            FallbackEvent("amp.auto", "FALLBACK", "Runtime error during pass init", "simulated AMP boom"),
            FallbackEvent("config.migrate", "MIGRATED", "schema v0 -> v1"),
        ]
        grouped = classify_events(events)
        self.assertEqual(len(grouped.applied), 1)
        self.assertEqual(len(grouped.skipped), 1)
        self.assertEqual(len(grouped.fallbacks), 1)
        self.assertEqual(len(grouped.migrated), 1)

        text = format_fallback_report(events, profile="balanced", exit_code=0)
        self.assertIn("[APPLIED] runtime.flags: tf32=on", text)
        self.assertIn("[SKIPPED (by arch/config)] kernels.auto: kernels.enabled is False", text)
        self.assertIn("[FALLBACK (runtime error)] amp.auto: Runtime error during pass init | error=simulated AMP boom", text)

        with tempfile.TemporaryDirectory() as tmp:
            fp = build_mock_fingerprint("cpu")
            saved = build_run_text_report(
                events,
                run_name="unit smoke",
                fp=fp,
                profile="balanced",
                command=["python", "train.py"],
                exit_code=0,
                report_dir=Path(tmp),
                repo_root=REPO_ROOT,
            )
            self.assertTrue(saved.is_file())
            self.assertIn("-run-unit-smoke.txt", saved.name)
            body = saved.read_text(encoding="utf-8")
            self.assertIn("FALLBACK (runtime error)", body)
            self.assertIn("simulated AMP boom", body)

    def test_hook_manager_records_fallback_and_formats_summary(self) -> None:
        mgr = HookManager()

        def _ok(_cfg: OptimizeConfig, _m: HookManager) -> str:
            return "ok pass"

        def _skip(_cfg: OptimizeConfig, _m: HookManager) -> str:
            return "SKIPPED: not supported on CPU"

        def _fail(_cfg: OptimizeConfig, _m: HookManager) -> str:
            raise ValueError("mock pass failure")

        saved_torch = sys.modules.pop("torch", None)
        try:
            mgr.register_pass("p.ok", _ok)
            mgr.register_pass("p.skip", _skip)
            mgr.register_pass("p.fail", _fail)
            mgr.bootstrap(env={"PTOPT_ACTIVE": "1", "PTOPT_STRICT": "0", "PTOPT_FALLBACK_REPORT": "1"})
            mgr.apply_torch_hooks(SimpleNamespace(__name__="torch"))
            summary = mgr.format_events_summary()
            self.assertIn("[APPLIED] p.ok: ok pass", summary)
            self.assertIn("[SKIPPED (by arch/config)] p.skip: not supported on CPU", summary)
            self.assertIn("[FALLBACK (runtime error)] p.fail: Runtime error during pass init | error=mock pass failure", summary)
        finally:
            mgr.reset()
            if saved_torch is not None:
                sys.modules["torch"] = saved_torch

    def test_cli_fallback_report_and_save_report_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run_reports"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ptopt.cli.main",
                    "--profile",
                    "safe",
                    "--fallback-report",
                    "--save-report",
                    str(out_dir),
                    "--run-name",
                    "e2e_diag",
                    "--",
                    sys.executable,
                    "-c",
                    "import torch; x = torch.nn.Linear(4, 2)(torch.randn(2, 4)); print(float(x.sum()))",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + "\n" + proc.stdout)
            self.assertIn("[ptopt fallback/hook summary]", proc.stderr)
            self.assertIn("SKIPPED (by arch/config)", proc.stderr)
            files = list(out_dir.glob("*-run-e2e_diag.txt"))
            self.assertEqual(len(files), 1, f"Expected saved run report in {out_dir}, got {files}")
            content = files[0].read_text(encoding="utf-8")
            self.assertIn("Pass & Fallback Audit", content)

    def test_doctor_and_reporter_preserve_lazy_torch_invariant(self) -> None:
        code = (
            "import sys; "
            "from ptopt.cli.explain import build_mock_fingerprint; "
            "from ptopt.cli.doctor import run_doctor, format_doctor_text; "
            "from ptopt.hooks.fallback_reporter import format_fallback_report; "
            "rep = run_doctor(fp=build_mock_fingerprint('t4')); "
            "_ = format_doctor_text(rep); "
            "_ = format_fallback_report([]); "
            "assert 'torch' not in sys.modules, 'torch eagerly imported'"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
