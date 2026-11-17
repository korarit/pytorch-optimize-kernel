"""Unit tests for Phase 4.3 Automated Regression Gate (`scripts/run_gate.py`)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_gate import GateStepResult, build_gate_report, discover_migrate_files
from src.cli.explain import build_mock_fingerprint


class TestRegressionGate(unittest.TestCase):
    """Verify fast vs full file discovery and Phase 4 report formatting."""

    def test_discover_migrate_files_fast_excludes_heavy_cpu_compile(self) -> None:
        fast_files = discover_migrate_files(mode="fast")
        full_files = discover_migrate_files(mode="full")
        fast_names = {p.name for p in fast_files}
        full_names = {p.name for p in full_files}

        self.assertIn("test_phase3_5_compile.py", full_names)
        self.assertNotIn("test_phase3_5_compile.py", fast_names)
        self.assertIn("test_code_migration_matrix.py", fast_names)
        self.assertIn("test_checkpoint_migration_matrix.py", fast_names)
        self.assertIn("test_config_cache_migration.py", fast_names)

    def test_build_gate_report_outputs_fast_and_full_formats(self) -> None:
        fp = build_mock_fingerprint("l4")
        steps = [
            GateStepResult("CLI: ptopt --doctor", "probe", 0.5, True, "OK"),
            GateStepResult("Unit Suite", "unit", 2.0, True, "Ran 99 tests -> OK"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            fast_path, fast_txt, fast_ok = build_gate_report(
                mode="fast",
                fp=fp,
                cache_key="sm_89-test",
                step_results=steps,
                report_dir=Path(tmp),
            )
            self.assertTrue(fast_ok)
            self.assertTrue(fast_path.name.endswith("-phase-4-fast.txt"))
            self.assertIn("Overall Verdict       : PASS", fast_txt)
            self.assertIn("PENDING_USER_REVIEW", fast_txt)

            full_path, full_txt, full_ok = build_gate_report(
                mode="full",
                fp=fp,
                cache_key="sm_89-test",
                step_results=steps,
                bench_rows=[["transformer_lm", "balanced", "99.1", "26.6", "3.73x", "1734/1033 (-701)", "7e-6", "PASS"]],
                bench_ok=True,
                report_dir=Path(tmp),
            )
            self.assertTrue(full_ok)
            self.assertTrue(full_path.name.endswith("-phase-4.txt"))
            self.assertIn("Smoke Benchmark (Baseline vs supersafe / safe / balanced / aggressive)", full_txt)
            self.assertIn("3.73x", full_txt)

    def test_run_gate_cli_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/run_gate.py"), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--mode", proc.stdout)
        self.assertIn("--bench-steps", proc.stdout)


if __name__ == "__main__":
    unittest.main()
