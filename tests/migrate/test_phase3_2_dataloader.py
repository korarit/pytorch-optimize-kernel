"""Migrate tests for Phase 3.2: DataLoader auto-tuning on cnn_vision & edge_cases."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "pipline-for-test"
WORKLOADS_DIR = BENCH_DIR / "workloads"
for _p in (str(REPO_ROOT), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from telemetry import check_parity

_TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_STEPS = 8 if _TARGET_DEVICE == "cuda" else 4


def _run_summary(cmd: list[str]) -> dict[str, object]:
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    for line in reversed(res.stdout.splitlines()):
        if line.startswith("WORKLOAD_SUMMARY_JSON: "):
            return json.loads(line.split("WORKLOAD_SUMMARY_JSON: ", 1)[1])
    raise AssertionError(f"Missing WORKLOAD_SUMMARY_JSON from {' '.join(cmd)}")


class TestPhase32DataLoaderMigration(unittest.TestCase):
    """Verify DataLoader auto-tuning accelerates/runs cnn_vision and safely skips unpicklable edge_cases."""

    def test_cnn_and_unpicklable_edge_case_with_dataloader_autotune(self) -> None:
        for script_name, extra in (
            ("cnn_vision.py", ["--batch-size", "16"]),
            ("edge_cases.py", ["--batch-size", "8", "--case", "unpicklable_dataset"]),
        ):
            script = str(WORKLOADS_DIR / script_name)
            base_args = ["--steps", str(_STEPS), "--seed", "42", "--device", _TARGET_DEVICE, *extra]
            base = _run_summary([sys.executable, script, *base_args])
            wrap = _run_summary([
                sys.executable,
                "-m",
                "src.cli.main",
                "--profile",
                "safe",
                "--override",
                "dataloader.auto_tune=true",
                "--override",
                "dataloader.num_workers=2",
                "--override",
                "dataloader.persistent_workers=true",
                "--",
                sys.executable,
                script,
                *base_args,
            ])
            parity = check_parity(base["losses"], wrap["losses"], profile="safe")
            self.assertTrue(parity.passed, f"{script_name} parity failed: {parity.reason}")


if __name__ == "__main__":
    unittest.main()
