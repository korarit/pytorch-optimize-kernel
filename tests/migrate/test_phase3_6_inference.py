"""Migrate tests for Phase 3.6: Inference optimization parity and train/eval coexistence."""

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
_INFER_STEPS = 12 if _TARGET_DEVICE == "cuda" else 6
_TRAIN_STEPS = 8 if _TARGET_DEVICE == "cuda" else 4


def _run_summary(cmd: list[str]) -> dict[str, object]:
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    for line in reversed(res.stdout.splitlines()):
        if line.startswith("WORKLOAD_SUMMARY_JSON: "):
            return json.loads(line.split("WORKLOAD_SUMMARY_JSON: ", 1)[1])
    raise AssertionError(f"Missing WORKLOAD_SUMMARY_JSON from {' '.join(cmd)}")


class TestPhase36InferenceMigration(unittest.TestCase):
    """Verify unmodified inference and training scripts pass parity with inference_opt enabled."""

    def test_infer_runner_bs1_and_bs32_parity(self) -> None:
        script = str(WORKLOADS_DIR / "infer_runner.py")
        for bs in (1, 32):
            common = ["--steps", str(_INFER_STEPS), "--batch-size", str(bs), "--seed", "42", "--device", _TARGET_DEVICE]
            base = _run_summary([sys.executable, script, *common])
            wrap_safe = _run_summary([
                sys.executable,
                "-m",
                "ptopt.cli.main",
                "--profile",
                "safe",
                "--no-compile",
                "--",
                sys.executable,
                script,
                *common,
            ])
            p_safe = check_parity(base["losses"], wrap_safe["losses"], profile="safe")
            self.assertTrue(p_safe.passed, f"infer_runner bs={bs} safe parity failed: {p_safe.reason}")

            wrap_bal = _run_summary([
                sys.executable,
                "-m",
                "ptopt.cli.main",
                "--profile",
                "balanced",
                "--no-compile",
                "--",
                sys.executable,
                script,
                *common,
            ])
            p_bal = check_parity(base["losses"], wrap_bal["losses"], profile="balanced")
            self.assertTrue(p_bal.passed, f"infer_runner bs={bs} balanced parity failed: {p_bal.reason}")

    def test_cnn_vision_training_unaffected_by_inference_opt(self) -> None:
        script = str(WORKLOADS_DIR / "cnn_vision.py")
        common = ["--steps", str(_TRAIN_STEPS), "--batch-size", "16", "--seed", "42", "--device", _TARGET_DEVICE]
        base = _run_summary([sys.executable, script, *common])
        wrap = _run_summary([
            sys.executable,
            "-m",
            "ptopt.cli.main",
            "--profile",
            "balanced",
            "--no-compile",
            "--",
            sys.executable,
            script,
            *common,
        ])
        parity = check_parity(base["losses"], wrap["losses"], profile="balanced")
        self.assertTrue(parity.passed, f"cnn_vision under inference_opt parity failed: {parity.reason}")


if __name__ == "__main__":
    unittest.main()
