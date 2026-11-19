"""Migrate tests for Phase 3.3: Fused Optimizer + Auto-AMP (bf16/fp16) & preexisting_amp safety."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
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


class TestPhase33Migration(unittest.TestCase):
    """Verify Fused Optimizer + Auto-AMP preserve edge_cases & checkpoint cross-compatibility."""

    def test_preexisting_amp_edge_case_and_transformer_with_auto_amp(self) -> None:
        for script_name, extra in (
            ("edge_cases.py", ["--case", "preexisting_amp", "--batch-size", "8"]),
            ("transformer_lm.py", ["--batch-size", "8"]),
        ):
            script = str(WORKLOADS_DIR / script_name)
            base_args = ["--steps", str(_STEPS), "--seed", "42", "--device", _TARGET_DEVICE, *extra]
            base = _run_summary([sys.executable, script, *base_args])
            wrap = _run_summary([
                sys.executable,
                "-m",
                "ptopt.cli.main",
                "--profile",
                "balanced",
                "--no-compile",
                "--no-dataloader-tune",
                "--",
                sys.executable,
                script,
                *base_args,
            ])
            parity = check_parity(base["losses"], wrap["losses"], profile="balanced")
            self.assertTrue(parity.passed, f"{script_name} parity failed: {parity.reason}")

    def test_auto_amp_checkpoint_cross_load_to_baseline(self) -> None:
        script = str(WORKLOADS_DIR / "transformer_lm.py")
        common = ["--batch-size", "4", "--seed", "77", "--device", _TARGET_DEVICE]
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "amp_ckpt.pt"
            wrap_train = _run_summary([
                sys.executable,
                "-m",
                "ptopt.cli.main",
                "--profile",
                "balanced",
                "--no-compile",
                "--no-dataloader-tune",
                "--",
                sys.executable,
                script,
                "--steps",
                "4",
                "--save-ckpt",
                str(ckpt),
                *common,
            ])
            self.assertEqual(len(wrap_train["losses"]), 4)
            self.assertTrue(ckpt.is_file())

            base_resume = _run_summary([
                sys.executable,
                script,
                "--steps",
                "4",
                "--load-ckpt",
                str(ckpt),
                *common,
            ])
            self.assertEqual(len(base_resume["losses"]), 4)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in base_resume["losses"]))


if __name__ == "__main__":
    unittest.main()
