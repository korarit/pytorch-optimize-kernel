"""Migrate tests for Phase 3.7: transformer_lm under `aggressive` (Custom Kernels ON) & checkpoint compat."""

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


class TestPhase37KernelsMigration(unittest.TestCase):
    """Verify unmodified transformer_lm runs under custom kernels (`aggressive`) + checkpoint cross-load."""

    def test_transformer_lm_aggressive_kernels_and_checkpoint_cross_load(self) -> None:
        script = str(WORKLOADS_DIR / "transformer_lm.py")
        common = ["--batch-size", "4", "--seed", "42", "--device", _TARGET_DEVICE]
        base = _run_summary([sys.executable, script, "--steps", str(_STEPS), *common])

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "aggressive_ckpt.pt"
            wrap_eager_k = _run_summary([
                sys.executable,
                "-m",
                "ptopt.cli.main",
                "--profile",
                "aggressive",
                "--no-compile",
                "--no-dataloader-tune",
                "--override",
                "kernels.enabled=true",
                "--",
                sys.executable,
                script,
                "--steps",
                str(_STEPS),
                "--save-ckpt",
                str(ckpt),
                *common,
            ])
            parity = check_parity(base["losses"], wrap_eager_k["losses"], profile="aggressive")
            self.assertTrue(parity.passed, f"transformer_lm aggressive(kernels) parity failed: {parity.reason}")
            self.assertTrue(ckpt.is_file())

            # Baseline PyTorch must load checkpoint produced under aggressive custom-kernel run cleanly
            base_resume = _run_summary([
                sys.executable,
                script,
                "--steps",
                "3",
                "--load-ckpt",
                str(ckpt),
                *common,
            ])
            self.assertEqual(len(base_resume["losses"]), 3)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in base_resume["losses"]))

            # Also verify full aggressive profile (with compile + kernels together) does not crash
            wrap_full = _run_summary([
                sys.executable,
                "-m",
                "ptopt.cli.main",
                "--profile",
                "aggressive",
                "--no-dataloader-tune",
                "--",
                sys.executable,
                script,
                "--steps",
                "4",
                *common,
            ])
            parity_full = check_parity(base["losses"][:4], wrap_full["losses"], profile="aggressive")
            self.assertTrue(parity_full.passed, f"transformer_lm full aggressive parity failed: {parity_full.reason}")


if __name__ == "__main__":
    unittest.main()
