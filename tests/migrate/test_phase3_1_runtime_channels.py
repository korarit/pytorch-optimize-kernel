"""Migrate tests for Phase 3.1: cnn_vision & transformer_lm under runtime flags + channels_last."""

from __future__ import annotations

import json
import os
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
_STEPS = 6 if _TARGET_DEVICE == "cuda" else 4


def _run_summary(cmd: list[str], env: dict[str, str] | None = None) -> dict[str, object]:
    res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True)
    for line in reversed(res.stdout.splitlines()):
        if line.startswith("WORKLOAD_SUMMARY_JSON: "):
            return json.loads(line.split("WORKLOAD_SUMMARY_JSON: ", 1)[1])
    raise AssertionError(f"Missing WORKLOAD_SUMMARY_JSON from {' '.join(cmd)}")


class TestPhase31Migration(unittest.TestCase):
    """Verify Phase 3.1 runtime flags + channels_last preserve parity and checkpoint compatibility."""

    def test_cnn_and_transformer_with_runtime_flags_and_channels_last(self) -> None:
        env_cl = {**os.environ, "PTOPT_CHANNELS_LAST": "1"}
        for script_name, bs in (("cnn_vision.py", "8"), ("transformer_lm.py", "4")):
            script = str(WORKLOADS_DIR / script_name)
            args = ["--steps", str(_STEPS), "--batch-size", bs, "--seed", "55", "--device", _TARGET_DEVICE]
            base = _run_summary([sys.executable, script, *args])
            wrap = _run_summary(
                [sys.executable, "-m", "src.cli.main", "--profile", "safe", "--", sys.executable, script, *args],
                env=env_cl,
            )
            parity = check_parity(base["losses"], wrap["losses"], profile="safe")
            self.assertTrue(parity.passed, f"{script_name} parity failed: {parity.reason}")

    def test_cnn_channels_last_checkpoint_cross_load(self) -> None:
        script = str(WORKLOADS_DIR / "cnn_vision.py")
        args = ["--batch-size", "8", "--seed", "55", "--device", _TARGET_DEVICE]
        env_cl = {**os.environ, "PTOPT_CHANNELS_LAST": "1"}

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "cnn_cl.pt"
            wrap_train = _run_summary(
                [
                    sys.executable,
                    "-m",
                    "src.cli.main",
                    "--profile",
                    "safe",
                    "--",
                    sys.executable,
                    script,
                    "--steps",
                    "4",
                    "--save-ckpt",
                    str(ckpt),
                    *args,
                ],
                env=env_cl,
            )
            self.assertEqual(len(wrap_train["losses"]), 4)

            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            for k, tensor in payload["model_state_dict"].items():
                if hasattr(tensor, "dim") and tensor.dim() == 4:
                    self.assertTrue(tensor.is_contiguous(), f"Expected contiguous 4D tensor in saved ckpt for {k}")

            base_resume = _run_summary([
                sys.executable,
                script,
                "--steps",
                "4",
                "--load-ckpt",
                str(ckpt),
                *args,
            ])
            self.assertEqual(len(base_resume["losses"]), 4)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in base_resume["losses"]))


if __name__ == "__main__":
    unittest.main()
