"""Migrate tests for Phase 5.4: Static-Shape Training CUDA Graphs parity & checkpoint save/load."""

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

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_STEPS = 12 if _DEVICE == "cuda" else 5


def _run_summary(cmd: list[str]) -> dict[str, object]:
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    for line in reversed(res.stdout.splitlines()):
        if line.startswith("WORKLOAD_SUMMARY_JSON: "):
            return json.loads(line.split("WORKLOAD_SUMMARY_JSON: ", 1)[1])
    raise AssertionError(f"Missing WORKLOAD_SUMMARY_JSON from {' '.join(cmd)}")


class TestPhase54TrainCudaGraphsMigration(unittest.TestCase):
    """Verify training CUDA Graphs preserves parity and checkpoint save/load roundtrip."""

    def test_deep_mlp_train_cuda_graphs_parity(self) -> None:
        script = str(WORKLOADS_DIR / "deep_mlp.py")
        extra = ["--width", "256", "--depth", "24"] if _DEVICE == "cuda" else ["--width", "64", "--depth", "6"]
        bs = "64" if _DEVICE == "cuda" else "16"
        common = ["--steps", str(_STEPS), "--batch-size", bs, "--seed", "42", "--device", _DEVICE, *extra]

        base = _run_summary([sys.executable, script, *common])
        wrap = _run_summary([
            sys.executable,
            "-m",
            "src.cli.main",
            "--profile",
            "balanced",
            "--override",
            "compile.train_cuda_graphs=true",
            "--",
            sys.executable,
            script,
            *common,
        ])
        parity = check_parity(base["losses"], wrap["losses"], profile="balanced")
        self.assertTrue(parity.passed, f"deep_mlp train_cuda_graphs parity failed: {parity.reason}")

    def test_transformer_lm_small_train_cuda_graphs_and_checkpoint_roundtrip(self) -> None:
        script = str(WORKLOADS_DIR / "transformer_lm.py")
        extra = (
            ["--vocab-size", "2048", "--d-model", "256", "--n-heads", "4", "--n-layers", "4", "--seq-len", "128"]
            if _DEVICE == "cuda"
            else ["--vocab-size", "512", "--d-model", "64", "--n-heads", "2", "--n-layers", "2", "--seq-len", "32"]
        )
        bs = "16" if _DEVICE == "cuda" else "4"
        common = ["--steps", str(_STEPS), "--batch-size", bs, "--seed", "42", "--device", _DEVICE, *extra]

        base = _run_summary([sys.executable, script, *common])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = str(Path(tmp) / "cg_ckpt.pt")
            half = max(3, _STEPS // 2)
            part1 = _run_summary([
                sys.executable,
                "-m",
                "src.cli.main",
                "--profile",
                "balanced",
                "--override",
                "compile.train_cuda_graphs=true",
                "--",
                sys.executable,
                script,
                "--steps",
                str(half),
                "--batch-size",
                bs,
                "--seed",
                "42",
                "--device",
                _DEVICE,
                "--save-ckpt",
                ckpt_path,
                *extra,
            ])
            part2 = _run_summary([
                sys.executable,
                "-m",
                "src.cli.main",
                "--profile",
                "balanced",
                "--override",
                "compile.train_cuda_graphs=true",
                "--",
                sys.executable,
                script,
                "--steps",
                str(_STEPS - half),
                "--batch-size",
                bs,
                "--seed",
                "42",
                "--device",
                _DEVICE,
                "--load-ckpt",
                ckpt_path,
                *extra,
            ])
            combined_losses = list(part1["losses"]) + list(part2["losses"])  # type: ignore[arg-type]
        parity = check_parity(base["losses"], combined_losses, profile="balanced")
        self.assertTrue(parity.passed, f"transformer_lm train_cuda_graphs ckpt roundtrip failed: {parity.reason}")


if __name__ == "__main__":
    unittest.main()
