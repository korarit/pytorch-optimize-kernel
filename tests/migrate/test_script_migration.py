"""Migrate Test Group 1: Run unmodified PyTorch workloads through `ptopt` and verify parity."""

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
_STEPS = 6 if _TARGET_DEVICE == "cuda" else 4

_WORKLOADS = (
    ("cnn_vision.py", ["--batch-size", "8"]),
    ("transformer_lm.py", ["--batch-size", "4"]),
    ("deep_mlp.py", ["--batch-size", "16"]),
    ("infer_runner.py", ["--batch-size", "1"]),
    ("edge_cases.py", ["--batch-size", "4", "--case", "all"]),
)


def _run_and_parse(cmd: list[str]) -> dict[str, object]:
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise AssertionError(
            f"Command failed (exit={res.returncode}): {' '.join(cmd)}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    for line in reversed(res.stdout.splitlines()):
        if line.startswith("WORKLOAD_SUMMARY_JSON: "):
            return json.loads(line.split("WORKLOAD_SUMMARY_JSON: ", 1)[1])
    raise AssertionError(f"Missing WORKLOAD_SUMMARY_JSON from command: {' '.join(cmd)}")


class TestUnmodifiedScriptMigration(unittest.TestCase):
    """Verify unmodified user scripts execute cleanly under `ptopt` and preserve parity."""

    def test_unmodified_workloads_under_ptopt_safe(self) -> None:
        for script_name, extra in _WORKLOADS:
            script_path = WORKLOADS_DIR / script_name
            base_args = [
                "--steps",
                str(_STEPS),
                "--seed",
                "42",
                "--device",
                _TARGET_DEVICE,
                *extra,
            ]
            base_summary = _run_and_parse([sys.executable, str(script_path), *base_args])
            wrap_summary = _run_and_parse([
                sys.executable,
                "-m",
                "src.cli.main",
                "--profile",
                "safe",
                "--",
                sys.executable,
                str(script_path),
                *base_args,
            ])

            parity = check_parity(base_summary["losses"], wrap_summary["losses"], profile="safe")
            self.assertTrue(
                parity.passed,
                f"Parity failed for {script_name} ({_TARGET_DEVICE}): {parity.reason}",
            )


if __name__ == "__main__":
    unittest.main()
