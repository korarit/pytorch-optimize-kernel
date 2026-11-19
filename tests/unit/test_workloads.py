"""Unit & smoke tests for Phase 2.2: Pure PyTorch workloads, determinism, and checkpoint round-trip."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKLOADS_DIR = REPO_ROOT / "pipline-for-test" / "workloads"

WORKLOAD_FILES = (
    "cnn_vision.py",
    "transformer_lm.py",
    "deep_mlp.py",
    "infer_runner.py",
    "edge_cases.py",
)


def _run_workload(script_name: str, extra_args: list[str]) -> dict[str, object]:
    cmd = [sys.executable, str(WORKLOADS_DIR / script_name), *extra_args]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    for line in reversed(res.stdout.splitlines()):
        if line.startswith("WORKLOAD_SUMMARY_JSON: "):
            return json.loads(line.split("WORKLOAD_SUMMARY_JSON: ", 1)[1])
    raise AssertionError(f"Missing WORKLOAD_SUMMARY_JSON in output of {script_name}:\n{res.stdout}")


class TestWorkloadsPureAndDeterministic(unittest.TestCase):
    """Verify workloads stay Pure PyTorch, run deterministically, and support checkpoints."""

    def test_workloads_do_not_import_wrapper(self) -> None:
        for path in WORKLOADS_DIR.glob("*.py"):
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("from ptopt", content, f"{path.name} must remain pure PyTorch")
            self.assertNotIn("import ptopt", content, f"{path.name} must remain pure PyTorch")
            self.assertNotIn("from src", content, f"{path.name} must remain pure PyTorch")
            self.assertNotIn("import src", content, f"{path.name} must remain pure PyTorch")

    def test_all_workloads_cpu_smoke_and_determinism(self) -> None:
        for script in WORKLOAD_FILES:
            args = ["--steps", "5", "--batch-size", "4", "--seed", "123", "--device", "cpu"]
            run1 = _run_workload(script, args)
            run2 = _run_workload(script, args)
            losses1 = run1["losses"]
            losses2 = run2["losses"]
            self.assertEqual(len(losses1), len(losses2))
            self.assertGreater(len(losses1), 0)
            for idx, (v1, v2) in enumerate(zip(losses1, losses2)):
                self.assertAlmostEqual(
                    float(v1),
                    float(v2),
                    delta=1e-4,
                    msg=f"Non-deterministic loss at step {idx} in {script}: {v1} vs {v2}",
                )

    def test_checkpoint_save_and_resume_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "tiny_gpt.pt"
            run_a = _run_workload(
                "transformer_lm.py",
                ["--steps", "3", "--batch-size", "4", "--seed", "77", "--device", "cpu", "--save-ckpt", str(ckpt)],
            )
            self.assertTrue(ckpt.is_file())
            run_b = _run_workload(
                "transformer_lm.py",
                ["--steps", "3", "--batch-size", "4", "--seed", "77", "--device", "cpu", "--load-ckpt", str(ckpt)],
            )
            self.assertEqual(len(run_a["losses"]), 3)
            self.assertEqual(len(run_b["losses"]), 3)
            # Resumed model starts from trained weights, so step losses differ from fresh init.
            self.assertNotEqual(run_a["losses"], run_b["losses"])


if __name__ == "__main__":
    unittest.main()
