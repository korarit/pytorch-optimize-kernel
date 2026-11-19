"""Migrate tests for Phase 3.5: Compiled checkpoint cross-load and dynamic-shape safety."""

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
_STEPS = 8 if _TARGET_DEVICE == "cuda" else 4


def _run_summary(cmd: list[str], env: dict[str, str] | None = None) -> dict[str, object]:
    res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True)
    for line in reversed(res.stdout.splitlines()):
        if line.startswith("WORKLOAD_SUMMARY_JSON: "):
            return json.loads(line.split("WORKLOAD_SUMMARY_JSON: ", 1)[1])
    raise AssertionError(f"Missing WORKLOAD_SUMMARY_JSON from {' '.join(cmd)}")


class TestPhase35CompileMigration(unittest.TestCase):
    """Verify compiled execution preserves checkpoint cross-compatibility and dynamic shape stability."""

    def test_compile_checkpoint_cross_load_and_keys_match(self) -> None:
        script = str(WORKLOADS_DIR / "transformer_lm.py")
        common = ["--batch-size", "4", "--seed", "99", "--device", _TARGET_DEVICE]
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "inductor_cache"
            ckpt_wrapped = Path(tmp) / "compiled_ckpt.pt"
            ckpt_base = Path(tmp) / "base_ckpt.pt"
            env_compile = {**os.environ, "PTOPT_COMPILE_CPU": "1"}

            wrap_res = _run_summary(
                [
                    sys.executable,
                    "-m",
                    "ptopt.cli.main",
                    "--profile",
                    "balanced",
                    "--no-dataloader-tune",
                    "--override",
                    "compile.enabled=true",
                    "--override",
                    f"compile.cache_dir={cache_dir}",
                    "--",
                    sys.executable,
                    script,
                    "--steps",
                    "3",
                    "--save-ckpt",
                    str(ckpt_wrapped),
                    *common,
                ],
                env=env_compile,
            )
            self.assertEqual(len(wrap_res["losses"]), 3)
            self.assertTrue(ckpt_wrapped.is_file())

            # Inspect keys directly: must NOT contain `_orig_mod.` prefix
            ckpt_obj = torch.load(ckpt_wrapped, map_location="cpu", weights_only=False)
            model_keys = list(ckpt_obj["model_state_dict"].keys())
            self.assertGreater(len(model_keys), 0)
            self.assertFalse(any("_orig_mod." in k for k in model_keys), f"Unexpected _orig_mod. in {model_keys}")

            # 1) Compiled wrapped save -> Unwrapped baseline PyTorch load/resume
            base_resume = _run_summary([
                sys.executable,
                script,
                "--steps",
                "3",
                "--load-ckpt",
                str(ckpt_wrapped),
                "--save-ckpt",
                str(ckpt_base),
                *common,
            ])
            self.assertEqual(len(base_resume["losses"]), 3)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in base_resume["losses"]))

            # 2) Baseline save -> Compiled wrapped load/resume
            wrap_resume = _run_summary(
                [
                    sys.executable,
                    "-m",
                    "ptopt.cli.main",
                    "--profile",
                    "balanced",
                    "--no-dataloader-tune",
                    "--override",
                    "compile.enabled=true",
                    "--override",
                    f"compile.cache_dir={cache_dir}",
                    "--",
                    sys.executable,
                    script,
                    "--steps",
                    "3",
                    "--load-ckpt",
                    str(ckpt_base),
                    *common,
                ],
                env=env_compile,
            )
            self.assertEqual(len(wrap_resume["losses"]), 3)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in wrap_resume["losses"]))

    def test_dynamic_shapes_edge_case_under_compile(self) -> None:
        script = str(WORKLOADS_DIR / "edge_cases.py")
        args = ["--case", "dynamic_shape", "--steps", str(_STEPS), "--batch-size", "8", "--seed", "42", "--device", _TARGET_DEVICE]
        base = _run_summary([sys.executable, script, *args])
        with tempfile.TemporaryDirectory() as tmp:
            env_compile = {**os.environ, "PTOPT_COMPILE_CPU": "1"}
            wrap = _run_summary(
                [
                    sys.executable,
                    "-m",
                    "ptopt.cli.main",
                    "--profile",
                    "balanced",
                    "--no-dataloader-tune",
                    "--override",
                    "compile.enabled=true",
                    "--override",
                    f"compile.cache_dir={Path(tmp) / 'inductor_cache'}",
                    "--",
                    sys.executable,
                    script,
                    *args,
                ],
                env=env_compile,
            )
        parity = check_parity(base["losses"], wrap["losses"], profile="balanced")
        self.assertTrue(parity.passed, f"dynamic_shape under compile failed parity: {parity.reason}")


if __name__ == "__main__":
    unittest.main()
