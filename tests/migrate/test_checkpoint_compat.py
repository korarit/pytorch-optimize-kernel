"""Migrate Test Group 2: Checkpoint cross-compatibility between Baseline and ptopt-wrapped runs."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKLOADS_DIR = REPO_ROOT / "pipline-for-test" / "workloads"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptopt.hooks.checkpoint_compat import (
    install_checkpoint_compat_hooks,
    strip_orig_mod_prefix,
    uninstall_checkpoint_compat_hooks,
)

_TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _run_summary(cmd: list[str]) -> dict[str, object]:
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    for line in reversed(res.stdout.splitlines()):
        if line.startswith("WORKLOAD_SUMMARY_JSON: "):
            return json.loads(line.split("WORKLOAD_SUMMARY_JSON: ", 1)[1])
    raise AssertionError(f"Missing WORKLOAD_SUMMARY_JSON from {' '.join(cmd)}")


class _FakeCompiledModule(nn.Module):
    """Simulates `torch._dynamo.OptimizedModule` storing wrapped model under `_orig_mod`."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self._orig_mod = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._orig_mod(x)


class TestCheckpointCrossCompatibility(unittest.TestCase):
    """Verify checkpoints save/load seamlessly between Baseline and Wrapped PyTorch."""

    def test_baseline_save_wrapped_resume_and_wrapped_save_baseline_resume(self) -> None:
        script = str(WORKLOADS_DIR / "transformer_lm.py")
        common_args = ["--batch-size", "4", "--seed", "99", "--device", _TARGET_DEVICE]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ckpt_base = tmp_dir / "from_baseline.pt"
            ckpt_wrap = tmp_dir / "from_wrapped.pt"

            # Reference: 10 continuous baseline steps
            ref_10 = _run_summary([sys.executable, script, "--steps", "10", *common_args])

            # Case A: Baseline 5 steps -> save -> Wrapped resume 5 steps
            _run_summary([sys.executable, script, "--steps", "5", "--save-ckpt", str(ckpt_base), *common_args])
            resumed_wrap = _run_summary([
                sys.executable,
                "-m",
                "ptopt.cli.main",
                "--profile",
                "safe",
                "--",
                sys.executable,
                script,
                "--steps",
                "5",
                "--load-ckpt",
                str(ckpt_base),
                *common_args,
            ])
            self.assertEqual(len(resumed_wrap["losses"]), 5)
            for val in resumed_wrap["losses"]:
                self.assertTrue(torch.isfinite(torch.tensor(val)))
            self.assertEqual(len(ref_10["losses"]), 10)

            # Case B: Wrapped 5 steps -> save -> Unwrapped Baseline resume 5 steps
            _run_summary([
                sys.executable,
                "-m",
                "ptopt.cli.main",
                "--profile",
                "safe",
                "--",
                sys.executable,
                script,
                "--steps",
                "5",
                "--save-ckpt",
                str(ckpt_wrap),
                *common_args,
            ])

            payload = torch.load(ckpt_wrap, map_location="cpu", weights_only=False)
            model_keys = list(payload["model_state_dict"].keys())
            self.assertTrue(model_keys)
            self.assertFalse(
                any("_orig_mod." in k for k in model_keys),
                f"Unexpected _orig_mod. prefix in saved keys: {model_keys[:5]}",
            )

            resumed_base = _run_summary([
                sys.executable,
                script,
                "--steps",
                "5",
                "--load-ckpt",
                str(ckpt_wrap),
                *common_args,
            ])
            self.assertEqual(len(resumed_base["losses"]), 5)
            for a, b in zip(resumed_wrap["losses"], resumed_base["losses"]):
                self.assertAlmostEqual(float(a), float(b), delta=5e-3)

    def test_orig_mod_prefix_transparency_hooks(self) -> None:
        torch.manual_seed(7)
        plain = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        compiled_like = _FakeCompiledModule(nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)))

        install_checkpoint_compat_hooks(torch)
        try:
            # 1. Saving from compiled_like emits clean keys without `_orig_mod.`
            clean_sd = compiled_like.state_dict()
            self.assertEqual(list(clean_sd.keys()), list(plain.state_dict().keys()))

            # 2. Clean state_dict from plain loads directly into compiled_like
            plain_sd = plain.state_dict()
            compiled_like.load_state_dict(plain_sd)
            for key, ref_tensor in plain_sd.items():
                loaded_tensor = compiled_like.state_dict()[key]
                self.assertEqual(loaded_tensor.shape, ref_tensor.shape)
                self.assertEqual(loaded_tensor.dtype, ref_tensor.dtype)
                self.assertTrue(torch.equal(loaded_tensor, ref_tensor))

            # 3. Legacy `_orig_mod.` state_dict loads cleanly into plain unwrapped model
            legacy_sd = {f"_orig_mod.{k}": v.clone() for k, v in plain_sd.items()}
            plain.load_state_dict(legacy_sd)
            self.assertEqual(list(strip_orig_mod_prefix(legacy_sd).keys()), list(plain_sd.keys()))
        finally:
            uninstall_checkpoint_compat_hooks(torch)


if __name__ == "__main__":
    unittest.main()
