"""Phase 4.1 Axis 2: Full 5x5 Checkpoint & Optimizer/Scaler State Cross-Load Matrix."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODES = ("baseline", "supersafe", "safe", "balanced", "aggressive")

_CKPT_WORKER = textwrap.dedent(
    """
    import argparse
    import json
    import warnings
    from pathlib import Path
    import torch
    import torch.nn as nn
    import torch.optim as optim

    def _build_model(device: str):
        torch.manual_seed(123)
        return nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 4),
        ).to(device)

    def _inspect_and_save(save_path: Path, device: str):
        model = _build_model(device)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

        losses = []
        for _ in range(3):
            x = torch.randn(4, 3, 12, 12, device=device)
            y = torch.randint(0, 4, (4,), device=device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().item()))

        model_sd = model.state_dict()
        opt_sd = opt.state_dict()
        scaler_sd = scaler.state_dict()

        payload = {
            "model": model_sd,
            "optimizer": opt_sd,
            "scaler": scaler_sd,
            "keys": list(model_sd.keys()),
            "shapes": {k: list(v.shape) for k, v in model_sd.items()},
            "losses": losses,
        }
        torch.save(payload, save_path)
        print("CKPT_MATRIX_JSON: " + json.dumps({
            "action": "save",
            "keys": payload["keys"],
            "has_orig_mod": any("_orig_mod." in k for k in payload["keys"]),
            "has_fused_flag": any(bool(g.get("fused")) for g in opt_sd.get("param_groups", [])),
            "last_loss": losses[-1],
        }))

    def _load_and_resume(ckpt_paths: list[Path], device: str):
        results = []
        for ckpt_path in ckpt_paths:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = _build_model(device)
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])

            x = torch.randn(4, 3, 12, 12, device=device)
            y = torch.randint(0, 4, (4,), device=device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loaded_sd = model.state_dict()
            resumed_val = float(loss.detach().item())
            results.append({
                "source": ckpt_path.stem,
                "resumed_finite": bool(torch.isfinite(loss).item()),
                "resumed_loss": resumed_val,
                "key_match": list(loaded_sd.keys()) == ckpt["keys"],
                "shape_match": all(list(loaded_sd[k].shape) == ckpt["shapes"][k] for k in ckpt["keys"]),
                "contiguous_4d_ok": all(
                    v.is_contiguous(memory_format=torch.contiguous_format)
                    for v in loaded_sd.values()
                    if v.ndim == 4
                ),
            })
        print("CKPT_MATRIX_JSON: " + json.dumps({"action": "load", "results": results}))

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--action", choices=["save", "load"], required=True)
        ap.add_argument("--device", default="cpu")
        ap.add_argument("--ckpt", nargs="+", required=True)
        args = ap.parse_args()
        if args.action == "save":
            _inspect_and_save(Path(args.ckpt[0]), args.device)
        else:
            _load_and_resume([Path(p) for p in args.ckpt], args.device)

    if __name__ == "__main__":
        main()
    """
)


def _run_ckpt_worker(script: Path, mode: str, action: str, ckpts: list[Path]) -> dict[str, object]:
    worker_cmd = [
        sys.executable,
        str(script),
        "--action",
        action,
        "--device",
        _TARGET_DEVICE,
        "--ckpt",
        *[str(p) for p in ckpts],
    ]
    if mode == "baseline":
        cmd = worker_cmd
    else:
        cmd = [sys.executable, "-m", "ptopt.cli.main", "--profile", mode, "--", *worker_cmd]

    env = dict(os.environ)
    # Exercise 4D channels_last state_dict contiguous normalization on CPU as well as GPU.
    env["PTOPT_CHANNELS_LAST"] = "1"
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(
            f"Checkpoint worker failed (mode={mode}, action={action}, exit={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("CKPT_MATRIX_JSON: "):
            return json.loads(line.split("CKPT_MATRIX_JSON: ", 1)[1])
    raise AssertionError(f"Missing CKPT_MATRIX_JSON output for mode={mode}, action={action}")


class TestCheckpointMigrationMatrix(unittest.TestCase):
    """Axis 2 Migrate Matrix: verify full 5x5 checkpoint cross-loading across all profiles."""

    def test_full_5x5_checkpoint_optimizer_scaler_cross_load_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            script = tmp_dir / "ckpt_matrix_worker.py"
            script.write_text(_CKPT_WORKER, encoding="utf-8")

            saved_paths: list[Path] = []
            for save_mode in _MODES:
                ckpt_path = tmp_dir / f"{save_mode}.pt"
                info = _run_ckpt_worker(script, save_mode, "save", [ckpt_path])
                self.assertFalse(bool(info["has_orig_mod"]), f"{save_mode} leaked _orig_mod prefix in state_dict")
                self.assertFalse(bool(info["has_fused_flag"]), f"{save_mode} leaked fused flag in optimizer.state_dict")
                saved_paths.append(ckpt_path)

            for load_mode in _MODES:
                load_out = _run_ckpt_worker(script, load_mode, "load", saved_paths)
                for row in load_out["results"]:
                    src_mode = row["source"]
                    with self.subTest(save_mode=src_mode, load_mode=load_mode, device=_TARGET_DEVICE):
                        self.assertTrue(row["resumed_finite"], f"{src_mode}->{load_mode} produced non-finite loss")
                        self.assertTrue(row["key_match"], f"{src_mode}->{load_mode} key mismatch")
                        self.assertTrue(row["shape_match"], f"{src_mode}->{load_mode} shape mismatch")
                        self.assertTrue(
                            row["contiguous_4d_ok"],
                            f"{src_mode}->{load_mode} saved non-contiguous 4D weights",
                        )


if __name__ == "__main__":
    unittest.main()
