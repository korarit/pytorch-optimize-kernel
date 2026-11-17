"""Phase 4.1 Axis 1: Full Code Migration Matrix across 5 real-world PyTorch coding patterns."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "pipline-for-test"
for _p in (str(REPO_ROOT), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from telemetry import check_parity

_TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Unmodified PyTorch script exercising 5 distinct legacy/complex user code patterns.
_MATRIX_SCRIPT = textwrap.dedent(
    """
    import argparse
    import json
    import warnings
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class _ScaleShiftFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.save_for_backward(alpha)
            return x * alpha

        @staticmethod
        def backward(ctx, grad_out):
            (alpha,) = ctx.saved_tensors
            return grad_out * alpha, None

    class _DynamicBranchNet(nn.Module):
        def __init__(self, dim: int = 32):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
            self.branch_a = nn.Linear(dim, dim)
            self.branch_b = nn.Linear(dim, dim)
            self.head = nn.Linear(dim, 4)
            self.register_buffer("alpha", torch.tensor(1.05))

        def forward(self, x):
            h = self.trunk(x)
            h = _ScaleShiftFn.apply(h, self.alpha)
            if float(h.detach().mean().item()) >= 0.0:
                h = h + torch.tanh(self.branch_a(h))
            else:
                h = h + torch.relu(self.branch_b(h))
            return self.head(h)

    def _run_legacy_amp(device: str, steps: int):
        torch.manual_seed(11)
        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4)).to(device)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        use_cuda_amp = device == "cuda"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)
        losses = []
        for step in range(steps):
            x = torch.randn(8, 32, device=device)
            y = torch.randint(0, 4, (8,), device=device)
            opt.zero_grad(set_to_none=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with torch.cuda.amp.autocast(enabled=use_cuda_amp):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().item()))
        return {"losses": losses}

    def _run_grad_accum(device: str, steps: int):
        torch.manual_seed(22)
        model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4)).to(device)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        accum_steps = 3
        losses = []
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _micro in range(accum_steps):
                x = torch.randn(4, 3, 12, 12, device=device)
                y = torch.randint(0, 4, (4,), device=device)
                loss = nn.functional.cross_entropy(model(x), y) / accum_steps
                loss.backward()
                accum_loss += float(loss.detach().item())
            opt.step()
            losses.append(accum_loss)
        return {"losses": losses}

    def _run_multi_opt_scheduler(device: str, steps: int):
        torch.manual_seed(33)
        backbone = nn.Sequential(nn.Linear(32, 32), nn.ReLU()).to(device)
        head = nn.Linear(32, 4).to(device)
        opt_adam = optim.AdamW(backbone.parameters(), lr=2e-3)
        opt_sgd = optim.SGD(head.parameters(), lr=1e-2, momentum=0.9)
        sched_cos = optim.lr_scheduler.CosineAnnealingLR(opt_adam, T_max=max(steps, 1))
        sched_one = optim.lr_scheduler.OneCycleLR(opt_sgd, max_lr=2e-2, total_steps=max(steps, 1))
        losses = []
        for _ in range(steps):
            x = torch.randn(8, 32, device=device)
            y = torch.randint(0, 4, (8,), device=device)
            opt_adam.zero_grad(set_to_none=True)
            opt_sgd.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(backbone(x)), y)
            loss.backward()
            opt_adam.step()
            opt_sgd.step()
            sched_cos.step()
            sched_one.step()
            losses.append(float(loss.detach().item()))
        return {"losses": losses, "lrs": [float(opt_adam.param_groups[0]["lr"]), float(opt_sgd.param_groups[0]["lr"])]}

    def _run_custom_autograd_dynamic(device: str, steps: int):
        torch.manual_seed(44)
        model = _DynamicBranchNet(32).to(device)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        for s in range(steps):
            sign = 1.0 if (s % 2 == 0) else -1.0
            x = torch.full((6, 32), 0.25 * sign, device=device) + 0.05 * torch.randn(6, 32, device=device)
            y = torch.randint(0, 4, (6,), device=device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().item()))
        return {"losses": losses}

    def _run_train_eval_interleaved(device: str, steps: int):
        torch.manual_seed(55)
        model = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32), nn.Dropout(0.1), nn.ReLU(), nn.Linear(32, 4)).to(device)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        eval_losses = []
        for _ in range(steps):
            model.train()
            x_tr = torch.randn(8, 32, device=device)
            y_tr = torch.randint(0, 4, (8,), device=device)
            opt.zero_grad(set_to_none=True)
            tr_loss = nn.functional.cross_entropy(model(x_tr), y_tr)
            tr_loss.backward()
            opt.step()
            losses.append(float(tr_loss.detach().item()))

            model.eval()
            with torch.no_grad():
                x_ev = torch.randn(4, 32, device=device)
                y_ev = torch.randint(0, 4, (4,), device=device)
                ev_loss = nn.functional.cross_entropy(model(x_ev), y_ev)
                eval_losses.append(float(ev_loss.detach().item()))
        return {"losses": losses, "eval_losses": eval_losses, "final_training_flag": bool(model.training)}

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--style", required=True)
        ap.add_argument("--device", default="cpu")
        ap.add_argument("--steps", type=int, default=6)
        args = ap.parse_args()
        runners = {
            "legacy_amp": _run_legacy_amp,
            "grad_accum": _run_grad_accum,
            "multi_opt_scheduler": _run_multi_opt_scheduler,
            "custom_autograd_dynamic": _run_custom_autograd_dynamic,
            "train_eval_interleaved": _run_train_eval_interleaved,
        }
        payload = runners[args.style](args.device, args.steps)
        print("CODE_MATRIX_JSON: " + json.dumps(payload))

    if __name__ == "__main__":
        main()
    """
)

_STYLES = (
    ("legacy_amp", "balanced"),
    ("grad_accum", "balanced"),
    ("multi_opt_scheduler", "balanced"),
    ("custom_autograd_dynamic", "aggressive"),
    ("train_eval_interleaved", "aggressive"),
)


def _run_style(script_path: Path, style: str, profile: str | None) -> dict[str, object]:
    script_cmd = [sys.executable, str(script_path), "--style", style, "--device", _TARGET_DEVICE, "--steps", "6"]
    if profile is None:
        cmd = script_cmd
    else:
        cmd = [sys.executable, "-m", "src.cli.main", "--profile", profile, "--", *script_cmd]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(
            f"Style={style} profile={profile} failed (exit={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("CODE_MATRIX_JSON: "):
            return json.loads(line.split("CODE_MATRIX_JSON: ", 1)[1])
    raise AssertionError(f"Missing CODE_MATRIX_JSON for style={style} profile={profile}")


class TestCodeMigrationMatrix(unittest.TestCase):
    """Axis 1 Migrate Matrix: verify 5 real-world PyTorch script styles run cleanly under ptopt."""

    def test_code_migration_matrix_parity_and_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "matrix_user_script.py"
            script_path.write_text(_MATRIX_SCRIPT, encoding="utf-8")

            for style, profile in _STYLES:
                with self.subTest(style=style, profile=profile, device=_TARGET_DEVICE):
                    base = _run_style(script_path, style, None)
                    wrap = _run_style(script_path, style, profile)

                    parity = check_parity(base["losses"], wrap["losses"], profile=profile)
                    self.assertTrue(
                        parity.passed,
                        f"[{style} / {profile} / {_TARGET_DEVICE}] {parity.reason}",
                    )
                    if style == "multi_opt_scheduler":
                        self.assertEqual(len(base["lrs"]), 2)
                        self.assertAlmostEqual(float(base["lrs"][0]), float(wrap["lrs"][0]), places=5)
                        self.assertAlmostEqual(float(base["lrs"][1]), float(wrap["lrs"][1]), places=5)
                    if style == "train_eval_interleaved":
                        ev_parity = check_parity(base["eval_losses"], wrap["eval_losses"], profile=profile)
                        self.assertTrue(
                            ev_parity.passed,
                            f"[{style} eval / {profile}] {ev_parity.reason}",
                        )
                        self.assertFalse(bool(wrap["final_training_flag"]))


if __name__ == "__main__":
    unittest.main()
