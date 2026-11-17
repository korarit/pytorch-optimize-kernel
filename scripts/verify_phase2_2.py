#!/usr/bin/env python3
"""Phase 2.2 verification script: smoke-runs all 5 workloads on live device + runs workload unit tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.text_report import TextReportBuilder
from src.detect import collect_fingerprint

WORKLOADS_DIR = REPO_ROOT / "pipline-for-test" / "workloads"


def _run_unit_tests(repo_root: Path) -> tuple[str, int, str]:
    test_file = repo_root / "tests" / "unit" / "test_workloads.py"
    if importlib.util.find_spec("pytest") is not None:
        cmd = [sys.executable, "-m", "pytest", str(test_file), "-v"]
        runner = "pytest"
    else:
        cmd = [sys.executable, "-m", "unittest", "tests/unit/test_workloads.py", "-v"]
        runner = "unittest"
    res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    return runner, res.returncode, (res.stdout + "\n" + res.stderr).strip()


def _run_live_workload(script_name: str, extra_args: list[str]) -> tuple[int, dict[str, object]]:
    cmd = [sys.executable, str(WORKLOADS_DIR / script_name), *extra_args]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        return res.returncode, {"workload": script_name, "error": res.stderr.strip() or res.stdout.strip()}
    for line in reversed(res.stdout.splitlines()):
        if line.startswith("WORKLOAD_SUMMARY_JSON: "):
            return 0, json.loads(line.split("WORKLOAD_SUMMARY_JSON: ", 1)[1])
    return 1, {"workload": script_name, "error": "Missing WORKLOAD_SUMMARY_JSON"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 2.2 Workloads on Baseline PyTorch.")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=15, help="Steps per workload for live baseline run")
    args = parser.parse_args()

    fp = collect_fingerprint()
    target_device = "cuda" if fp.effective_backend == "cuda" else "cpu"
    steps = args.steps if target_device == "cuda" else min(args.steps, 6)

    runner, test_rc, test_output = _run_unit_tests(REPO_ROOT)

    specs = [
        ("cnn_vision.py", ["--steps", str(steps), "--batch-size", "16", "--device", target_device]),
        ("transformer_lm.py", ["--steps", str(steps), "--batch-size", "8", "--device", target_device]),
        ("deep_mlp.py", ["--steps", str(steps), "--batch-size", "32", "--device", target_device]),
        ("infer_runner.py", ["--steps", str(steps), "--batch-size", "1", "--device", target_device]),
        ("infer_runner.py", ["--steps", str(steps), "--batch-size", "32", "--device", target_device]),
        ("edge_cases.py", ["--steps", str(min(steps, 10)), "--batch-size", "8", "--device", target_device]),
    ]

    rows = []
    live_fail = 0
    for script_name, wl_args in specs:
        rc, data = _run_live_workload(script_name, wl_args)
        if rc != 0:
            live_fail += 1
            rows.append([script_name, target_device, "FAIL", "-", "-", "-", data.get("error", "error")])
            continue
        rows.append([
            data["workload"],
            data["device"],
            "PASS",
            data["steps"],
            f"{data['elapsed_sec']}s",
            f"{data['step_ms_avg']}ms",
            f"{data['peak_allocated_mb']} MB | loss={data['final_loss']}",
        ])

    overall_rc = 0 if (test_rc == 0 and live_fail == 0) else 1

    builder = TextReportBuilder(
        phase_title="PHASE 2.2: PURE PYTORCH WORKLOADS BASELINE VERIFICATION",
        repo_root=REPO_ROOT,
    )
    builder.add_header({
        "Google Colab": fp.is_colab,
        "Live Backend": fp.effective_backend,
        "GPU": fp.primary_gpu.name if fp.primary_gpu else "None",
        "Target Device": target_device,
        "Test Runner": f"{runner} (exit_code={test_rc})",
    })
    builder.add_table(
        "BASELINE WORKLOAD RUNS (Unwrapped Pure PyTorch)",
        ["Workload", "Device", "Status", "Steps", "Wall Time", "Avg Step", "VRAM & Final Metric"],
        rows,
    )
    builder.add_lines_section(f"UNIT TEST OUTPUT ({runner})", test_output.splitlines())
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if overall_rc == 0 else "FAILED_NEEDS_ATTENTION",
        note="Verify baseline workload timings and VRAM fit Google Colab runtime before Phase 2.3",
    )

    out_path = builder.write("phase-2.2", report_dir=args.report_dir)
    print(builder.render())
    print(f"[SAVED] Text report written to: {out_path}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
