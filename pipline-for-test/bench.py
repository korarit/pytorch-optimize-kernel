#!/usr/bin/env python3
"""Subprocess A/B benchmark orchestrator for Baseline vs ptopt-wrapped runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
for _p in (str(REPO_ROOT), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from report import build_comparison_rows, export_csv_report, export_json_report, generate_benchmark_text_report
from src.config import ConfigResolver
from src.detect import collect_fingerprint
from telemetry import RunRecord, merge_summary_and_telemetry

WORKLOADS_DIR = BENCH_DIR / "workloads"

WORKLOAD_SPECS: dict[str, dict[str, Any]] = {
    "cnn_vision": {"script": "cnn_vision.py", "batch_size": 16, "extra": []},
    "transformer_lm": {"script": "transformer_lm.py", "batch_size": 8, "extra": []},
    "deep_mlp": {"script": "deep_mlp.py", "batch_size": 32, "extra": []},
    "infer_runner_bs1": {"script": "infer_runner.py", "batch_size": 1, "extra": []},
    "infer_runner_bs32": {"script": "infer_runner.py", "batch_size": 32, "extra": []},
    "edge_cases": {"script": "edge_cases.py", "batch_size": 8, "extra": ["--case", "all"]},
}

MODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "smoke": {"steps": 10, "profiles": ["safe"], "cache_modes": ["warm"]},
    "standard": {"steps": 25, "profiles": ["safe", "balanced"], "cache_modes": ["warm"]},
    "full": {"steps": 40, "profiles": ["safe", "balanced", "aggressive"], "cache_modes": ["cold", "warm"]},
}


def _extract_workload_summary(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith("WORKLOAD_SUMMARY_JSON: "):
            return json.loads(line.split("WORKLOAD_SUMMARY_JSON: ", 1)[1])
    return None


def _run_single_subprocess(
    *,
    workload_name: str,
    script_path: Path,
    script_args: list[str],
    mode: str,
    cache_state: str,
    cache_dir: Path,
) -> RunRecord:
    """Execute one isolated subprocess (either baseline or ptopt-wrapped)."""
    if cache_state == "cold" and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)

    with tempfile.TemporaryDirectory() as tmp:
        tel_path = Path(tmp) / "telemetry.json"
        if mode == "baseline":
            cmd = [sys.executable, str(script_path), *script_args]
        else:
            env["PTOPT_TELEMETRY_OUT"] = str(tel_path)
            cmd = [
                sys.executable,
                "-m",
                "src.cli.main",
                "--profile",
                mode,
                "--",
                sys.executable,
                str(script_path),
                *script_args,
            ]

        t0 = time.perf_counter()
        res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
        wall_sec = time.perf_counter() - t0

        summary = _extract_workload_summary(res.stdout)
        telemetry_data = json.loads(tel_path.read_text(encoding="utf-8")) if tel_path.is_file() else None
        err = None
        if res.returncode != 0:
            err = (res.stderr.strip() or res.stdout.strip() or f"Exited with code {res.returncode}")[-400:]

        return merge_summary_and_telemetry(
            workload=workload_name,
            mode=mode,
            cache_state=cache_state,
            exit_code=res.returncode,
            summary=summary,
            telemetry=telemetry_data,
            wall_sec=wall_sec,
            error=err,
        )


def run_benchmark_suite(
    *,
    workloads: Sequence[str],
    profiles: Sequence[str],
    cache_modes: Sequence[str],
    steps: int,
    device: str,
    seed: int = 42,
    trials: int = 1,
) -> list[RunRecord]:
    """Run baseline and wrapped profiles in isolated subprocesses across all selected workloads."""
    records: list[RunRecord] = []
    with tempfile.TemporaryDirectory() as cache_tmp:
        cache_dir = Path(cache_tmp) / "inductor_cache"
        for wl_name in workloads:
            spec = WORKLOAD_SPECS[wl_name]
            script_path = WORKLOADS_DIR / spec["script"]
            wl_steps = min(steps, 10) if wl_name == "edge_cases" else steps
            script_args = [
                "--steps",
                str(wl_steps),
                "--batch-size",
                str(spec["batch_size"]),
                "--seed",
                str(seed),
                "--device",
                device,
                *spec["extra"],
            ]

            base_trials = [
                _run_single_subprocess(
                    workload_name=wl_name,
                    script_path=script_path,
                    script_args=script_args,
                    mode="baseline",
                    cache_state="none",
                    cache_dir=cache_dir,
                )
                for _ in range(max(1, trials))
            ]
            base_trials.sort(key=lambda r: (r.exit_code != 0, r.end_to_end_sec))
            records.append(base_trials[len(base_trials) // 2])

            for prof in profiles:
                for cache_state in cache_modes:
                    trial_runs = [
                        _run_single_subprocess(
                            workload_name=wl_name,
                            script_path=script_path,
                            script_args=script_args,
                            mode=prof,
                            cache_state=cache_state,
                            cache_dir=cache_dir,
                        )
                        for _ in range(max(1, trials))
                    ]
                    trial_runs.sort(key=lambda r: (r.exit_code != 0, r.end_to_end_sec))
                    records.append(trial_runs[len(trial_runs) // 2])
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Subprocess A/B Benchmark Runner (Baseline vs ptopt)")
    parser.add_argument("--mode", default="smoke", choices=list(MODE_DEFAULTS.keys()))
    parser.add_argument("--workloads", default="all", help="Comma-separated workload names or 'all'")
    parser.add_argument("--profiles", default=None, help="Comma-separated profiles (safe,balanced,aggressive)")
    parser.add_argument("--cache-modes", default=None, help="Comma-separated cache states (cold,warm)")
    parser.add_argument("--steps", type=int, default=None, help="Override steps per workload")
    parser.add_argument("--trials", type=int, default=1, help="Trials per configuration")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-name", default=None, help="Report suffix (default: phase-2.3 for smoke else benchmark)")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args(argv)

    preset = MODE_DEFAULTS[args.mode]
    fp = collect_fingerprint()
    target_device = ("cuda" if fp.effective_backend == "cuda" else "cpu") if args.device == "auto" else args.device

    selected_workloads = list(WORKLOAD_SPECS.keys()) if args.workloads == "all" else [
        w.strip() for w in args.workloads.split(",") if w.strip()
    ]
    for w in selected_workloads:
        if w not in WORKLOAD_SPECS:
            raise ValueError(f"Unknown workload '{w}'. Available: {list(WORKLOAD_SPECS.keys())}")

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()] if args.profiles else preset["profiles"]
    cache_modes = [c.strip() for c in args.cache_modes.split(",") if c.strip()] if args.cache_modes else preset["cache_modes"]
    steps = args.steps if args.steps is not None else (preset["steps"] if target_device == "cuda" else min(preset["steps"], 6))

    resolver = ConfigResolver()
    profile_configs = {
        prof: resolver.resolve(fp, profile=prof, env={}).config.to_dict()
        for prof in profiles
    }

    records = run_benchmark_suite(
        workloads=selected_workloads,
        profiles=profiles,
        cache_modes=cache_modes,
        steps=steps,
        device=target_device,
        seed=args.seed,
        trials=args.trials,
    )

    report_name = args.report_name or ("phase-2.3" if args.mode == "smoke" else f"bench-{args.mode}")
    out_path, report_text = generate_benchmark_text_report(
        phase_title=f"PHASE 2.3 BENCHMARK HARNESS ({args.mode.upper()})",
        report_name=report_name,
        records=records,
        fingerprint=fp,
        profile_configs=profile_configs,
        report_dir=args.report_dir,
        repo_root=REPO_ROOT,
    )

    comparisons = build_comparison_rows(records)
    if args.json_out is not None:
        export_json_report(args.json_out, records, comparisons)
    if args.csv_out is not None:
        export_csv_report(args.csv_out, comparisons)

    print(report_text)
    print(f"[SAVED] Text report written to: {out_path}")
    any_fail = any(r.exit_code != 0 for r in records) or any(c.status == "FAIL" for c in comparisons)
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
