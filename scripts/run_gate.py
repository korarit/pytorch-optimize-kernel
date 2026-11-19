#!/usr/bin/env python3
"""Phase 4.3 Automated Regression Gate: `--mode fast` (CPU/CI) and `--mode full` (Colab GPU)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

BENCH_DIR = Path(__file__).resolve().parent.parent / "pipline-for-test"
REPO_ROOT = BENCH_DIR.parent
for _p in (str(REPO_ROOT), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench import WORKLOAD_SPECS, _extract_workload_summary
from ptopt.common.text_report import TextReportBuilder
from ptopt.config import detect_runtime_cache_key
from ptopt.detect import collect_fingerprint
from telemetry import RunRecord, check_parity, compute_speedup, compute_vram_delta, merge_summary_and_telemetry

# Migrate files that explicitly force `PTOPT_COMPILE_CPU=1`; excluded from `--mode fast` CPU gate.
_HEAVY_CPU_COMPILE_TESTS = frozenset({"test_phase3_5_compile.py"})


@dataclass(frozen=True)
class GateStepResult:
    """Execution summary for one step in the regression gate."""

    stage: str
    scope: str
    duration_s: float
    passed: bool
    detail: str
    failure_trace: str = ""


def discover_migrate_files(*, mode: str, migrate_dir: Path | None = None) -> list[Path]:
    """Return ordered migrate test files for `fast` (CPU-safe) or `full` gate modes."""
    root = migrate_dir or (REPO_ROOT / "tests" / "migrate")
    files = sorted(root.glob("test_*.py"))
    if mode == "fast":
        return [p for p in files if p.name not in _HEAVY_CPU_COMPILE_TESTS]
    return files


def _extract_unittest_summary(output: str, rc: int) -> str:
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    ran_line = next((ln for ln in reversed(lines) if ln.startswith("Ran ")), "")
    status_line = next((ln for ln in reversed(lines) if ln in ("OK", "FAILED") or ln.startswith("FAILED (")), "")
    if ran_line and status_line:
        return f"{ran_line} -> {status_line}"
    return ran_line or f"exit={rc}"


def _run_subprocess(cmd: Sequence[str], *, env: dict[str, str] | None = None) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(list(cmd), cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode, combined, elapsed


def _run_diagnostics_checks() -> list[GateStepResult]:
    results: list[GateStepResult] = []
    for flag, marker, label in (
        ("--doctor", "[Recommended Profile]", "CLI: ptopt --doctor"),
        ("--explain", "[FINAL RESOLVED CONFIG]", "CLI: ptopt --explain"),
    ):
        rc, out, dt = _run_subprocess([sys.executable, "-m", "ptopt.cli.main", flag])
        passed = rc == 0 and marker in out
        results.append(
            GateStepResult(
                stage=label,
                scope="Hardware probe & diagnostics",
                duration_s=dt,
                passed=passed,
                detail="Checks & summary OK" if passed else f"exit={rc}",
                failure_trace="" if passed else out[-1500:],
            )
        )
    return results


def _run_unit_suite() -> GateStepResult:
    # `unittest discover -s tests/unit` adds tests/unit directly to sys.path (immune to site-packages/tests shadowing).
    rc, out, dt = _run_subprocess([sys.executable, "-m", "unittest", "discover", "-s", "tests/unit", "-v"])
    passed = rc == 0
    return GateStepResult(
        stage="Unit Suite (tests/unit)",
        scope="Config, detect, passes, kernels, doctor, migrator",
        duration_s=dt,
        passed=passed,
        detail=_extract_unittest_summary(out, rc),
        failure_trace="" if passed else out[-2500:],
    )


def _run_migrate_files(files: Sequence[Path], *, mode: str) -> list[GateStepResult]:
    results: list[GateStepResult] = []
    for path in files:
        # Execute each migrate file by direct path so site-packages `tests/` packages cannot shadow `tests/migrate/`.
        rc, out, dt = _run_subprocess([sys.executable, str(path), "-v"])
        passed = rc == 0
        results.append(
            GateStepResult(
                stage=f"Migrate: {path.name}",
                scope=f"{mode} gate",
                duration_s=dt,
                passed=passed,
                detail=_extract_unittest_summary(out, rc),
                failure_trace="" if passed else out[-2000:],
            )
        )
    return results


def _run_smoke_benchmark(
    *,
    device: str,
    steps: int,
    profiles: Sequence[str] = ("supersafe", "safe", "balanced", "aggressive"),
) -> tuple[list[list[object]], bool, list[str]]:
    import json
    import os

    is_cuda = device == "cuda"
    workloads: list[tuple[str, int, int | None, list[str] | None]] = (
        [
            ("transformer_lm", steps, 16, ["--d-model", "512", "--n-heads", "8", "--n-layers", "6", "--seq-len", "256"]),
            ("cnn_vision", steps, 32, ["--image-size", "128", "--width", "96"]),
            ("deep_mlp", steps, 128, ["--width", "1024", "--depth", "24"]),
            ("infer_runner_bs1", max(steps, 20), 1, []),
            ("infer_runner_bs32", steps, 32, []),
            ("edge_cases", min(steps, 8), 8, ["--case", "all"]),
        ]
        if is_cuda
        else [
            ("transformer_lm", min(steps, 4), None, None),
            ("cnn_vision", min(steps, 4), None, None),
            ("deep_mlp", min(steps, 4), None, None),
            ("infer_runner_bs1", min(steps, 5), 1, None),
            ("edge_cases", min(steps, 4), None, None),
        ]
    )

    def _exec_wl(
        wl: str,
        wl_steps: int,
        bs_ovr: int | None,
        extra_ovr: list[str] | None,
        profile: str,
        is_base: bool,
        cache_dir: Path | None,
    ) -> RunRecord:
        spec = WORKLOAD_SPECS[wl]
        script_path = Path(REPO_ROOT / "pipline-for-test" / "workloads" / spec["script"])
        bs = bs_ovr if bs_ovr is not None else int(spec["batch_size"])
        extra = extra_ovr if extra_ovr is not None else list(spec["extra"])
        wl_args = ["--steps", str(wl_steps), "--batch-size", str(bs), "--seed", "42", "--device", device, *extra]
        env = os.environ.copy()
        with tempfile.TemporaryDirectory() as tmp:
            tel_path = Path(tmp) / "tel.json"
            if is_base:
                cmd = [sys.executable, str(script_path), *wl_args]
            else:
                env["PTOPT_TELEMETRY_OUT"] = str(tel_path)
                ovr: list[str] = []
                if cache_dir is not None:
                    ovr.extend(["--override", f"compile.cache_dir={cache_dir}"])
                if profile == "aggressive" and is_cuda:
                    ovr.extend(["--override", "compile.mode=default"])
                cmd = [sys.executable, "-m", "ptopt.cli.main", "--profile", profile, *ovr, "--", sys.executable, str(script_path), *wl_args]
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
            wall = time.perf_counter() - t0
            summary = _extract_workload_summary(proc.stdout)
            tel = json.loads(tel_path.read_text(encoding="utf-8")) if tel_path.is_file() else None
            err = (proc.stderr.strip() or proc.stdout.strip())[-300:] if proc.returncode != 0 else None
            return merge_summary_and_telemetry(
                workload=wl,
                mode="baseline" if is_base else profile,
                cache_state="warm" if cache_dir else "none",
                exit_code=proc.returncode,
                summary=summary,
                telemetry=tel,
                wall_sec=wall,
                error=err,
            )

    rows: list[list[object]] = []
    traces: list[str] = []
    bench_ok = True
    with tempfile.TemporaryDirectory() as cache_root:
        for wl, wl_steps, bs_ovr, extra_ovr in workloads:
            wl_cache = Path(cache_root) / wl
            wl_cache.mkdir(parents=True, exist_ok=True)
            base_rec = _exec_wl(wl, wl_steps, bs_ovr, extra_ovr, "safe", True, None)
            for prof in profiles:
                rec = _exec_wl(wl, wl_steps, bs_ovr, extra_ovr, prof, False, wl_cache)
                if base_rec.exit_code != 0 or rec.exit_code != 0:
                    bench_ok = False
                    rows.append([wl, prof, f"{base_rec.step_stats.median:.2f}", "-", "0.00x", "N/A", "N/A", "FAIL"])
                    traces.append(f"[BENCH FAIL] {wl} ({prof}): {rec.error or base_rec.error}")
                else:
                    parity = check_parity(base_rec.losses, rec.losses, profile=prof)
                    speedup = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                    vram_d = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                    if not parity.passed:
                        bench_ok = False
                        traces.append(f"[PARITY FAIL] {wl} ({prof}): {parity.reason}")
                    rows.append([
                        wl,
                        prof,
                        f"{base_rec.step_stats.median:.2f}",
                        f"{rec.step_stats.median:.2f}",
                        f"{speedup:.2f}x",
                        f"{base_rec.peak_allocated_mb:.1f}/{rec.peak_allocated_mb:.1f} ({vram_d:+.1f})",
                        parity.max_rel_diff,
                        "PASS" if parity.passed else "FAIL",
                    ])
    return rows, bench_ok, traces


def build_gate_report(
    *,
    mode: str,
    fp: object,
    cache_key: str,
    step_results: Sequence[GateStepResult],
    bench_rows: Sequence[Sequence[object]] | None = None,
    bench_ok: bool = True,
    bench_traces: Sequence[str] = (),
    report_dir: Path | None = None,
) -> tuple[Path, str, bool]:
    """Render and write `phase-4-fast.txt` or `phase-4.txt` using TextReportBuilder."""
    gpu_name = fp.primary_gpu.name if fp.primary_gpu else "CPU-only"  # type: ignore[attr-defined]
    device_label = (
        f"{fp.effective_backend} (sm_{fp.sm_int})"  # type: ignore[attr-defined]
        if fp.sm_int > 0  # type: ignore[attr-defined]
        else fp.effective_backend  # type: ignore[attr-defined]
    )
    all_ok = bench_ok and all(s.passed for s in step_results)
    title = (
        "PHASE 4.3 AUTOMATED REGRESSION GATE REPORT (FAST CPU / CI GATE)"
        if mode == "fast"
        else "PHASE 4 FINAL CONSOLIDATED REPORT: REGRESSION GATE & FULL VERIFICATION"
    )

    builder = TextReportBuilder(title, REPO_ROOT)
    builder.add_header({
        "Gate Mode": mode,
        "Google Colab": fp.is_colab,  # type: ignore[attr-defined]
        "GPU / Backend": f"{gpu_name} | {device_label}",
        "Python / PyTorch": f"{fp.host.python_version} | {fp.host.torch_version}",  # type: ignore[attr-defined]
        "CUDA / Triton": f"{fp.host.cuda_version or 'N/A'} | {fp.host.triton_version or 'N/A'}",  # type: ignore[attr-defined]
        "Runtime Cache Key": cache_key,
    })
    builder.add_table(
        "Regression Gate Suites",
        ["Stage", "Scope", "Duration", "Status", "Details"],
        [
            [s.stage, s.scope, f"{s.duration_s:.2f}s", "PASS" if s.passed else "FAIL", s.detail]
            for s in step_results
        ],
    )

    if bench_rows is not None:
        builder.add_table(
            "Smoke Benchmark (Baseline vs supersafe / safe / balanced / aggressive)",
            ["Workload", "Profile", "Base ms", "Wrap ms", "Speedup", "VRAM MB (B/W/d)", "MaxRelDiff", "Status"],
            bench_rows,
        )

    failures = [s.failure_trace for s in step_results if s.failure_trace] + list(bench_traces)
    builder.add_lines_section(
        "Failure Diagnostics & Traces",
        failures if failures else ["None — all regression gate checks passed cleanly."],
    )

    total_wall = sum(s.duration_s for s in step_results)
    builder.add_kv_section(
        "Final Gate Verdict",
        {
            "Gate Mode": mode,
            "Stages Passed": f"{sum(1 for s in step_results if s.passed)}/{len(step_results)}",
            "Smoke Benchmark": "SKIPPED (fast mode)" if bench_rows is None else ("PASS" if bench_ok else "FAIL"),
            "Suite Wall Time": f"{total_wall:.2f}s",
            "Overall Verdict": "PASS" if all_ok else "FAIL",
        },
    )
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if all_ok else "NEEDS_FIX",
        note=(
            "Fast CI gate complete"
            if mode == "fast"
            else "Review Final Text Report (`report/*-phase-4.txt`) on Google Colab for final project sign-off"
        ),
    )

    report_name = "phase-4-fast" if mode == "fast" else "phase-4"
    out_path = builder.write(name=report_name, report_dir=report_dir)
    return out_path, builder.render(), all_ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Automated Regression Gate runner for Phase 4.3.")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast", help="Gate scope: fast (CPU/CI) or full (Colab GPU)")
    parser.add_argument("--bench-steps", type=int, default=12, help="Steps per smoke benchmark workload in full mode")
    parser.add_argument("--report-dir", type=Path, default=None, help="Optional custom output directory for reports")
    args = parser.parse_args(list(argv) if argv is not None else None)

    fp = collect_fingerprint()
    cache_key = detect_runtime_cache_key().cache_key
    device = "cuda" if fp.effective_backend == "cuda" else "cpu"

    step_results: list[GateStepResult] = []
    step_results.extend(_run_diagnostics_checks())
    step_results.append(_run_unit_suite())
    migrate_files = discover_migrate_files(mode=args.mode)
    step_results.extend(_run_migrate_files(migrate_files, mode=args.mode))

    bench_rows: list[list[object]] | None = None
    bench_ok = True
    bench_traces: list[str] = []
    if args.mode == "full":
        bench_rows, bench_ok, bench_traces = _run_smoke_benchmark(device=device, steps=args.bench_steps)

    out_path, rendered, all_ok = build_gate_report(
        mode=args.mode,
        fp=fp,
        cache_key=cache_key,
        step_results=step_results,
        bench_rows=bench_rows,
        bench_ok=bench_ok,
        bench_traces=bench_traces,
        report_dir=args.report_dir,
    )
    print(rendered)
    print(f"Report written to: {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
