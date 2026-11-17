#!/usr/bin/env python3
"""Phase 3.5 Experiment Runner: Auto `torch.compile` Cold vs Warm Cache, Break-Even & Dynamic Shapes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent.parent / "pipline-for-test"
REPO_ROOT = BENCH_DIR.parent
for _p in (str(REPO_ROOT), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench import WORKLOADS_DIR, WORKLOAD_SPECS, _extract_workload_summary
from src.common.text_report import TextReportBuilder
from src.detect import collect_fingerprint
from telemetry import RunRecord, check_parity, compute_speedup, compute_vram_delta, merge_summary_and_telemetry

CUDA_COMPILE_TARGETS: list[tuple[str, str, int, list[str]]] = [
    ("transformer_lm", "small", 16, ["--d-model", "256", "--n-heads", "4", "--n-layers", "4", "--seq-len", "128"]),
    ("transformer_lm", "large", 16, ["--d-model", "768", "--n-heads", "12", "--n-layers", "12", "--seq-len", "512"]),
    ("cnn_vision", "small", 32, ["--image-size", "64", "--width", "64"]),
    ("cnn_vision", "large", 32, ["--image-size", "224", "--width", "128"]),
    ("deep_mlp", "small", 64, ["--width", "256", "--depth", "24"]),
    ("deep_mlp", "medium", 128, ["--width", "1024", "--depth", "24"]),
    ("edge_cases", "dynamic_shape", 8, ["--case", "dynamic_shape"]),
]

CPU_COMPILE_TARGETS: list[tuple[str, str, int, list[str]]] = [
    ("transformer_lm", "small", 8, ["--d-model", "128", "--n-heads", "4", "--n-layers", "2", "--seq-len", "64"]),
    ("deep_mlp", "small", 16, ["--width", "128", "--depth", "6"]),
    ("edge_cases", "dynamic_shape", 8, ["--case", "dynamic_shape"]),
]


def _run_test_file(rel_path: str) -> tuple[str, int, str]:
    test_file = REPO_ROOT / rel_path
    if importlib.util.find_spec("pytest") is not None:
        cmd = [sys.executable, "-m", "pytest", str(test_file), "-v"]
        runner = "pytest"
    else:
        cmd = [sys.executable, "-m", "unittest", rel_path, "-v"]
        runner = "unittest"
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return runner, res.returncode, (res.stdout + "\n" + res.stderr).strip()


def _run_compile_variant(
    *,
    wl_name: str,
    scale_name: str,
    batch_size: int,
    extra_args: list[str],
    steps: int,
    device: str,
    variant_name: str,
    cache_state: str,
    cache_dir: Path | None,
    is_baseline_raw: bool = False,
    compile_enabled: bool = False,
    overrides: list[str] | None = None,
) -> tuple[RunRecord, float]:
    spec = WORKLOAD_SPECS[wl_name]
    script_path = WORKLOADS_DIR / spec["script"]
    wl_args = [
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--seed",
        "42",
        "--device",
        device,
        *extra_args,
    ]
    env = os.environ.copy()

    with tempfile.TemporaryDirectory() as tmp:
        tel_path = Path(tmp) / "tel.json"
        if is_baseline_raw:
            cmd = [sys.executable, str(script_path), *wl_args]
        else:
            env["PTOPT_TELEMETRY_OUT"] = str(tel_path)
            if device == "cpu" and compile_enabled:
                env["PTOPT_COMPILE_CPU"] = "1"
            if cache_dir is not None:
                env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
                env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

            ovr_flags: list[str] = []
            for item in (overrides or []):
                ovr_flags.extend(["--override", item])
            if cache_dir is not None and compile_enabled:
                ovr_flags.extend(["--override", f"compile.cache_dir={cache_dir}"])

            cmd = [
                sys.executable,
                "-m",
                "src.cli.main",
                "--profile",
                "balanced",
                *(["--override", "compile.enabled=true"] if compile_enabled else ["--no-compile"]),
                *ovr_flags,
                "--",
                sys.executable,
                str(script_path),
                *wl_args,
            ]

        t0 = time.perf_counter()
        res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
        wall = time.perf_counter() - t0
        summary = _extract_workload_summary(res.stdout)
        tel_data = json.loads(tel_path.read_text(encoding="utf-8")) if tel_path.is_file() else None
        err = (res.stderr.strip() or res.stdout.strip())[-350:] if res.returncode != 0 else None
        raw_steps = (summary or {}).get("step_times_ms") or []
        step0_ms = float(raw_steps[0]) if raw_steps else 0.0
        rec = merge_summary_and_telemetry(
            workload=f"{wl_name}:{scale_name}",
            mode=variant_name,
            cache_state=cache_state,
            exit_code=res.returncode,
            summary=summary,
            telemetry=tel_data,
            wall_sec=wall,
            error=err,
        )
        return rec, round(step0_ms, 2)


def _format_breakeven(base_ms: float, warm_ms: float, cold_step0_ms: float) -> str:
    gain_ms = base_ms - warm_ms
    compile_penalty_ms = max(0.0, cold_step0_ms - warm_ms)
    if gain_ms <= 0.0:
        return "N/A (slower)"
    if compile_penalty_ms <= 0.0:
        return "1"
    return str(int(math.ceil(compile_penalty_ms / gain_ms)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3.5 Auto torch.compile Cold vs Warm Experiment")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    fp = collect_fingerprint()
    target_device = "cuda" if fp.effective_backend == "cuda" else "cpu"
    is_cuda = target_device == "cuda"
    steps = args.steps if is_cuda else min(args.steps, 5)
    targets = CUDA_COMPILE_TARGETS if is_cuda else CPU_COMPILE_TARGETS

    unit_runner, unit_rc, unit_out = _run_test_file("tests/unit/test_pass_compile.py")
    mig_runner, mig_rc, mig_out = _run_test_file("tests/migrate/test_phase3_5_compile.py")

    ab_rows: list[list[object]] = []
    breakeven_rows: list[list[object]] = []
    hook_lines: list[str] = []
    any_fail = False

    with tempfile.TemporaryDirectory() as cache_root:
        for wl_name, scale_name, bs, extra in targets:
            wl_cache = Path(cache_root) / f"{wl_name}_{scale_name}"
            if wl_cache.exists():
                shutil.rmtree(wl_cache)
            wl_cache.mkdir(parents=True, exist_ok=True)

            base_rec, base_step0 = _run_compile_variant(
                wl_name=wl_name,
                scale_name=scale_name,
                batch_size=bs,
                extra_args=extra,
                steps=steps,
                device=target_device,
                variant_name="baseline",
                cache_state="none",
                cache_dir=None,
                is_baseline_raw=True,
                compile_enabled=False,
            )

            eager_rec, eager_step0 = _run_compile_variant(
                wl_name=wl_name,
                scale_name=scale_name,
                batch_size=bs,
                extra_args=extra,
                steps=steps,
                device=target_device,
                variant_name="3.4_eager",
                cache_state="none",
                cache_dir=None,
                compile_enabled=False,
            )
            cold_rec, cold_step0 = _run_compile_variant(
                wl_name=wl_name,
                scale_name=scale_name,
                batch_size=bs,
                extra_args=extra,
                steps=steps,
                device=target_device,
                variant_name="3.5a_compile_cold",
                cache_state="cold",
                cache_dir=wl_cache,
                compile_enabled=True,
            )
            warm_rec, warm_step0 = _run_compile_variant(
                wl_name=wl_name,
                scale_name=scale_name,
                batch_size=bs,
                extra_args=extra,
                steps=steps,
                device=target_device,
                variant_name="3.5b_compile_warm",
                cache_state="warm",
                cache_dir=wl_cache,
                compile_enabled=True,
            )

            for label, rec in (
                ("3.4_eager", eager_rec),
                ("3.5a_compile_cold", cold_rec),
                ("3.5b_compile_warm", warm_rec),
            ):
                if base_rec.exit_code != 0 or rec.exit_code != 0:
                    any_fail = True
                    ab_rows.append([
                        f"{wl_name}:{scale_name}",
                        label,
                        f"{base_rec.step_stats.median:.2f}",
                        "ERR",
                        "-",
                        "0.00x",
                        f"{base_rec.end_to_end_sec:.2f}/{rec.end_to_end_sec:.2f}",
                        "N/A",
                        "None",
                        "FAIL",
                    ])
                    hook_lines.append(f"[ERROR] {wl_name}:{scale_name} ({label}): {rec.error or base_rec.error}")
                else:
                    parity = check_parity(base_rec.losses, rec.losses, profile="balanced")
                    speedup = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                    vram_d = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                    status = "PASS" if parity.passed else "FAIL"
                    if not parity.passed:
                        any_fail = True
                    ab_rows.append([
                        f"{wl_name}:{scale_name}",
                        label,
                        f"{base_rec.step_stats.median:.2f}",
                        f"{rec.step_stats.median:.2f}",
                        f"{rec.step_stats.p95:.2f}",
                        f"{speedup:.2f}x",
                        f"{base_rec.end_to_end_sec:.2f}/{rec.end_to_end_sec:.2f}",
                        f"{base_rec.peak_allocated_mb:.1f}/{rec.peak_allocated_mb:.1f} ({vram_d:+.1f})",
                        parity.max_rel_diff,
                        status,
                    ])

            if base_rec.exit_code == 0 and cold_rec.exit_code == 0 and warm_rec.exit_code == 0:
                sp_warm_vs_base = compute_speedup(base_rec.step_stats.median, warm_rec.step_stats.median)
                sp_warm_vs_eager = compute_speedup(eager_rec.step_stats.median, warm_rec.step_stats.median)
                be_steps = _format_breakeven(
                    base_rec.step_stats.median,
                    warm_rec.step_stats.median,
                    cold_step0,
                )
                breakeven_rows.append([
                    f"{wl_name}:{scale_name}",
                    f"{base_step0:.1f}",
                    f"{cold_step0:.1f}",
                    f"{warm_step0:.1f}",
                    f"{eager_rec.step_stats.median:.2f}",
                    f"{warm_rec.step_stats.median:.2f}",
                    f"{sp_warm_vs_base:.2f}x",
                    f"{sp_warm_vs_eager:.2f}x",
                    be_steps,
                ])

            for ev in cold_rec.fallback_events:
                hook_lines.append(
                    f"[{ev.get('status')}] {wl_name}:{scale_name} (cold) -> {ev.get('pass_name')}: {ev.get('reason')}"
                )

    overall_rc = 0 if (unit_rc == 0 and mig_rc == 0 and not any_fail) else 1
    builder = TextReportBuilder(
        phase_title="PHASE 3.5 EXPERIMENT REPORT: AUTO TORCH.COMPILE, CACHE & CHECKPOINT TRANSPARENCY",
        repo_root=REPO_ROOT,
    )
    builder.add_header({
        "Google Colab": fp.is_colab,
        "Backend / GPU": (
            f"{fp.effective_backend} / {fp.primary_gpu.name} ({fp.primary_gpu.sm_name})"
            if fp.primary_gpu
            else fp.effective_backend
        ),
        "Steps per Workload": steps,
        "Unit Tests": f"{unit_runner} (exit_code={unit_rc})",
        "Migrate Tests": f"{mig_runner} (exit_code={mig_rc})",
        "Overall Experiment": "PASS" if overall_rc == 0 else "FAIL",
    })
    builder.add_table(
        "COMPILE ABLATION (BASELINE VS EAGER VS COLD CACHE VS WARM CACHE)",
        ["Workload:Scale", "Variant", "Base ms", "Wrap ms", "Wrap P95", "Speedup", "E2E s (B/W)", "VRAM MB (B/W/d)", "MaxRelDiff", "Status"],
        ab_rows,
    )
    builder.add_table(
        "COLD VS WARM COMPILE BREAKDOWN & BREAK-EVEN STEPS",
        ["Workload:Scale", "Base Step0", "Cold Step0", "Warm Step0", "Eager ms", "Warm ms", "vs Base", "vs Eager", "BreakEven Steps"],
        breakeven_rows,
    )
    builder.add_lines_section("COMPILE & FALLBACK LOGS (COLD RUN)", hook_lines)
    builder.add_lines_section(f"MIGRATE TEST OUTPUT ({mig_runner})", mig_out.splitlines())
    builder.add_lines_section(f"UNIT TEST OUTPUT ({unit_runner})", unit_out.splitlines())
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if overall_rc == 0 else "FAILED_NEEDS_ATTENTION",
        note="Verify Cold vs Warm torch.compile speedup, break-even steps, dynamic-shape guard, and checkpoint parity on Google Colab before Phase 3.6",
    )

    out_path = builder.write("phase-3.5", report_dir=args.report_dir)
    print(builder.render())
    print(f"[SAVED] Text report written to: {out_path}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
