#!/usr/bin/env python3
"""Phase 3.4 Experiment Runner: Scale-Up Bottleneck Profiling & Deep Debug Diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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

CUDA_SCALE_MATRIX: dict[str, list[tuple[str, int, list[str]]]] = {
    "transformer_lm": [
        ("small", 16, ["--d-model", "256", "--n-heads", "4", "--n-layers", "4", "--seq-len", "128"]),
        ("medium", 16, ["--d-model", "512", "--n-heads", "8", "--n-layers", "6", "--seq-len", "256"]),
        ("large", 16, ["--d-model", "768", "--n-heads", "12", "--n-layers", "12", "--seq-len", "512"]),
    ],
    "cnn_vision": [
        ("small", 32, ["--image-size", "64", "--width", "64"]),
        ("medium", 32, ["--image-size", "128", "--width", "96"]),
        ("large", 32, ["--image-size", "224", "--width", "128"]),
    ],
    "deep_mlp": [
        ("small", 64, ["--width", "256", "--depth", "24"]),
        ("medium", 128, ["--width", "1024", "--depth", "24"]),
        ("large", 256, ["--width", "2048", "--depth", "24"]),
    ],
}

CPU_SCALE_MATRIX: dict[str, list[tuple[str, int, list[str]]]] = {
    "transformer_lm": [
        ("small", 8, ["--d-model", "128", "--n-heads", "4", "--n-layers", "2", "--seq-len", "64"]),
        ("medium_cpu", 8, ["--d-model", "256", "--n-heads", "4", "--n-layers", "4", "--seq-len", "128"]),
    ],
    "cnn_vision": [
        ("small", 16, ["--image-size", "64", "--width", "32"]),
        ("medium_cpu", 16, ["--image-size", "96", "--width", "48"]),
    ],
    "deep_mlp": [
        ("small", 32, ["--width", "256", "--depth", "12"]),
        ("medium_cpu", 64, ["--width", "512", "--depth", "16"]),
    ],
}


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


def _run_scaled_variant(
    *,
    wl_name: str,
    scale_name: str,
    batch_size: int,
    extra_args: list[str],
    steps: int,
    device: str,
    variant_name: str,
    is_baseline_raw: bool = False,
    disable_builtin_passes: bool = False,
    overrides: list[str] | None = None,
    debug_profile: bool = False,
    torch_profile: bool = False,
    profile_start_step: int = 2,
    profile_num_steps: int = 3,
) -> tuple[RunRecord, dict[str, Any] | None]:
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
            if disable_builtin_passes:
                env["PTOPT_DISABLE_BUILTIN_PASSES"] = "1"
            if debug_profile:
                env["PTOPT_DEBUG_PROFILE"] = "1"
            if torch_profile:
                env["PTOPT_TORCH_PROFILE"] = "1"
                env["PTOPT_TORCH_PROFILE_START"] = str(profile_start_step)
                env["PTOPT_TORCH_PROFILE_STEPS"] = str(profile_num_steps)

            ovr_flags: list[str] = []
            for item in (overrides or []):
                ovr_flags.extend(["--override", item])
            cmd = [
                sys.executable,
                "-m",
                "src.cli.main",
                "--profile",
                "safe" if disable_builtin_passes else "balanced",
                "--no-compile",
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
        rec = merge_summary_and_telemetry(
            workload=f"{wl_name}:{scale_name}",
            mode=variant_name,
            cache_state="none",
            exit_code=res.returncode,
            summary=summary,
            telemetry=tel_data,
            wall_sec=wall,
            error=err,
        )
        return rec, tel_data


def _diagnose_bottlenecks(
    ab_records: dict[tuple[str, str, str], RunRecord],
    debug_payloads: dict[tuple[str, str, str], dict[str, Any]],
) -> list[str]:
    """Generate actionable bottleneck classification & optimization hints from A/B + debug telemetry."""
    lines: list[str] = []
    seen_pairs = sorted({(wl, sc) for wl, sc, _ in ab_records})
    for wl, sc in seen_pairs:
        base = ab_records.get((wl, sc, "baseline"))
        no_amp = ab_records.get((wl, sc, "3.1_3.3_no_amp"))
        full = ab_records.get((wl, sc, "3.1_3.3_full"))
        if not base or not no_amp or not full:
            continue

        sp_no_amp = compute_speedup(base.step_stats.median, no_amp.step_stats.median)
        sp_full = compute_speedup(base.step_stats.median, full.step_stats.median)
        vram_saved = compute_vram_delta(base.peak_allocated_mb, full.peak_allocated_mb)

        dbg_wrap = (debug_payloads.get((wl, sc, "wrapped_dbg")) or {}).get("debug_profile", {})
        pb = dbg_wrap.get("phase_breakdown", {})
        shares = pb.get("steady_shares_pct", {})

        tags: list[str] = []
        if shares.get("data_other", 0.0) >= 28.0:
            tags.append(f"HOST/DATALOADER_BOUND (data_other={shares['data_other']}%)")
        if sp_full < sp_no_amp * 0.94:
            tags.append(
                f"EAGER_AMP_OVERHEAD (no_amp={sp_no_amp:.2f}x -> +amp={sp_full:.2f}x; eager autocast/cast slower than compute -> target for torch.compile 3.5 or size gate)"
            )
        elif sp_full >= 1.15:
            tags.append(f"COMPUTE_AMP_WIN ({sp_full:.2f}x total, no_amp={sp_no_amp:.2f}x -> TensorCore scaling works)")
        if shares.get("optim", 0.0) >= 18.0:
            tags.append(f"OPTIMIZER_STEP_HEAVY (optim={shares['optim']}%)")
        if shares.get("loss_clip", 0.0) >= 12.0:
            tags.append(f"LOSS_OR_CLIP_OVERHEAD (loss_clip={shares['loss_clip']}% -> candidate for fused CrossEntropy)")
        if shares.get("forward", 0.0) + shares.get("backward", 0.0) >= 65.0 and not any("AMP_WIN" in t for t in tags):
            tags.append(
                f"KERNEL_LAUNCH_OR_FWD_BWD_BOUND (fwd+bwd={shares.get('forward', 0.0) + shares.get('backward', 0.0):.1f}% -> fuse ops via Phase 3.5 torch.compile / 3.7 Triton)"
            )
        if not tags:
            tags.append(f"BALANCED_EXECUTION (speedup={sp_full:.2f}x)")

        lines.append(
            f"- {wl} [{sc}]: base={base.step_stats.median:.2f}ms | no_amp={no_amp.step_stats.median:.2f}ms ({sp_no_amp:.2f}x) "
            f"| full={full.step_stats.median:.2f}ms ({sp_full:.2f}x, VRAM {vram_saved:+.1f}MB) -> {' ; '.join(tags)}"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3.4 Scale-Up Bottleneck Profiling & Deep Debug Diagnostics")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--debug-steps", type=int, default=8)
    args = parser.parse_args()

    fp = collect_fingerprint()
    target_device = "cuda" if fp.effective_backend == "cuda" else "cpu"
    is_cuda = target_device == "cuda"
    ab_steps = args.steps if is_cuda else min(args.steps, 4)
    dbg_steps = args.debug_steps if is_cuda else min(args.debug_steps, 4)
    prof_start = 2 if dbg_steps >= 5 else 1
    prof_count = 3 if dbg_steps >= 5 else 2

    unit_runner, unit_rc, unit_out = _run_test_file("tests/unit/test_debug_profiler.py")

    scale_matrix = CUDA_SCALE_MATRIX if is_cuda else CPU_SCALE_MATRIX
    ab_variants: list[tuple[str, bool, list[str]]] = [
        ("baseline", True, []),
        ("3.1_3.3_no_amp", False, ["amp.enabled=false", "amp.grad_scaler=false"]),
        ("3.1_3.3_full", False, []),
    ]

    ab_records: dict[tuple[str, str, str], RunRecord] = {}
    ab_rows: list[list[object]] = []
    hook_lines: list[str] = []
    any_fail = False

    # 1. Clean Multi-Scale A/B Benchmark (no debug sync overhead)
    for wl_name, scales in scale_matrix.items():
        for scale_name, bs, extra in scales:
            base_rec: RunRecord | None = None
            for var_name, is_raw, ov in ab_variants:
                rec, _ = _run_scaled_variant(
                    wl_name=wl_name,
                    scale_name=scale_name,
                    batch_size=bs,
                    extra_args=extra,
                    steps=ab_steps,
                    device=target_device,
                    variant_name=var_name,
                    is_baseline_raw=is_raw,
                    overrides=ov,
                    debug_profile=False,
                    torch_profile=False,
                )
                ab_records[(wl_name, scale_name, var_name)] = rec
                if is_raw:
                    base_rec = rec
                    continue
                assert base_rec is not None
                if base_rec.exit_code != 0 or rec.exit_code != 0:
                    any_fail = True
                    ab_rows.append([
                        wl_name,
                        scale_name,
                        var_name,
                        f"{base_rec.step_stats.median:.2f}",
                        f"{rec.step_stats.median:.2f}",
                        "-",
                        "0.00x",
                        f"{base_rec.end_to_end_sec:.2f}/{rec.end_to_end_sec:.2f}",
                        "N/A",
                        "None",
                        "FAIL",
                    ])
                    hook_lines.append(f"[ERROR] {wl_name}:{scale_name} ({var_name}) -> {rec.error or base_rec.error}")
                else:
                    parity = check_parity(base_rec.losses, rec.losses, profile="balanced")
                    speedup = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                    vram_d = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                    status = "PASS" if parity.passed else "FAIL"
                    if not parity.passed:
                        any_fail = True
                    ab_rows.append([
                        wl_name,
                        scale_name,
                        var_name,
                        f"{base_rec.step_stats.median:.2f}",
                        f"{rec.step_stats.median:.2f}",
                        f"{rec.step_stats.p95:.2f}",
                        f"{speedup:.2f}x",
                        f"{base_rec.end_to_end_sec:.2f}/{rec.end_to_end_sec:.2f}",
                        f"{base_rec.peak_allocated_mb:.1f}/{rec.peak_allocated_mb:.1f} ({vram_d:+.1f})",
                        parity.max_rel_diff,
                        status,
                    ])
                    if var_name == "3.1_3.3_full":
                        for ev in rec.fallback_events:
                            hook_lines.append(
                                f"[{ev.get('status')}] {wl_name}:{scale_name} -> {ev.get('pass_name')}: {ev.get('reason')}"
                            )

    # 2. Deep Debug Profiling on smallest & largest scale per workload
    debug_payloads: dict[tuple[str, str, str], dict[str, Any]] = {}
    phase_rows: list[list[object]] = []
    trace_lines: list[str] = []
    mem_rows: list[list[object]] = []
    prof_lines: list[str] = []

    for wl_name, scales in scale_matrix.items():
        target_scales = [scales[0], scales[-1]] if len(scales) > 1 else [scales[0]]
        for scale_name, bs, extra in target_scales:
            for dbg_mode, disable_passes in (("baseline_dbg", True), ("wrapped_dbg", False)):
                rec, tel = _run_scaled_variant(
                    wl_name=wl_name,
                    scale_name=scale_name,
                    batch_size=bs,
                    extra_args=extra,
                    steps=dbg_steps,
                    device=target_device,
                    variant_name=dbg_mode,
                    is_baseline_raw=False,
                    disable_builtin_passes=disable_passes,
                    debug_profile=True,
                    torch_profile=True,
                    profile_start_step=prof_start,
                    profile_num_steps=prof_count,
                )
                if rec.exit_code != 0 or not tel:
                    any_fail = True
                    hook_lines.append(f"[ERROR] debug {wl_name}:{scale_name} ({dbg_mode}) -> {rec.error}")
                    continue
                debug_payloads[(wl_name, scale_name, dbg_mode)] = tel
                dbg = tel.get("debug_profile", {})
                pb = dbg.get("phase_breakdown", {})
                if pb:
                    st = pb["steady_mean_ms"]
                    sh = pb["steady_shares_pct"]
                    phase_rows.append([
                        wl_name,
                        scale_name,
                        dbg_mode,
                        f"{pb['step_0_cold']['total_ms']:.2f}",
                        f"{st['total_ms']:.2f}",
                        f"{st['data_other_ms']:.2f} ({sh['data_other']}%)",
                        f"{st['forward_ms']:.2f} ({sh['forward']}%)",
                        f"{st['loss_clip_ms']:.2f} ({sh['loss_clip']}%)",
                        f"{st['backward_ms']:.2f} ({sh['backward']}%)",
                        f"{st['optim_ms']:.2f} ({sh['optim']}%)",
                    ])
                    trace_lines.append(f"[{wl_name}:{scale_name} | {dbg_mode}] First steps breakdown (ms):")
                    for s in pb.get("per_step_trace", [])[:5]:
                        trace_lines.append(
                            f"  step {s['step']:>2}: total={s['total_ms']:>7.2f} "
                            f"(data={s['data_other_ms']:.2f}, fwd={s['forward_ms']:.2f}, "
                            f"loss={s['loss_clip_ms']:.2f}, bwd={s['backward_ms']:.2f}, opt={s['optim_ms']:.2f})"
                        )

                mem_dbg = dbg.get("cuda_memory_debug")
                if mem_dbg:
                    mem_rows.append([
                        wl_name,
                        scale_name,
                        dbg_mode,
                        f"{mem_dbg['peak_allocated_mb']:.1f}",
                        f"{mem_dbg['peak_reserved_mb']:.1f}",
                        f"{mem_dbg['reserved_overhead_mb']:.1f}",
                        f"{mem_dbg['total_vram_mb']:.1f}",
                        f"{mem_dbg['headroom_mb']:.1f}",
                        f"{mem_dbg['utilization_pct']:.1f}%",
                        mem_dbg["num_alloc_retries"],
                        mem_dbg["num_ooms"],
                    ])

                tp = dbg.get("torch_profiler")
                if tp:
                    win = tp.get("window_steps", [])
                    ops_key = "top_cuda_ops" if tp.get("top_cuda_ops") else "top_cpu_ops"
                    ops = tp.get(ops_key, [])[:8]
                    prof_lines.append(
                        f"[{wl_name}:{scale_name} | {dbg_mode}] Profiler window={win} "
                        f"total_self_cuda={tp.get('total_self_cuda_ms')}ms total_self_cpu={tp.get('total_self_cpu_ms')}ms ({ops_key}):"
                    )
                    for op in ops:
                        self_val = op.get("self_cuda_ms", op.get("self_cpu_ms", 0.0))
                        tot_val = op.get("total_cuda_ms", op.get("total_cpu_ms", 0.0))
                        prof_lines.append(
                            f"  {op['name']:<48} calls={op['calls']:<5} self={self_val:>7.3f}ms ({op['pct']:>4.1f}%) total={tot_val:>7.3f}ms"
                        )

    diag_lines = _diagnose_bottlenecks(ab_records, debug_payloads)
    overall_rc = 0 if (unit_rc == 0 and not any_fail) else 1

    builder = TextReportBuilder(
        phase_title="PHASE 3.4 EXPERIMENT REPORT: SCALE-UP BOTTLENECK PROFILING & DEEP DEBUG",
        repo_root=REPO_ROOT,
    )
    builder.add_header({
        "Google Colab": fp.is_colab,
        "Backend / GPU": (
            f"{fp.effective_backend} / {fp.primary_gpu.name} ({fp.primary_gpu.sm_name})"
            if fp.primary_gpu
            else fp.effective_backend
        ),
        "A/B Steps / Debug Steps": f"{ab_steps} / {dbg_steps}",
        "Unit Tests (Debug Prof)": f"{unit_runner} (exit_code={unit_rc})",
        "Overall Experiment": "PASS" if overall_rc == 0 else "FAIL",
    })
    builder.add_table(
        "SCALE-UP CLEAN A/B BENCHMARK (SMALL VS MEDIUM VS LARGE)",
        ["Workload", "Scale", "Variant", "Base ms", "Wrap ms", "Wrap P95", "Speedup", "E2E s (B/W)", "VRAM MB (B/W/d)", "MaxRelDiff", "Status"],
        ab_rows,
    )
    builder.add_table(
        "DEEP DEBUG: STEP PHASE BREAKDOWN (STEADY-STATE MEAN MS & SHARE %)",
        ["Workload", "Scale", "Mode", "Step0 ms", "Steady ms", "Data/Other", "Forward", "Loss/Clip", "Backward", "Optim"],
        phase_rows,
    )
    if mem_rows:
        builder.add_table(
            "DEEP DEBUG: CUDA ALLOCATOR & VRAM HEADROOM",
            ["Workload", "Scale", "Mode", "Alloc MB", "Resv MB", "Ovhead MB", "Total MB", "Headroom", "Util %", "Retries", "OOMs"],
            mem_rows,
        )
    builder.add_lines_section("AUTOMATED BOTTLENECK DIAGNOSIS & RECOMMENDATIONS", diag_lines)
    builder.add_lines_section("WARMUP VS STEADY-STATE STEP TRACE", trace_lines)
    builder.add_lines_section("TOP PYTORCH PROFILER OPS (BASELINE VS WRAPPED)", prof_lines)
    builder.add_lines_section("PASS & HOOK LOGS", hook_lines)
    builder.add_lines_section(f"UNIT TEST OUTPUT ({unit_runner})", unit_out.splitlines())
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if overall_rc == 0 else "FAILED_NEEDS_ATTENTION",
        note="Inspect Scale-Up bottlenecks, phase shares, and top ops on Google Colab before designing Phase 3.5 (torch.compile) & profile heuristics",
    )

    out_path = builder.write("phase-3.4", report_dir=args.report_dir)
    print(builder.render())
    print(f"[SAVED] Text report written to: {out_path}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
