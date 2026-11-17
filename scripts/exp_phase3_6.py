#!/usr/bin/env python3
"""Phase 3.6 Experiment Runner: Inference / Running Optimization + Deep Debug Profiling."""

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

from bench import WORKLOADS_DIR, _extract_workload_summary
from src.common.text_report import TextReportBuilder
from src.detect import collect_fingerprint
from telemetry import RunRecord, check_parity, compute_speedup, compute_vram_delta, merge_summary_and_telemetry, percentile


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


def _run_infer_variant(
    *,
    case_name: str,
    batch_size: int,
    steps: int,
    extra_args: list[str],
    device: str,
    variant_name: str,
    profile: str,
    is_baseline: bool = False,
    disable_builtin_passes: bool = False,
    compile_enabled: bool = False,
    overrides: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    cache_dir: Path | None = None,
    debug_profile: bool = False,
    torch_profile: bool = False,
    prof_start: int = 4,
    prof_steps: int = 5,
) -> tuple[RunRecord, float, float, dict[str, Any] | None]:
    script_path = WORKLOADS_DIR / "infer_runner.py"
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
    if extra_env:
        env.update(extra_env)

    with tempfile.TemporaryDirectory() as tmp:
        tel_path = Path(tmp) / "tel.json"
        if is_baseline:
            cmd = [sys.executable, str(script_path), *wl_args]
        else:
            env["PTOPT_TELEMETRY_OUT"] = str(tel_path)
            if disable_builtin_passes:
                env["PTOPT_DISABLE_BUILTIN_PASSES"] = "1"
            if debug_profile:
                env["PTOPT_DEBUG_PROFILE"] = "1"
            if torch_profile:
                env["PTOPT_TORCH_PROFILE"] = "1"
                env["PTOPT_TORCH_PROFILE_START"] = str(prof_start)
                env["PTOPT_TORCH_PROFILE_STEPS"] = str(prof_steps)
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
                profile,
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
        rec = merge_summary_and_telemetry(
            workload=case_name,
            mode=variant_name,
            cache_state="warm" if cache_dir else "none",
            exit_code=res.returncode,
            summary=summary,
            telemetry=tel_data,
            wall_sec=wall,
            error=err,
        )
        raw_steps = (summary or {}).get("step_times_ms") or []
        warmup = max(1, int(0.2 * len(raw_steps))) if len(raw_steps) >= 5 else (1 if len(raw_steps) >= 3 else 0)
        steady_steps = [float(x) for x in raw_steps[warmup:]] if raw_steps else []
        p99 = round(percentile(steady_steps, 99.0), 3) if steady_steps else 0.0
        step0 = round(float(raw_steps[0]), 2) if raw_steps else 0.0
        return rec, p99, step0, tel_data


def _diagnose_inference(ab_records: dict[tuple[str, str], RunRecord], is_cuda: bool) -> list[str]:
    lines: list[str] = []
    cases = sorted({c for c, _ in ab_records})
    for case in cases:
        base = ab_records.get((case, "baseline"))
        inf_m = ab_records.get((case, "3.6a_inference_mode"))
        comp = ab_records.get((case, "3.6b_infer_amp_compile"))
        cg = ab_records.get((case, "3.6c_cuda_graphs"))
        fp8 = ab_records.get((case, "3.6d_fp8_sm89"))
        if not base or not inf_m or not comp:
            continue
        sp_inf = compute_speedup(base.step_stats.median, inf_m.step_stats.median)
        sp_comp = compute_speedup(base.step_stats.median, comp.step_stats.median)
        msg = f"- {case}: base={base.step_stats.median:.2f}ms | inference_mode={inf_m.step_stats.median:.2f}ms ({sp_inf:.2f}x) | amp+compile={comp.step_stats.median:.2f}ms ({sp_comp:.2f}x)"
        if cg:
            sp_cg = compute_speedup(base.step_stats.median, cg.step_stats.median)
            msg += f" | cuda_graphs={cg.step_stats.median:.2f}ms ({sp_cg:.2f}x)"
        if fp8:
            sp_fp8 = compute_speedup(base.step_stats.median, fp8.step_stats.median)
            msg += f" | fp8_sm89={fp8.step_stats.median:.2f}ms ({sp_fp8:.2f}x)"
        if is_cuda and "bs1" in case and cg and compute_speedup(comp.step_stats.median, cg.step_stats.median) >= 1.3:
            msg += " -> LAUNCH_BOUND: CUDA Graphs eliminated Python/kernel dispatch overhead"
        elif is_cuda and "bs64" in case and sp_comp >= 1.5:
            msg += " -> COMPUTE_BOUND: BF16 + Inductor compile dominated throughput & cut VRAM"
        lines.append(msg)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3.6 Inference / Running Optimization Experiment")
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()

    fp = collect_fingerprint()
    target_device = "cuda" if fp.effective_backend == "cuda" else "cpu"
    is_cuda = target_device == "cuda"
    sm_int = fp.sm_int

    unit_runner, unit_rc, unit_out = _run_test_file("tests/unit/test_pass_inference.py")
    mig_runner, mig_rc, mig_out = _run_test_file("tests/migrate/test_phase3_6_inference.py")

    if is_cuda:
        cases: list[tuple[str, int, int, list[str]]] = [
            ("infer_bs1_small", 1, 50, ["--d-model", "256", "--n-layers", "4", "--seq-len", "64"]),
            ("infer_bs1_scaled", 1, 40, ["--d-model", "512", "--n-layers", "8", "--seq-len", "128"]),
            ("infer_bs64_tput", 64, 30, ["--d-model", "512", "--n-layers", "6", "--seq-len", "128"]),
        ]
    else:
        cases = [
            ("infer_bs1_small", 1, 12, ["--d-model", "128", "--n-layers", "2", "--seq-len", "32"]),
            ("infer_bs32_tput", 32, 8, ["--d-model", "128", "--n-layers", "2", "--seq-len", "32"]),
        ]

    ab_records: dict[tuple[str, str], RunRecord] = {}
    ab_rows: list[list[object]] = []
    hook_lines: list[str] = []
    any_fail = False

    with tempfile.TemporaryDirectory() as cache_root:
        for case_name, bs, steps, extra_args in cases:
            cache_dir = Path(cache_root) / case_name
            cache_dir.mkdir(parents=True, exist_ok=True)

            variants: list[tuple[str, str, bool, bool, list[str], dict[str, str], str]] = [
                ("baseline", "safe", True, False, [], {}, "safe"),
                ("3.6a_inference_mode", "safe", False, False, [], {}, "safe"),
                ("3.6b_infer_amp_compile", "balanced", False, True, [], {}, "balanced"),
            ]
            if is_cuda:
                variants.append((
                    "3.6c_cuda_graphs",
                    "aggressive",
                    False,
                    True,
                    ["inference.cuda_graphs=true", "kernels.enabled=false", "compile.mode=default"],
                    {},
                    "aggressive",
                ))
                if sm_int >= 89:
                    variants.append((
                        "3.6d_fp8_sm89",
                        "balanced",
                        False,
                        False,
                        ["amp.enabled=true", "amp.dtype=bf16"],
                        {"PTOPT_INFERENCE_FP8": "1"},
                        "aggressive",
                    ))
            else:
                variants.append((
                    "3.6d_cpu_int8",
                    "safe",
                    False,
                    False,
                    [],
                    {"PTOPT_INFERENCE_QUANT": "int8"},
                    "aggressive",
                ))

            base_rec: RunRecord | None = None
            for var_name, prof, is_base, comp_en, ovr, extra_env, parity_prof in variants:
                if comp_en and is_cuda:
                    _run_infer_variant(
                        case_name=case_name,
                        batch_size=bs,
                        steps=min(steps, 10),
                        extra_args=extra_args,
                        device=target_device,
                        variant_name=f"{var_name}_warmup",
                        profile=prof,
                        compile_enabled=True,
                        overrides=ovr,
                        extra_env=extra_env,
                        cache_dir=cache_dir,
                    )

                rec, p99, step0, _ = _run_infer_variant(
                    case_name=case_name,
                    batch_size=bs,
                    steps=steps,
                    extra_args=extra_args,
                    device=target_device,
                    variant_name=var_name,
                    profile=prof,
                    is_baseline=is_base,
                    compile_enabled=comp_en,
                    overrides=ovr,
                    extra_env=extra_env,
                    cache_dir=cache_dir if comp_en else None,
                )
                ab_records[(case_name, var_name)] = rec
                steady_sps = round((bs * 1000.0) / rec.step_stats.median, 1) if rec.step_stats.median > 0 else 0.0

                if is_base:
                    base_rec = rec
                    ab_rows.append([
                        case_name,
                        var_name,
                        f"{step0:.1f}",
                        f"{rec.step_stats.median:.2f}",
                        f"{rec.step_stats.p95:.2f}",
                        f"{p99:.2f}",
                        f"{steady_sps:.1f}",
                        f"{rec.end_to_end_sec:.2f}s",
                        "1.00x",
                        f"{rec.peak_allocated_mb:.1f} (+0.0)",
                        0.0,
                        "BASE",
                    ])
                    continue

                assert base_rec is not None
                if base_rec.exit_code != 0 or rec.exit_code != 0:
                    any_fail = True
                    ab_rows.append([case_name, var_name, "-", "-", "-", "-", "0.0", "-", "0.00x", "N/A", "None", "FAIL"])
                    hook_lines.append(f"[ERROR] {case_name} ({var_name}): {rec.error or base_rec.error}")
                else:
                    parity = check_parity(base_rec.losses, rec.losses, profile=parity_prof)
                    speedup = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                    vram_d = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                    status = "PASS" if parity.passed else "FAIL"
                    if not parity.passed:
                        any_fail = True
                    ab_rows.append([
                        case_name,
                        var_name,
                        f"{step0:.1f}",
                        f"{rec.step_stats.median:.2f}",
                        f"{rec.step_stats.p95:.2f}",
                        f"{p99:.2f}",
                        f"{steady_sps:.1f}",
                        f"{rec.end_to_end_sec:.2f}s",
                        f"{speedup:.2f}x",
                        f"{rec.peak_allocated_mb:.1f} ({vram_d:+.1f})",
                        parity.max_rel_diff,
                        status,
                    ])
                    for ev in rec.fallback_events:
                        hook_lines.append(
                            f"[{ev.get('status')}] {case_name} ({var_name}) -> {ev.get('pass_name')}: {ev.get('reason')}"
                        )

        # 2. Deep Debug & Profiler Pass (Host/Data vs Forward + Jitter + Top Ops)
        debug_rows: list[list[object]] = []
        mem_rows: list[list[object]] = []
        trace_lines: list[str] = []
        prof_lines: list[str] = []

        dbg_targets = [cases[0], cases[-1]]
        for case_name, bs, _steps, extra_args in dbg_targets:
            dbg_steps = 12 if is_cuda else 6
            prof_start = 4 if is_cuda else 2
            prof_cnt = 5 if is_cuda else 3
            dbg_modes: list[tuple[str, str, bool, bool, list[str], dict[str, str]]] = [
                ("baseline_dbg", "safe", True, False, [], {}),
            ]
            if is_cuda:
                dbg_modes.append((
                    "cuda_graphs_dbg",
                    "aggressive",
                    False,
                    True,
                    ["inference.cuda_graphs=true", "kernels.enabled=false", "compile.mode=default"],
                    {},
                ))
                if sm_int >= 89:
                    dbg_modes.append((
                        "fp8_sm89_dbg",
                        "balanced",
                        False,
                        False,
                        ["amp.enabled=true", "amp.dtype=bf16"],
                        {"PTOPT_INFERENCE_FP8": "1"},
                    ))
            else:
                dbg_modes.append(("infer_mode_dbg", "safe", False, False, [], {}))

            dbg_cache = Path(cache_root) / case_name
            for d_name, prof, dis_passes, comp_en, ovr, env_x in dbg_modes:
                rec, _p99, _s0, tel = _run_infer_variant(
                    case_name=case_name,
                    batch_size=bs,
                    steps=dbg_steps,
                    extra_args=extra_args,
                    device=target_device,
                    variant_name=d_name,
                    profile=prof,
                    disable_builtin_passes=dis_passes,
                    compile_enabled=comp_en,
                    overrides=ovr,
                    extra_env=env_x,
                    cache_dir=dbg_cache if comp_en else None,
                    debug_profile=True,
                    torch_profile=True,
                    prof_start=prof_start,
                    prof_steps=prof_cnt,
                )
                if rec.exit_code != 0 or not tel:
                    hook_lines.append(f"[ERROR] debug {case_name} ({d_name}): {rec.error}")
                    continue
                dbg = tel.get("debug_profile", {})
                pb = dbg.get("phase_breakdown", {})
                if pb:
                    st = pb["steady_mean_ms"]
                    sh = pb["steady_shares_pct"]
                    debug_rows.append([
                        case_name,
                        d_name,
                        f"{pb['step_0_cold']['total_ms']:.2f}",
                        f"{st['total_ms']:.2f}",
                        f"{rec.step_stats.min:.2f}/{rec.step_stats.max:.2f}/{rec.step_stats.std:.2f}",
                        f"{st['data_other_ms']:.2f} ({sh['data_other']}%)",
                        f"{st['forward_ms']:.2f} ({sh['forward']}%)",
                    ])
                    trace_lines.append(f"[{case_name} | {d_name}] First steps trace (ms):")
                    for s in pb.get("per_step_trace", [])[:5]:
                        trace_lines.append(
                            f"  step {s['step']:>2}: total={s['total_ms']:>7.2f} (host_data={s['data_other_ms']:.2f}, fwd={s['forward_ms']:.2f})"
                        )
                mem_dbg = dbg.get("cuda_memory_debug")
                if mem_dbg:
                    mem_rows.append([
                        case_name,
                        d_name,
                        f"{mem_dbg['peak_allocated_mb']:.1f}",
                        f"{mem_dbg['peak_reserved_mb']:.1f}",
                        f"{mem_dbg['reserved_overhead_mb']:.1f}",
                        f"{mem_dbg['headroom_mb']:.1f}",
                        f"{mem_dbg['utilization_pct']:.1f}%",
                        mem_dbg["num_alloc_retries"],
                    ])
                tp = dbg.get("torch_profiler")
                if tp:
                    ops_key = "top_cuda_ops" if tp.get("top_cuda_ops") else "top_cpu_ops"
                    ops = tp.get(ops_key, [])[:8]
                    prof_lines.append(
                        f"[{case_name} | {d_name}] Profiler window={tp.get('window_steps')} "
                        f"self_cuda={tp.get('total_self_cuda_ms')}ms self_cpu={tp.get('total_self_cpu_ms')}ms ({ops_key}):"
                    )
                    for op in ops:
                        sv = op.get("self_cuda_ms", op.get("self_cpu_ms", 0.0))
                        tv = op.get("total_cuda_ms", op.get("total_cpu_ms", 0.0))
                        prof_lines.append(
                            f"  {op['name']:<48} calls={op['calls']:<5} self={sv:>7.3f}ms ({op['pct']:>4.1f}%) total={tv:>7.3f}ms"
                        )

    diag_lines = _diagnose_inference(ab_records, is_cuda=is_cuda)
    overall_rc = 0 if (unit_rc == 0 and mig_rc == 0 and not any_fail) else 1
    builder = TextReportBuilder(
        phase_title="PHASE 3.6 EXPERIMENT REPORT: INFERENCE / RUNNING OPTIMIZATION",
        repo_root=REPO_ROOT,
    )
    builder.add_header({
        "Google Colab": fp.is_colab,
        "Backend / GPU": (
            f"{fp.effective_backend} / {fp.primary_gpu.name} ({fp.primary_gpu.sm_name})"
            if fp.primary_gpu
            else fp.effective_backend
        ),
        "Unit Tests": f"{unit_runner} (exit_code={unit_rc})",
        "Migrate Tests": f"{mig_runner} (exit_code={mig_rc})",
        "Overall Experiment": "PASS" if overall_rc == 0 else "FAIL",
    })
    builder.add_table(
        "INFERENCE LATENCY (BS=1) & THROUGHPUT (BS=64) ABLATION",
        ["Case", "Variant", "Step0 ms", "Med ms", "P95 ms", "P99 ms", "Steady Samp/s", "E2E s", "Speedup", "VRAM MB (d)", "MaxRelDiff", "Status"],
        ab_rows,
    )
    builder.add_table(
        "DEEP DEBUG: INFERENCE PHASE BREAKDOWN & LATENCY JITTER",
        ["Case", "Mode", "Step0 ms", "Steady ms", "Min/Max/Std", "Host/Data (%)", "Model Fwd (%)"],
        debug_rows,
    )
    if mem_rows:
        builder.add_table(
            "DEEP DEBUG: CUDA ALLOCATOR & VRAM HEADROOM",
            ["Case", "Mode", "Alloc MB", "Resv MB", "Ovhead MB", "Headroom MB", "Util %", "Retries"],
            mem_rows,
        )
    builder.add_lines_section("AUTOMATED INFERENCE DIAGNOSIS", diag_lines)
    builder.add_lines_section("WARMUP VS STEADY-STATE STEP TRACE", trace_lines)
    builder.add_lines_section("TOP PYTORCH PROFILER OPS", prof_lines)
    builder.add_lines_section("INFERENCE HOOK & FALLBACK LOGS", hook_lines)
    builder.add_lines_section(f"MIGRATE TEST OUTPUT ({mig_runner})", mig_out.splitlines())
    builder.add_lines_section(f"UNIT TEST OUTPUT ({unit_runner})", unit_out.splitlines())
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if overall_rc == 0 else "FAILED_NEEDS_ATTENTION",
        note="Verify Inference Latency (bs=1), CUDA Graphs, Throughput (bs=64), Top Ops, and FP8 parity on Google Colab before Phase 3.7",
    )

    out_path = builder.write("phase-3.6", report_dir=args.report_dir)
    print(builder.render())
    print(f"[SAVED] Text report written to: {out_path}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
