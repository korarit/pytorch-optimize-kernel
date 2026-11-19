#!/usr/bin/env python3
"""Phase 5.1 Verification (Phase 3-Grade Deep Debug): Tiny Dynamic-Shape Compile & AMP Bypass."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent / "pipline-for-test"
REPO_ROOT = BENCH_DIR.parent
for _p in (str(REPO_ROOT), str(BENCH_DIR), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase5_report_helper import (
    extract_allocator_row,
    extract_phase_breakdown_row,
    extract_step0_and_p99,
    format_step_trace_block,
    format_torch_profiler_block,
    run_workload_variant,
)
from ptopt.common.text_report import TextReportBuilder
from ptopt.config import detect_runtime_cache_key
from ptopt.detect import collect_fingerprint
from telemetry import RunRecord, check_parity, compute_speedup, compute_vram_delta


def _run_unit_file(rel_path: str) -> tuple[bool, float, str, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / rel_path), "-v"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    dt = time.perf_counter() - t0
    out = (proc.stdout + "\n" + proc.stderr).strip()
    ran = next((ln.strip() for ln in reversed(out.splitlines()) if ln.strip().startswith("Ran ")), f"exit={proc.returncode}")
    return proc.returncode == 0, dt, ran, out


def main() -> int:
    fp = collect_fingerprint()
    device = "cuda" if fp.effective_backend == "cuda" else "cpu"
    is_cuda = device == "cuda"
    cache_key = detect_runtime_cache_key().cache_key

    # 1. Targeted unit suites (executed by file path to avoid site-packages shadowing)
    ok_comp, dt_comp, ran_comp, out_comp = _run_unit_file("tests/unit/test_pass_compile.py")
    ok_amp, dt_amp, ran_amp, out_amp = _run_unit_file("tests/unit/test_pass_optimizer_amp.py")
    all_ok = ok_comp and ok_amp

    # 2. A/B Ablation Matrix on edge_cases + transformer_lm regression
    cases: list[tuple[str, str, int, int, list[str]]] = (
        [
            ("edge_dynamic_only", "edge_cases", 15, 8, ["--case", "dynamic_shape"]),
            ("edge_all", "edge_cases", 15, 8, ["--case", "all"]),
            ("transformer_lm", "transformer_lm", 15, 16, ["--d-model", "512", "--n-heads", "8", "--n-layers", "6", "--seq-len", "256"]),
        ]
        if is_cuda
        else [
            ("edge_dynamic_only", "edge_cases", 6, 8, ["--case", "dynamic_shape"]),
            ("edge_all", "edge_cases", 6, 8, ["--case", "all"]),
            ("transformer_lm", "transformer_lm", 4, 4, []),
        ]
    )

    ablation_rows: list[list[object]] = []
    event_notes: list[str] = []
    step_traces: list[str] = []
    records: dict[tuple[str, str], RunRecord] = {}

    with tempfile.TemporaryDirectory() as cache_root:
        for label, wl, steps, bs, extra in cases:
            wl_cache = Path(cache_root) / label
            wl_cache.mkdir(parents=True, exist_ok=True)
            base_rec, base_sum, base_tel = run_workload_variant(
                case_label=label,
                wl_name=wl,
                steps=steps,
                device=device,
                profile="safe",
                variant_name="baseline",
                extra_args=extra,
                batch_size=bs,
                is_baseline=True,
            )
            records[(label, "baseline")] = base_rec
            step_traces.extend(format_step_trace_block(label, "baseline", base_sum, base_tel))

            for prof in ("safe", "balanced", "aggressive"):
                ovr = ["compile.mode=default"] if (prof == "aggressive" and is_cuda) else []
                rec, summ, tel = run_workload_variant(
                    case_label=label,
                    wl_name=wl,
                    steps=steps,
                    device=device,
                    profile=prof,
                    variant_name=prof,
                    extra_args=extra,
                    batch_size=bs,
                    overrides=ovr,
                    cache_dir=wl_cache,
                )
                records[(label, prof)] = rec
                if base_rec.exit_code != 0 or rec.exit_code != 0:
                    all_ok = False
                    ablation_rows.append([label, prof, "-", f"{base_rec.step_stats.median:.2f}", "-", "-", "-", "0.00x", "N/A", "N/A", "FAIL"])
                    event_notes.append(f"[ERROR] {label} ({prof}): {rec.error or base_rec.error}")
                    continue

                step0_w, p99_w = extract_step0_and_p99(summ, tel)
                parity = check_parity(base_rec.losses, rec.losses, profile=prof)
                speedup = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                vram_d = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                if not parity.passed:
                    all_ok = False

                ablation_rows.append([
                    label,
                    prof,
                    f"{step0_w:.1f}",
                    f"{base_rec.step_stats.median:.2f}",
                    f"{rec.step_stats.median:.2f}",
                    f"{rec.step_stats.p95:.2f}",
                    f"{p99_w:.2f}",
                    f"{base_rec.end_to_end_sec:.2f}/{rec.end_to_end_sec:.2f}s",
                    f"{speedup:.2f}x",
                    f"{base_rec.peak_allocated_mb:.1f}/{rec.peak_allocated_mb:.1f} ({vram_d:+.1f})",
                    parity.max_rel_diff,
                    "PASS" if parity.passed else "FAIL",
                ])
                step_traces.extend(format_step_trace_block(label, prof, summ, tel))
                for ev in rec.fallback_events:
                    pname = str(ev.get("pass_name", ""))
                    if pname in ("compile.auto", "amp.auto"):
                        event_notes.append(f"[{ev.get('status')}] {label} ({prof}) -> {pname}: {ev.get('reason')}")

        # 3. Deep Debug + PyTorch Profiler pass on edge_dynamic_only & transformer_lm
        phase_rows: list[list[object]] = []
        alloc_rows: list[list[object]] = []
        prof_lines: list[str] = []
        dbg_steps = 8 if is_cuda else 5
        prof_start = 2 if is_cuda else 1
        prof_cnt = 3

        dbg_targets = [cases[0], cases[-1]]
        for label, wl, _steps, bs, extra in dbg_targets:
            wl_cache = Path(cache_root) / f"dbg_{label}"
            wl_cache.mkdir(parents=True, exist_ok=True)
            dbg_modes: list[tuple[str, str, bool, list[str]]] = [
                ("baseline_dbg", "safe", True, []),
                ("safe_dbg", "safe", False, []),
                ("balanced_dbg", "balanced", False, []),
                ("aggressive_dbg", "aggressive", False, ["compile.mode=default"] if is_cuda else []),
            ]
            for d_name, prof, dis_passes, ovr in dbg_modes:
                rec_d, summ_d, tel_d = run_workload_variant(
                    case_label=label,
                    wl_name=wl,
                    steps=dbg_steps,
                    device=device,
                    profile=prof,
                    variant_name=d_name,
                    extra_args=extra,
                    batch_size=bs,
                    disable_builtin_passes=dis_passes,
                    overrides=ovr,
                    cache_dir=None if dis_passes else wl_cache,
                    debug_profile=True,
                    torch_profile=True,
                    prof_start=prof_start,
                    prof_steps=prof_cnt,
                )
                if rec_d.exit_code != 0 or not tel_d:
                    event_notes.append(f"[ERROR] debug {label} ({d_name}): {rec_d.error}")
                    continue
                pb_row = extract_phase_breakdown_row(label, d_name, tel_d)
                if pb_row:
                    phase_rows.append(pb_row)
                mem_row = extract_allocator_row(label, d_name, tel_d)
                if mem_row:
                    alloc_rows.append(mem_row)
                prof_lines.extend(format_torch_profiler_block(label, d_name, tel_d))
                step_traces.extend(format_step_trace_block(label, d_name, summ_d, tel_d))

    # 4. Automated root-cause diagnosis
    diag_lines: list[str] = []
    for label, _, _, _, _ in cases:
        b = records.get((label, "baseline"))
        s = records.get((label, "safe"))
        bal = records.get((label, "balanced"))
        agg = records.get((label, "aggressive"))
        if not b or not s or not bal or not agg:
            continue
        sp_s = compute_speedup(b.step_stats.median, s.step_stats.median)
        sp_b = compute_speedup(b.step_stats.median, bal.step_stats.median)
        sp_a = compute_speedup(b.step_stats.median, agg.step_stats.median)
        vd_b = compute_vram_delta(b.peak_allocated_mb, bal.peak_allocated_mb)
        diag_lines.append(
            f"- {label}: base={b.step_stats.median:.2f}ms | safe={s.step_stats.median:.2f}ms ({sp_s:.2f}x) "
            f"| balanced={bal.step_stats.median:.2f}ms ({sp_b:.2f}x, VRAM {vd_b:+.1f}MB) "
            f"| aggressive={agg.step_stats.median:.2f}ms ({sp_a:.2f}x)"
        )

    gpu_name = fp.primary_gpu.name if fp.primary_gpu else "CPU-only"
    report = TextReportBuilder("PHASE 5.1 REPORT: TINY DYNAMIC-SHAPE COMPILE & AMP BYPASS (DEEP DEBUG)", REPO_ROOT)
    report.add_header({
        "Google Colab": fp.is_colab,
        "GPU / Backend": f"{gpu_name} | {device} (sm_{fp.sm_int})",
        "Python / PyTorch": f"{fp.host.python_version} | {fp.host.torch_version}",
        "CUDA / Triton": f"{fp.host.cuda_version or 'N/A'} | {fp.host.triton_version or 'N/A'}",
        "Runtime Cache Key": cache_key,
        "Unit (compile)": f"{ran_comp} ({dt_comp:.2f}s) -> {'PASS' if ok_comp else 'FAIL'}",
        "Unit (amp/optim)": f"{ran_amp} ({dt_amp:.2f}s) -> {'PASS' if ok_amp else 'FAIL'}",
        "Overall Phase 5.1": "PASS" if all_ok else "FAIL",
    })
    report.add_table(
        "DYNAMIC-SHAPE BYPASS & PERCENTILE ABLATION MATRIX",
        ["Case", "Profile", "Step0 ms", "Base ms", "Wrap ms", "Wrap P95", "Wrap P99", "E2E s (B/W)", "Speedup", "VRAM MB (B/W/d)", "MaxRelDiff", "Status"],
        ablation_rows,
    )
    report.add_table(
        "DEEP DEBUG: STEP PHASE BREAKDOWN (STEADY MEAN MS & SHARE %)",
        ["Case", "Mode", "Step0 ms", "Steady ms", "Data/Other", "Forward", "Loss/Clip", "Backward", "Optim"],
        phase_rows,
    )
    if alloc_rows:
        report.add_table(
            "DEEP DEBUG: CUDA ALLOCATOR & VRAM HEADROOM",
            ["Case", "Mode", "Alloc MB", "Resv MB", "Ovhead MB", "Headroom", "Util %", "Retries"],
            alloc_rows,
        )
    report.add_lines_section(
        "TOP PYTORCH PROFILER OPS (BASELINE VS SAFE VS BALANCED VS AGGRESSIVE)",
        prof_lines if prof_lines else ["(no profiler ops captured)"],
    )
    report.add_lines_section("WARMUP VS STEADY STEP TRACE", step_traces)
    report.add_lines_section("AUTOMATED DIAGNOSIS", diag_lines)
    report.add_lines_section(
        "COMPILE & AMP DECISION AUDIT (`compile.auto` / `amp.auto`)",
        event_notes if event_notes else ["(no compile.auto or amp.auto events triggered on this backend)"],
    )
    if not all_ok:
        err_lines: list[str] = []
        if not ok_comp:
            err_lines.extend(out_comp[-1500:].splitlines())
        if not ok_amp:
            err_lines.extend(out_amp[-1500:].splitlines())
        if err_lines:
            report.add_lines_section("UNIT FAILURE TRACE", err_lines)

    report.add_verification_footer(
        status="PENDING_USER_REVIEW" if all_ok else "NEEDS_FIX",
        note="Verify Phase 5.1 speedup, zero +47MB bloat, and Deep Debug profiler traces on Colab L4 before Plan 5.2",
    )

    out_path = report.write(name="phase-5.1")
    print(report.render())
    print(f"Report written to: {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
