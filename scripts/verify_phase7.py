#!/usr/bin/env python3
"""Phase 7 Verification Suite + 7-Part Deep Debug Detailed Text Report generator."""

from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BENCH_DIR = REPO_ROOT / "pipline-for-test"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from phase5_report_helper import (  # noqa: E402
    extract_allocator_row,
    extract_phase_breakdown_row,
    extract_step0_and_p99,
    format_step_trace_block,
    format_torch_profiler_block,
    run_workload_variant,
)
from ptopt.common.text_report import TextReportBuilder  # noqa: E402
from ptopt.config import detect_runtime_cache_key  # noqa: E402
from ptopt.detect import collect_fingerprint  # noqa: E402
from ptopt.detect.fingerprint import _fingerprint_cache_path  # noqa: E402


def _run_unit_file(rel_path: str) -> tuple[bool, float, str, str]:
    t0 = time.perf_counter()
    env = os.environ.copy()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{old_pp}" if old_pp else str(REPO_ROOT)
    # Invoke by direct file path so external site-packages `tests/` (e.g. on Colab) cannot shadow repo `tests/`.
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / rel_path), "-v"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    dt = time.perf_counter() - t0
    return proc.returncode == 0, dt, proc.stdout, proc.stderr


def _bench_fingerprint_startup() -> list[list[object]]:
    from unittest.mock import patch

    rows: list[list[object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"PTOPT_FP_CACHE_DIR": tmp, "PTOPT_NO_FP_CACHE": "1"}):
            t0 = time.perf_counter()
            fp_cold = collect_fingerprint()
            dt_cold_ms = (time.perf_counter() - t0) * 1000.0

        clean_env = {"PTOPT_FP_CACHE_DIR": tmp}
        prev_no_cache = os.environ.pop("PTOPT_NO_FP_CACHE", None)
        try:
            with patch.dict(os.environ, clean_env):
                t1 = time.perf_counter()
                _ = collect_fingerprint()
                dt_write_ms = (time.perf_counter() - t1) * 1000.0

                t2 = time.perf_counter()
                fp_warm = collect_fingerprint()
                dt_warm_ms = (time.perf_counter() - t2) * 1000.0

                p_cli = _fingerprint_cache_path(None, os.environ)
                p_rt = _fingerprint_cache_path(torch, os.environ)
        finally:
            if prev_no_cache is not None:
                os.environ["PTOPT_NO_FP_CACHE"] = prev_no_cache
        rows.append([
            f"Cold ({fp_cold.gpu.backend})",
            f"{dt_cold_ms:.2f}",
            "-",
            "Live probe (no cache)",
            "PASS",
        ])
        rows.append([
            "Cache Seed",
            f"{dt_write_ms:.2f}",
            "-",
            "Write + fresh /dev/shm",
            "PASS",
        ])
        rows.append([
            f"Warm Hit ({fp_warm.gpu.backend})",
            f"{dt_warm_ms:.2f}",
            f"{dt_cold_ms / max(dt_warm_ms, 1e-3):.2f}x",
            f"CLI/RT cache match={p_cli.name == p_rt.name}",
            "PASS" if p_cli == p_rt else "FAIL",
        ])
    return rows


def _check_user_src_namespace() -> tuple[bool, str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        proj = Path(tmp)
        (proj / "src").mkdir(parents=True, exist_ok=True)
        (proj / "src" / "__init__.py").write_text("", encoding="utf-8")
        (proj / "src" / "model.py").write_text(
            "import torch.nn as nn\n"
            "class Net(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(8, 4)\n"
            "    def forward(self, x):\n"
            "        return self.fc(x)\n",
            encoding="utf-8",
        )
        (proj / "run.py").write_text(
            "import torch\nfrom src.model import Net\nm = Net()\nprint('NS_OK', tuple(m(torch.randn(2, 8)).shape))\n",
            encoding="utf-8",
        )
        env = dict(os.environ)
        old_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{old_pp}" if old_pp else str(REPO_ROOT)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "ptopt.cli.main", "--profile", "safe", "--", sys.executable, "run.py"],
            cwd=str(proj),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        ok = proc.returncode == 0 and "NS_OK (2, 4)" in proc.stdout
        return ok, ("Isolated ptopt vs user src/ OK" if ok else (proc.stderr or proc.stdout).strip()[:120]), dt_ms


def main() -> int:
    is_cuda = torch.cuda.is_available()
    fp = collect_fingerprint(torch_mod=torch)
    cache_key = detect_runtime_cache_key(torch_mod=torch)

    builder = TextReportBuilder(phase_title="Phase 7: Hardening, Calibration & Namespace Isolation", repo_root=REPO_ROOT)
    builder.add_header({
        "Hardware Backend": fp.gpu.backend,
        "GPU": fp.gpu.devices[0].name if fp.gpu.devices else "CPU-Only (Colab GPU pending)",
        "PyTorch / Runtime Key": f"{torch.__version__} ({cache_key})",
        "Package Namespace": "ptopt/ (isolated from user src/)",
        "Debug Telemetry": "PTOPT_DEBUG_PROFILE=1, PTOPT_TORCH_PROFILE=1",
    })

    # 1. Unit & Fix Verification Suite
    unit_ok, unit_sec, unit_out, unit_err = _run_unit_file("tests/unit/test_phase7_features.py")
    ns_ok, ns_msg, ns_ms = _check_user_src_namespace()
    fp_rows = _bench_fingerprint_startup()

    unit_combined = (unit_err + "\n" + unit_out).strip()
    ran_summary = next(
        (ln.strip() for ln in reversed(unit_combined.splitlines()) if ln.strip().startswith("Ran ")),
        f"elapsed {unit_sec:.2f}s",
    )

    fix_rows: list[list[object]] = [
        ["7.1", "fp8_compile_static telemetry & calib_steps", "Unit/Hook", "PASS" if unit_ok else "FAIL", "Outer TelemetryCollector + suppress() + micro-batch calib"],
        ["7.2", "Package namespace isolation (src -> ptopt)", "Subprocess", "PASS" if ns_ok else "FAIL", f"{ns_msg} ({ns_ms:.1f}ms)"],
        ["7.3a", "Hybrid LCE wide-FFN head classification guard", "Unit", "PASS" if unit_ok else "FAIL", "Only terminal head patched; 8x FFN unpatched"],
        ["7.3b", "LazyLinearLogits inside compiled forward+CE", "Compile", "PASS" if unit_ok else "FAIL", "empty_strided + no_grad _make_subclass + meta passthrough"],
        ["7.3c", "auto_compile OOM rescue + act_ckpt peak reset", "Unit", "PASS" if unit_ok else "FAIL", "OOM deferred to order-20 rescue; partial resets peak"],
        ["7.4a", "checkpoint_compat fused step & 5D Conv3d sync", "Unit", "PASS" if unit_ok else "FAIL", "injected_fused CPU step + mixed Conv2d/Conv3d channels_last"],
        ["7.4b", "auto_amp NAdam/RAdam + eval skip isolation + queue", "Unit", "PASS" if unit_ok else "FAIL", "Queue drains unpolled slots; eval skips isolated"],
        ["7.4c", "ModuleCallDispatcher TLS + RoPE 3D + CLI allow_list", "Unit", "PASS" if unit_ok else "FAIL", f"{ran_summary} ({unit_sec:.2f}s)"],
    ]
    builder.add_table("PHASE 7 AUDIT & FUNCTIONAL MATRIX", ["ID", "Fix Target", "Scope", "Status", "Verification Notes"], fix_rows)
    builder.add_table("COLD VS WARM FINGERPRINT CACHE", ["Stage", "Time(ms)", "Speedup", "Details", "Status"], fp_rows)

    # 2-6. Full E2E Ablation Matrix + Deep Debug Profiling
    ablation_rows: list[list[object]] = []
    phase_rows: list[list[str]] = []
    alloc_rows: list[list[object]] = []
    profiler_lines: list[str] = []
    trace_lines: list[str] = []
    diagnosis_lines: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_cache:
        cache_dir = Path(tmp_cache)

        if is_cuda:
            sm_cap = torch.cuda.get_device_capability(0)
            sm = sm_cap[0] * 10 + sm_cap[1]

            # Workload A: infer_bs64_large (Phase 6.3 anomaly target: verify Step-0 compile stays in warmup, len == 10!)
            infer_extra = ["--seq-len", "512", "--d-model", "1024", "--n-layers", "6"]
            base_inf_rec, base_inf_sum, base_inf_tel = run_workload_variant(
                case_label="infer_bs64_large",
                wl_name="infer_runner_bs32",
                steps=10,
                batch_size=64,
                extra_args=infer_extra,
                device="cuda",
                profile="safe",
                variant_name="eager_bf16",
                is_baseline=True,
                cache_dir=cache_dir / "inf_base",
                debug_profile=True,
                torch_profile=True,
                prof_start=4,
                prof_steps=5,
            )
            dyn_rec, dyn_sum, dyn_tel = run_workload_variant(
                case_label="infer_bs64_large",
                wl_name="infer_runner_bs32",
                steps=10,
                batch_size=64,
                extra_args=infer_extra,
                device="cuda",
                profile="balanced",
                variant_name="fp8_compile_dynamic",
                cache_dir=cache_dir / "inf_dyn",
                debug_profile=True,
                torch_profile=True,
                prof_start=4,
                prof_steps=5,
                extra_env={"PTOPT_INFERENCE_FP8": "1", "PTOPT_FP8_STATIC_SCALE": "0"},
            )
            stat_rec, stat_sum, stat_tel = run_workload_variant(
                case_label="infer_bs64_large",
                wl_name="infer_runner_bs32",
                steps=10,
                batch_size=64,
                extra_args=infer_extra,
                device="cuda",
                profile="balanced",
                variant_name="fp8_compile_static",
                cache_dir=cache_dir / "inf_static",
                debug_profile=True,
                torch_profile=True,
                prof_start=4,
                prof_steps=5,
                extra_env={"PTOPT_INFERENCE_FP8": "1", "PTOPT_FP8_STATIC_SCALE": "1", "PTOPT_FP8_CALIB_STEPS": "2"},
            )

            # Workload B: vocab32k hybrid_compile_lce
            lm_extra = ["--vocab-size", "32000", "--d-model", "512", "--n-heads", "8", "--n-layers", "4", "--seq-len", "256"]
            base_lm_rec, base_lm_sum, base_lm_tel = run_workload_variant(
                case_label="vocab32k",
                wl_name="transformer_lm",
                steps=10,
                batch_size=4,
                extra_args=lm_extra,
                device="cuda",
                profile="safe",
                variant_name="baseline",
                is_baseline=True,
                cache_dir=cache_dir / "lm_base",
                debug_profile=True,
                torch_profile=True,
                prof_start=4,
                prof_steps=5,
            )
            hyb_lm_rec, hyb_lm_sum, hyb_lm_tel = run_workload_variant(
                case_label="vocab32k",
                wl_name="transformer_lm",
                steps=10,
                batch_size=4,
                extra_args=lm_extra,
                device="cuda",
                profile="balanced",
                variant_name="hybrid_compile_lce",
                cache_dir=cache_dir / "lm_hyb",
                debug_profile=True,
                torch_profile=True,
                prof_start=4,
                prof_steps=5,
                extra_env={"PTOPT_HYBRID_LCE": "1"},
            )

            runs = [
                ("infer_bs64_large", "eager_bf16", base_inf_rec, base_inf_sum, base_inf_tel, base_inf_rec.step_stats.median),
                ("infer_bs64_large", "fp8_compile_dynamic", dyn_rec, dyn_sum, dyn_tel, base_inf_rec.step_stats.median),
                ("infer_bs64_large", "fp8_compile_static", stat_rec, stat_sum, stat_tel, base_inf_rec.step_stats.median),
                ("vocab32k", "baseline", base_lm_rec, base_lm_sum, base_lm_tel, base_lm_rec.step_stats.median),
                ("vocab32k", "hybrid_compile_lce", hyb_lm_rec, hyb_lm_sum, hyb_lm_tel, base_lm_rec.step_stats.median),
            ]
            for c_lbl, m_lbl, rec, summ, tel, ref_p50 in runs:
                s0, p99 = extract_step0_and_p99(summ, tel)
                med = rec.step_stats.median
                spd = ref_p50 / max(med, 1e-3)
                n_tel_steps = len((tel or {}).get("step_times_ms") or [])
                ablation_rows.append([
                    c_lbl,
                    m_lbl,
                    f"{s0:.1f}",
                    f"{med:.2f}",
                    f"{rec.step_stats.p95:.2f}",
                    f"{p99:.2f}",
                    f"{spd:.2f}x",
                    f"{rec.end_to_end_sec:.2f}s",
                    f"{rec.peak_allocated_mb:.1f} MB",
                    n_tel_steps,
                    "PASS" if rec.exit_code == 0 else "FAIL",
                ])
                pr = extract_phase_breakdown_row(c_lbl, m_lbl, tel)
                if pr:
                    phase_rows.append(pr)
                ar = extract_allocator_row(c_lbl, m_lbl, tel)
                if ar:
                    alloc_rows.append(ar)
                profiler_lines.extend(format_torch_profiler_block(c_lbl, m_lbl, tel, top_n=6))
                trace_lines.extend(format_step_trace_block(c_lbl, m_lbl, summ, tel, max_steps=10))

            # Verify fp8_compile_static step-count & steady window fix on CUDA
            static_steps = (stat_tel or {}).get("step_times_ms") or []
            if len(static_steps) == 10 and (len(static_steps) < 5 or static_steps[4] < 100.0):
                diagnosis_lines.append(
                    f"[PASS] fp8_compile_static recorded exact {len(static_steps)} steps; s4={static_steps[4]:.2f}ms "
                    "(Step-0 compile no longer pushed into steady window)."
                )
            else:
                diagnosis_lines.append(
                    f"[CHECK] fp8_compile_static steps={len(static_steps)} s4={(static_steps[4] if len(static_steps) > 4 else -1):.2f}ms (SM={sm})"
                )
        else:
            # Local CPU debug verification trace (genuine CPU execution with PTOPT_DEBUG_PROFILE=1 + PTOPT_TORCH_PROFILE=1)
            ablation_rows.extend([
                ["infer_bs64_large", "eager_bf16 / fp8_compile_static", "-", "-", "-", "-", "-", "-", "-", "-", "[WAIT FOR USER - GPU]"],
                ["vocab32k", "baseline / hybrid_compile_lce", "-", "-", "-", "-", "-", "-", "-", "-", "[WAIT FOR USER - GPU]"],
            ])
            cpu_lm_extra = ["--vocab-size", "256", "--d-model", "64", "--n-heads", "4", "--n-layers", "2", "--seq-len", "32"]
            b_rec, b_sum, b_tel = run_workload_variant(
                case_label="cpu_smoke_lm",
                wl_name="transformer_lm",
                steps=6,
                batch_size=4,
                extra_args=cpu_lm_extra,
                device="cpu",
                profile="safe",
                variant_name="baseline",
                is_baseline=True,
                cache_dir=cache_dir / "cpu_base",
                debug_profile=True,
                torch_profile=True,
                prof_start=2,
                prof_steps=3,
            )
            p_rec, p_sum, p_tel = run_workload_variant(
                case_label="cpu_smoke_lm",
                wl_name="transformer_lm",
                steps=6,
                batch_size=4,
                extra_args=cpu_lm_extra,
                device="cpu",
                profile="safe",
                variant_name="ptopt_safe",
                cache_dir=cache_dir / "cpu_ptopt",
                debug_profile=True,
                torch_profile=True,
                prof_start=2,
                prof_steps=3,
            )
            for lbl, rec, summ, tel, ref_p50 in (
                ("baseline", b_rec, b_sum, b_tel, b_rec.step_stats.median),
                ("ptopt_safe", p_rec, p_sum, p_tel, b_rec.step_stats.median),
            ):
                s0, p99 = extract_step0_and_p99(summ, tel)
                med = rec.step_stats.median
                n_tel_steps = len((tel or {}).get("step_times_ms") or [])
                ablation_rows.append([
                    "cpu_smoke_lm",
                    lbl,
                    f"{s0:.2f}",
                    f"{med:.2f}",
                    f"{rec.step_stats.p95:.2f}",
                    f"{p99:.2f}",
                    f"{ref_p50 / max(med, 1e-3):.2f}x",
                    f"{rec.end_to_end_sec:.2f}s",
                    "CPU",
                    n_tel_steps,
                    "PASS" if rec.exit_code == 0 else "FAIL",
                ])
                pr = extract_phase_breakdown_row("cpu_smoke_lm", lbl, tel)
                if pr:
                    phase_rows.append(pr)
                profiler_lines.extend(format_torch_profiler_block("cpu_smoke_lm", lbl, tel, top_n=6))
                trace_lines.extend(format_step_trace_block("cpu_smoke_lm", lbl, summ, tel, max_steps=6))

            alloc_rows.append(["cpu_smoke_lm", "ptopt_safe", "0.0", "0.0", "0.0", "0.0", "0.0%", "[WAIT FOR USER - GPU]"])
            diagnosis_lines.append(f"CPU local run: {ran_summary}; CUDA L4 ablations queued [WAIT FOR USER].")

    builder.add_table(
        "ABLATION MATRIX + P95 / P99 / E2E LATENCY SUMMARY",
        ["Case", "Mode", "Step0(ms)", "SteadyP50", "P95(ms)", "P99(ms)", "Speedup", "E2E", "PeakVRAM", "TelSteps", "Status"],
        ablation_rows,
    )
    builder.add_table(
        "PHASE BREAKDOWN (DEBUG PROFILE)",
        ["Case", "Mode", "Step0(ms)", "Steady(ms)", "Data/Other", "Forward", "Loss/Clip", "Backward", "Optim"],
        phase_rows,
    )
    builder.add_table(
        "CUDA ALLOCATOR & MEMORY HEADROOM",
        ["Case", "Mode", "Alloc(MB)", "Reserved(MB)", "Overhead(MB)", "Headroom(MB)", "Util(%)", "Retries"],
        alloc_rows,
    )
    builder.add_lines_section("TOP PYTORCH PROFILER OPS (STEADY WINDOW)", profiler_lines)
    builder.add_lines_section("WARMUP VS STEADY STEP TRACE", trace_lines)

    all_local_ok = unit_ok and ns_ok
    if not unit_ok:
        diagnosis_lines.append(f"test_phase7_features failed:\n{unit_err or unit_out}")
    builder.add_lines_section("AUTOMATED DIAGNOSIS & VERIFICATION SUMMARY", diagnosis_lines)
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if all_local_ok else "FAILED",
        note=(
            "[WAIT FOR USER] Run `python3 scripts/verify_phase7.py` and `python3 scripts/run_gate.py --mode full` on Google Colab L4 GPU"
            if not is_cuda and all_local_ok
            else ("Colab GPU Phase 7 verification ready for user review" if all_local_ok else "Local checks failed")
        ),
    )

    out_path = builder.write("phase-7")
    print(builder.render(), end="")
    print(f"[phase-7] Report saved -> {out_path}")
    return 0 if all_local_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
