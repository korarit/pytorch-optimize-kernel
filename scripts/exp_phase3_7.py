#!/usr/bin/env python3
"""Phase 3.7 Experiment Runner: Custom Triton Kernels Micro-Benchmark & E2E Transformer A/B."""

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
from typing import Any, Callable

import torch

BENCH_DIR = Path(__file__).resolve().parent.parent / "pipline-for-test"
REPO_ROOT = BENCH_DIR.parent
for _p in (str(REPO_ROOT), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench import WORKLOADS_DIR, WORKLOAD_SPECS, _extract_workload_summary
from src.common.text_report import TextReportBuilder
from src.detect import collect_fingerprint
from src.kernels import (
    HAS_TRITON,
    ref_cross_entropy,
    ref_layernorm,
    ref_rmsnorm,
    ref_swiglu,
    triton_cross_entropy,
    triton_layernorm,
    triton_rmsnorm,
    triton_swiglu,
)
from telemetry import RunRecord, check_parity, compute_speedup, compute_vram_delta, merge_summary_and_telemetry


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


def _bench_op(fn: Callable[[], torch.Tensor], *, is_cuda: bool, bytes_moved: int, iters: int = 30) -> tuple[float, float]:
    # Warmup
    for _ in range(5):
        out = fn()
        if out.requires_grad:
            out.sum().backward()
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        if out.requires_grad:
            out.sum().backward()
    if is_cuda:
        torch.cuda.synchronize()
    elapsed_ms = ((time.perf_counter() - t0) * 1000.0) / iters
    gb_per_s = (bytes_moved / (1024.0 ** 3)) / max(1e-9, elapsed_ms / 1000.0)
    return round(elapsed_ms, 3), round(gb_per_s, 1)


def _run_microbenchmarks(is_cuda: bool) -> list[list[object]]:
    device = torch.device("cuda" if is_cuda else "cpu")
    dtype = torch.bfloat16 if is_cuda else torch.float32
    elem_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    rows: list[list[object]] = []

    # 1. RMSNorm (M=2048, N=768)
    x_rms = torch.randn(2048, 768, device=device, dtype=dtype, requires_grad=True)
    w_rms = torch.ones(768, device=device, dtype=dtype, requires_grad=True)
    rms_bytes = 2 * x_rms.numel() * elem_bytes * 2  # fwd+bwd approx
    eager_ms, eager_bw = _bench_op(lambda: ref_rmsnorm(x_rms, w_rms), is_cuda=is_cuda, bytes_moved=rms_bytes)
    comp_rms = torch.compile(ref_rmsnorm) if is_cuda else ref_rmsnorm
    comp_ms, comp_bw = _bench_op(lambda: comp_rms(x_rms, w_rms), is_cuda=is_cuda, bytes_moved=rms_bytes)
    tri_ms, tri_bw = _bench_op(lambda: triton_rmsnorm(x_rms, w_rms), is_cuda=is_cuda, bytes_moved=rms_bytes)
    max_diff = round(float((ref_rmsnorm(x_rms, w_rms) - triton_rmsnorm(x_rms, w_rms)).abs().max().item()), 6)
    rows.append(["RMSNorm (2048x768)", f"{eager_ms} ({eager_bw}GB/s)", f"{comp_ms} ({comp_bw}GB/s)", f"{tri_ms} ({tri_bw}GB/s)", max_diff])

    # 2. LayerNorm (M=2048, N=1024)
    x_ln = torch.randn(2048, 1024, device=device, dtype=dtype, requires_grad=True)
    w_ln = torch.ones(1024, device=device, dtype=dtype, requires_grad=True)
    b_ln = torch.zeros(1024, device=device, dtype=dtype, requires_grad=True)
    ln_bytes = 2 * x_ln.numel() * elem_bytes * 2
    eager_ms, eager_bw = _bench_op(lambda: ref_layernorm(x_ln, w_ln, b_ln), is_cuda=is_cuda, bytes_moved=ln_bytes)
    comp_ln = torch.compile(ref_layernorm) if is_cuda else ref_layernorm
    comp_ms, comp_bw = _bench_op(lambda: comp_ln(x_ln, w_ln, b_ln), is_cuda=is_cuda, bytes_moved=ln_bytes)
    tri_ms, tri_bw = _bench_op(lambda: triton_layernorm(x_ln, w_ln, b_ln), is_cuda=is_cuda, bytes_moved=ln_bytes)
    max_diff = round(float((ref_layernorm(x_ln, w_ln, b_ln) - triton_layernorm(x_ln, w_ln, b_ln)).abs().max().item()), 6)
    rows.append(["LayerNorm (2048x1024)", f"{eager_ms} ({eager_bw}GB/s)", f"{comp_ms} ({comp_bw}GB/s)", f"{tri_ms} ({tri_bw}GB/s)", max_diff])

    # 3. CrossEntropy (M=2048, V=2048)
    logits = torch.randn(2048, 2048, device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(0, 2048, (2048,), device=device)
    ce_bytes = logits.numel() * elem_bytes * 2
    eager_ms, eager_bw = _bench_op(lambda: ref_cross_entropy(logits, targets), is_cuda=is_cuda, bytes_moved=ce_bytes)
    comp_ce = torch.compile(ref_cross_entropy) if is_cuda else ref_cross_entropy
    comp_ms, comp_bw = _bench_op(lambda: comp_ce(logits, targets), is_cuda=is_cuda, bytes_moved=ce_bytes)
    tri_ms, tri_bw = _bench_op(lambda: triton_cross_entropy(logits, targets), is_cuda=is_cuda, bytes_moved=ce_bytes)
    max_diff = round(float((ref_cross_entropy(logits, targets) - triton_cross_entropy(logits, targets)).abs().max().item()), 6)
    rows.append(["CrossEntropy (2048x2048)", f"{eager_ms} ({eager_bw}GB/s)", f"{comp_ms} ({comp_bw}GB/s)", f"{tri_ms} ({tri_bw}GB/s)", max_diff])

    # 4. SwiGLU (2048x3072)
    g_sw = torch.randn(2048, 3072, device=device, dtype=dtype, requires_grad=True)
    u_sw = torch.randn(2048, 3072, device=device, dtype=dtype, requires_grad=True)
    sw_bytes = 3 * g_sw.numel() * elem_bytes * 2
    eager_ms, eager_bw = _bench_op(lambda: ref_swiglu(g_sw, u_sw), is_cuda=is_cuda, bytes_moved=sw_bytes)
    comp_sw = torch.compile(ref_swiglu) if is_cuda else ref_swiglu
    comp_ms, comp_bw = _bench_op(lambda: comp_sw(g_sw, u_sw), is_cuda=is_cuda, bytes_moved=sw_bytes)
    tri_ms, tri_bw = _bench_op(lambda: triton_swiglu(g_sw, u_sw), is_cuda=is_cuda, bytes_moved=sw_bytes)
    max_diff = round(float((ref_swiglu(g_sw, u_sw) - triton_swiglu(g_sw, u_sw)).abs().max().item()), 6)
    rows.append(["SwiGLU (2048x3072)", f"{eager_ms} ({eager_bw}GB/s)", f"{comp_ms} ({comp_bw}GB/s)", f"{tri_ms} ({tri_bw}GB/s)", max_diff])

    return rows


def _run_e2e_variant(
    *,
    scale_name: str,
    batch_size: int,
    extra_args: list[str],
    steps: int,
    device: str,
    variant_name: str,
    profile: str,
    is_baseline: bool = False,
    compile_enabled: bool = False,
    overrides: list[str] | None = None,
) -> RunRecord:
    script_path = WORKLOADS_DIR / WORKLOAD_SPECS["transformer_lm"]["script"]
    wl_args = ["--steps", str(steps), "--batch-size", str(batch_size), "--seed", "42", "--device", device, *extra_args]
    env = os.environ.copy()

    with tempfile.TemporaryDirectory() as tmp:
        tel_path = Path(tmp) / "tel.json"
        if is_baseline:
            cmd = [sys.executable, str(script_path), *wl_args]
        else:
            env["PTOPT_TELEMETRY_OUT"] = str(tel_path)
            if device == "cpu" and compile_enabled:
                env["PTOPT_COMPILE_CPU"] = "1"
            ovr_flags: list[str] = []
            for item in (overrides or []):
                ovr_flags.extend(["--override", item])
            cmd = [
                sys.executable,
                "-m",
                "src.cli.main",
                "--profile",
                profile,
                *(["--override", "compile.enabled=true"] if compile_enabled else ["--no-compile"]),
                "--no-dataloader-tune",
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
        return merge_summary_and_telemetry(
            workload=f"transformer_lm:{scale_name}",
            mode=variant_name,
            cache_state="none",
            exit_code=res.returncode,
            summary=summary,
            telemetry=tel_data,
            wall_sec=wall,
            error=err,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3.7 Custom Triton Kernels & E2E Transformer Experiment")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=15)
    args = parser.parse_args()

    fp = collect_fingerprint()
    target_device = "cuda" if fp.effective_backend == "cuda" else "cpu"
    is_cuda = target_device == "cuda"
    steps = args.steps if is_cuda else min(args.steps, 5)

    unit_runner, unit_rc, unit_out = _run_test_file("tests/unit/test_kernels_parity.py")
    mig_runner, mig_rc, mig_out = _run_test_file("tests/migrate/test_phase3_7_kernels.py")

    micro_rows = _run_microbenchmarks(is_cuda=is_cuda)

    scales = (
        [
            ("small", 16, ["--d-model", "256", "--n-heads", "4", "--n-layers", "4", "--seq-len", "128"]),
            ("large", 16, ["--d-model", "768", "--n-heads", "12", "--n-layers", "12", "--seq-len", "512"]),
        ]
        if is_cuda
        else [("small", 8, ["--d-model", "128", "--n-heads", "4", "--n-layers", "2", "--seq-len", "64"])]
    )

    e2e_rows: list[list[object]] = []
    hook_lines: list[str] = []
    any_fail = False

    for scale_name, bs, extra in scales:
        variants: list[tuple[str, str, bool, bool, list[str], str]] = [
            ("baseline", "safe", True, False, [], "safe"),
            ("3.7a_kernels_eager", "aggressive", False, False, ["kernels.enabled=true", "compile.mode=default"], "aggressive"),
            ("3.7b_compile_only", "balanced", False, True, ["kernels.enabled=false"], "balanced"),
            ("3.7c_aggressive_full", "aggressive", False, True, ["kernels.enabled=true", "compile.mode=default"], "aggressive"),
        ]
        base_rec: RunRecord | None = None
        for var_name, prof, is_base, comp_en, ovr, parity_prof in variants:
            rec = _run_e2e_variant(
                scale_name=scale_name,
                batch_size=bs,
                extra_args=extra,
                steps=steps,
                device=target_device,
                variant_name=var_name,
                profile=prof,
                is_baseline=is_base,
                compile_enabled=comp_en,
                overrides=ovr,
            )
            if is_base:
                base_rec = rec
                continue
            assert base_rec is not None
            if base_rec.exit_code != 0 or rec.exit_code != 0:
                any_fail = True
                e2e_rows.append([f"transformer:{scale_name}", var_name, "-", "-", "-", "0.00x", "N/A", "None", "FAIL"])
                hook_lines.append(f"[ERROR] {scale_name} ({var_name}): {rec.error or base_rec.error}")
            else:
                parity = check_parity(base_rec.losses, rec.losses, profile=parity_prof)
                speedup = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                vram_d = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                status = "PASS" if parity.passed else "FAIL"
                if not parity.passed:
                    any_fail = True
                e2e_rows.append([
                    f"transformer:{scale_name}",
                    var_name,
                    f"{base_rec.step_stats.median:.2f}",
                    f"{rec.step_stats.median:.2f}",
                    f"{rec.step_stats.p95:.2f}",
                    f"{speedup:.2f}x",
                    f"{base_rec.peak_allocated_mb:.1f}/{rec.peak_allocated_mb:.1f} ({vram_d:+.1f})",
                    parity.max_rel_diff,
                    status,
                ])
                for ev in rec.fallback_events:
                    if str(ev.get("pass_name", "")).startswith("kernels") or ev.get("status") != "APPLIED":
                        hook_lines.append(
                            f"[{ev.get('status')}] {scale_name} ({var_name}) -> {ev.get('pass_name')}: {ev.get('reason')}"
                        )

    overall_rc = 0 if (unit_rc == 0 and mig_rc == 0 and not any_fail) else 1
    builder = TextReportBuilder(
        phase_title="PHASE 3.7 EXPERIMENT REPORT: CUSTOM TRITON KERNELS & CAPABILITY GATE",
        repo_root=REPO_ROOT,
    )
    builder.add_header({
        "Google Colab": fp.is_colab,
        "Backend / GPU": (
            f"{fp.effective_backend} / {fp.primary_gpu.name} ({fp.primary_gpu.sm_name})"
            if fp.primary_gpu
            else fp.effective_backend
        ),
        "Triton Installed": HAS_TRITON,
        "Unit Tests (Parity)": f"{unit_runner} (exit_code={unit_rc})",
        "Migrate Tests": f"{mig_runner} (exit_code={mig_rc})",
        "Overall Experiment": "PASS" if overall_rc == 0 else "FAIL",
    })
    builder.add_table(
        "KERNEL MICRO-BENCHMARK (PYTORCH EAGER VS TORCH.COMPILE VS CUSTOM TRITON)",
        ["Kernel (Shape)", "Eager ms (BW)", "Compile ms (BW)", "Triton ms (BW)", "MaxAbsDiff"],
        micro_rows,
    )
    builder.add_table(
        "END-TO-END TRANSFORMER_LM ABLATION (KERNELS VS COMPILE)",
        ["Workload", "Variant", "Base ms", "Wrap ms", "Wrap P95", "Speedup", "VRAM MB (B/W/d)", "MaxRelDiff", "Status"],
        e2e_rows,
    )
    builder.add_lines_section("KERNEL CAPABILITY GATE & FALLBACK LOGS", hook_lines)
    builder.add_lines_section(f"MIGRATE TEST OUTPUT ({mig_runner})", mig_out.splitlines())
    builder.add_lines_section(f"UNIT TEST OUTPUT ({unit_runner})", unit_out.splitlines())
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if overall_rc == 0 else "FAILED_NEEDS_ATTENTION",
        note="Verify Triton kernel micro-benchmarks vs torch.compile and E2E parity on Google Colab before finalizing Phase 3 rules",
    )

    out_path = builder.write("phase-3.7", report_dir=args.report_dir)
    print(builder.render())
    print(f"[SAVED] Text report written to: {out_path}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
