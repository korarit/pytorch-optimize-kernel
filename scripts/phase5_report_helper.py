"""Shared Phase 3-grade Deep Debug, Torch Profiler, Allocator, and Step Trace helper for Phase 5."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

BENCH_DIR = Path(__file__).resolve().parent.parent / "pipline-for-test"
REPO_ROOT = BENCH_DIR.parent
for _p in (str(REPO_ROOT), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench import WORKLOAD_SPECS, WORKLOADS_DIR, _extract_workload_summary
from telemetry import RunRecord, merge_summary_and_telemetry, percentile


def run_workload_variant(
    *,
    case_label: str,
    wl_name: str,
    steps: int,
    device: str,
    profile: str,
    variant_name: str,
    extra_args: Sequence[str] = (),
    batch_size: int | None = None,
    is_baseline: bool = False,
    disable_builtin_passes: bool = False,
    overrides: Sequence[str] = (),
    cache_dir: Path | None = None,
    debug_profile: bool = False,
    torch_profile: bool = False,
    prof_start: int = 2,
    prof_steps: int = 3,
    extra_env: dict[str, str] | None = None,
) -> tuple[RunRecord, dict[str, Any] | None, dict[str, Any] | None]:
    """Run one workload configuration in a clean subprocess and return (RunRecord, summary, telemetry)."""
    spec = WORKLOAD_SPECS[wl_name]
    script_path = WORKLOADS_DIR / spec["script"]
    bs = batch_size if batch_size is not None else int(spec["batch_size"])
    extra = list(extra_args) if extra_args else list(spec["extra"])
    wl_args = ["--steps", str(steps), "--batch-size", str(bs), "--seed", "42", "--device", device, *extra]

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

            ovr_flags: list[str] = []
            for item in overrides:
                ovr_flags.extend(["--override", item])
            if cache_dir is not None:
                ovr_flags.extend(["--override", f"compile.cache_dir={cache_dir}"])
            cmd = [
                sys.executable,
                "-m",
                "ptopt.cli.main",
                "--profile",
                profile,
                *ovr_flags,
                "--",
                sys.executable,
                str(script_path),
                *wl_args,
            ]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
        wall = time.perf_counter() - t0
        summary = _extract_workload_summary(proc.stdout)
        tel_data = json.loads(tel_path.read_text(encoding="utf-8")) if tel_path.is_file() else None
        err = (proc.stderr.strip() or proc.stdout.strip())[-350:] if proc.returncode != 0 else None

        rec = merge_summary_and_telemetry(
            workload=case_label,
            mode=variant_name,
            cache_state="warm" if cache_dir else "none",
            exit_code=proc.returncode,
            summary=summary,
            telemetry=tel_data,
            wall_sec=wall,
            error=err,
        )
        return rec, summary, tel_data


def extract_step0_and_p99(summary: dict[str, Any] | None, tel_data: dict[str, Any] | None) -> tuple[float, float]:
    """Compute Step 0 latency (ms) and steady P99 latency (ms)."""
    src = (tel_data or {}).get("step_times_ms") or (summary or {}).get("step_times_ms") or []
    vals = [float(v) for v in src]
    if not vals:
        return 0.0, 0.0
    step0 = vals[0]
    steady = vals[1:] if len(vals) > 1 else vals
    return step0, percentile(steady, 99.0)


def extract_phase_breakdown_row(case_label: str, mode_label: str, tel_data: dict[str, Any] | None) -> list[str] | None:
    """Format `[Case, Mode, Step0 ms, Steady ms, Data/Other, Forward, Loss/Clip, Backward, Optim]`."""
    dbg = (tel_data or {}).get("debug_profile") or {}
    pb = dbg.get("phase_breakdown")
    if not pb:
        return None
    st = pb["steady_mean_ms"]
    sh = pb["steady_shares_pct"]
    return [
        case_label,
        mode_label,
        f"{pb['step_0_cold']['total_ms']:.2f}",
        f"{st['total_ms']:.2f}",
        f"{st['data_other_ms']:.2f} ({sh['data_other']}%)",
        f"{st['forward_ms']:.2f} ({sh['forward']}%)",
        f"{st['loss_clip_ms']:.2f} ({sh['loss_clip']}%)",
        f"{st['backward_ms']:.2f} ({sh['backward']}%)",
        f"{st['optim_ms']:.2f} ({sh['optim']}%)",
    ]


def extract_allocator_row(case_label: str, mode_label: str, tel_data: dict[str, Any] | None) -> list[object] | None:
    """Format `[Case, Mode, Alloc MB, Resv MB, Ovhead MB, Headroom, Util %, Retries]`."""
    dbg = (tel_data or {}).get("debug_profile") or {}
    mem = dbg.get("cuda_memory_debug")
    if not mem:
        return None
    return [
        case_label,
        mode_label,
        f"{mem['peak_allocated_mb']:.1f}",
        f"{mem['peak_reserved_mb']:.1f}",
        f"{mem['reserved_overhead_mb']:.1f}",
        f"{mem['headroom_mb']:.1f}",
        f"{mem['utilization_pct']:.1f}%",
        mem["num_alloc_retries"],
    ]


def format_torch_profiler_block(
    case_label: str,
    mode_label: str,
    tel_data: dict[str, Any] | None,
    *,
    top_n: int = 8,
) -> list[str]:
    """Render top CUDA/CPU PyTorch Profiler ops block cleanly."""
    tp = ((tel_data or {}).get("debug_profile") or {}).get("torch_profiler")
    if not tp:
        return []
    ops_key = "top_cuda_ops" if tp.get("top_cuda_ops") else "top_cpu_ops"
    ops = tp.get(ops_key, [])[:top_n]
    lines = [
        f"[{case_label} | {mode_label}] Profiler window={tp.get('window_steps')} "
        f"self_cuda={tp.get('total_self_cuda_ms')}ms self_cpu={tp.get('total_self_cpu_ms')}ms ({ops_key}):"
    ]
    for op in ops:
        sv = op.get("self_cuda_ms", op.get("self_cpu_ms", 0.0))
        tv = op.get("total_cuda_ms", op.get("total_cpu_ms", 0.0))
        lines.append(
            f"  {op['name']:<48} calls={op['calls']:<5} self={sv:>7.3f}ms ({op['pct']:>4.1f}%) total={tv:>7.3f}ms"
        )
    return lines


def format_step_trace_block(
    case_label: str,
    mode_label: str,
    summary: dict[str, Any] | None,
    tel_data: dict[str, Any] | None,
    *,
    max_steps: int = 8,
) -> list[str]:
    """Render per-step warmup vs steady latency trace in ms."""
    src = (tel_data or {}).get("step_times_ms") or (summary or {}).get("step_times_ms") or []
    vals = [float(v) for v in src]
    if not vals:
        return []
    shown = vals[:max_steps]
    steady = vals[1:] if len(vals) > 1 else vals
    steady_med = percentile(steady, 50.0)
    per_step_str = ", ".join(f"s{idx}={v:.2f}ms" for idx, v in enumerate(shown))
    suffix = " ..." if len(vals) > max_steps else ""
    return [f"[{case_label} | {mode_label}] step0={vals[0]:.2f}ms | steady_med={steady_med:.2f}ms | {per_step_str}{suffix}"]
