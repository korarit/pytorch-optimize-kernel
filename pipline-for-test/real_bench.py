#!/usr/bin/env python3
"""Zero-code external GitHub pipeline A/B benchmark runner with structured artifact bundles and deep optimization logs."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
for _p in (str(REPO_ROOT), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ptopt.cli.explain import build_config_snapshot_dict
from ptopt.cli.main import _collect_overrides, parse_ptopt_argv
from ptopt.common.text_report import TextReportBuilder, format_report_filename
from ptopt.config import ConfigResolver
from ptopt.detect import HardwareFingerprint, collect_fingerprint
from ptopt.hooks.fallback_reporter import format_fallback_report
from telemetry import (
    PARITY_THRESHOLDS,
    StepStats,
    check_parity,
    compute_speedup,
    compute_step_stats,
    compute_vram_delta,
)

_TORCH_LOGS_SPEC = "graph_breaks,recompiles,perf_hints"


@dataclass(frozen=True)
class RealTargetSpec:
    """Benchmark specification for a single unmodified external repository pipeline."""

    name: str
    cwd: Path
    cmd: list[str]
    wrap_cmd: list[str] | None = None
    native_cmd: list[str] | None = None
    test_conflict: bool = False
    conflict_profile: str = "balanced"
    profiles: tuple[str, ...] = ("safe", "balanced")
    cache_modes: tuple[str, ...] = ("auto",)
    trials: int = 1
    metric_regexes: dict[str, str] = field(default_factory=dict)
    parity_mode: str = "auto"  # auto | loss | metric | exit_only
    metric_rel_tol: float = 0.05
    measure_pure_wall: bool = True
    diag_mode: str = "deep"  # deep | off
    stream_output: bool = False
    ptopt_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    timeout_sec: float = 3600.0


@dataclass(frozen=True)
class RealRunRecord:
    """Captured execution metrics and diagnostic logs for one arm."""

    target: str
    arm: str  # baseline | safe | balanced | aggressive | native_opt | conflict:<prof>
    cache_state: str  # none | cold | warm
    exit_code: int
    end_to_end_sec: float
    train_stats: StepStats
    eval_stats: StepStats
    peak_allocated_mb: float
    peak_reserved_mb: float
    losses: list[float] = field(default_factory=list)
    extracted_metrics: dict[str, float] = field(default_factory=dict)
    fallback_events: list[dict[str, Any]] = field(default_factory=list)
    pure_wall_sec: float | None = None
    nvsmi_peak_mb: float | None = None
    gpu_util_avg_pct: float | None = None
    debug_profile: dict[str, Any] = field(default_factory=dict)
    artifact_dir: str | None = None
    error: str | None = None

    def primary_step_stats(self) -> StepStats:
        """Return training step stats if optimizer steps ran, else eval/inference stats."""
        return self.train_stats if self.train_stats.steady_steps > 0 else self.eval_stats

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RealComparisonRow:
    """Normalized comparison of one non-baseline arm against its baseline."""

    target: str
    arm: str
    cache_state: str
    kind: str  # train | eval
    base_step_ms: float
    arm_step_ms: float
    arm_p95_ms: float
    step_speedup: float
    eval_base_ms: float
    eval_arm_ms: float
    eval_speedup: float
    base_e2e_s: float
    arm_e2e_s: float
    e2e_speedup: float
    pure_wall_base_s: float | None
    pure_wall_arm_s: float | None
    pure_wall_speedup: float | None
    base_vram_mb: float
    arm_vram_mb: float
    vram_delta_mb: float
    max_rel_diff: float | None
    metrics_summary: str
    audit_summary: str
    status: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_cmd_tokens(cmd: str | Sequence[str]) -> list[str]:
    tokens = shlex.split(cmd) if isinstance(cmd, str) else [str(t) for t in cmd]
    if not tokens:
        raise ValueError("Target command must not be empty.")
    if tokens[0] in ("python", "python3"):
        tokens[0] = sys.executable
    return tokens


def _flatten_ptopt_args(extra_ptopt_args: Sequence[str]) -> list[str]:
    out: list[str] = []
    for item in extra_ptopt_args:
        out.extend(shlex.split(item))
    return out


def _sanitize_slug(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip()).strip("_")
    return cleaned or "run"


def parse_metric_specs(items: Sequence[str]) -> dict[str, str]:
    """Parse `name=regex` items into a `{name: regex}` mapping."""
    out: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid --metric-regex '{raw}'; expected 'name=regex'.")
        name, pattern = raw.split("=", 1)
        key = name.strip()
        if not key or not pattern:
            raise ValueError(f"Invalid --metric-regex '{raw}'; name and pattern must be non-empty.")
        re.compile(pattern)
        out[key] = pattern
    return out


def parse_env_items(items: Sequence[str]) -> dict[str, str]:
    """Parse `KEY=VALUE` items into an environment mapping."""
    out: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid --env '{raw}'; expected 'KEY=VALUE'.")
        key, val = raw.split("=", 1)
        out[key.strip()] = val
    return out


def extract_metrics_from_output(text: str, regexes: Mapping[str, str]) -> dict[str, float]:
    """Extract the last finite numeric match for each configured metric regex."""
    found: dict[str, float] = {}
    for name, pattern in regexes.items():
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if not matches:
            continue
        last_val = matches[-1]
        if isinstance(last_val, tuple):
            last_val = next((x for x in last_val if x), "")
        try:
            num = float(last_val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(num):
            found[name] = num
    return found


def _resolve_cache_states(
    profile: str,
    requested: Sequence[str],
    resolver: ConfigResolver,
    fp: HardwareFingerprint,
) -> list[str]:
    if "auto" not in requested:
        return list(requested)
    cfg = resolver.resolve(fp, profile=profile, env={}).config
    uses_compile_cache = bool(cfg.compile.enabled or cfg.kernels.enabled or cfg.inference.cuda_graphs)
    return ["cold", "warm"] if uses_compile_cache else ["none"]


def _resolve_config_snapshot(
    fp: HardwareFingerprint,
    profile: str | None,
    extra_ptopt_args: Sequence[str],
) -> dict[str, Any]:
    flat = _flatten_ptopt_args(extra_ptopt_args)
    prefix = ["--profile", profile, *flat] if profile else flat
    parsed = parse_ptopt_argv([*prefix, "--", "python3"])
    overrides = _collect_overrides(parsed.ptopt_args)
    res = ConfigResolver().resolve(fp, profile=parsed.ptopt_args.profile, overrides=overrides)
    return build_config_snapshot_dict(fp, res)


class _NvidiaSmiSampler:
    """Background sampler recording GPU memory, SM/mem utilization, power, and temperature timeline."""

    def __init__(self, interval_sec: float = 0.25) -> None:
        self.interval_sec = interval_sec
        self.samples: list[dict[str, float]] = []
        self.peak_mb: float | None = None
        self.gpu_util_avg_pct: float | None = None
        self._t0 = time.perf_counter()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_NvidiaSmiSampler":
        if shutil.which("nvidia-smi") is not None:
            self._t0 = time.perf_counter()
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        if self.samples:
            self.peak_mb = round(max(s["mem_used_mb"] for s in self.samples), 2)
            utils = [s["gpu_util_pct"] for s in self.samples if s["gpu_util_pct"] >= 0]
            if utils:
                self.gpu_util_avg_pct = round(sum(utils) / len(utils), 1)

    @staticmethod
    def _parse_float(raw: str, default: float = -1.0) -> float:
        try:
            return float(raw.strip())
        except ValueError:
            return default

    def _sample_once(self) -> None:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            if proc.returncode != 0:
                return
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            if not lines:
                return
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) < 5:
                return
            sample = {
                "t_sec": round(time.perf_counter() - self._t0, 3),
                "mem_used_mb": self._parse_float(parts[0], 0.0),
                "gpu_util_pct": self._parse_float(parts[1], -1.0),
                "mem_util_pct": self._parse_float(parts[2], -1.0),
                "power_w": self._parse_float(parts[3], -1.0),
                "temp_c": self._parse_float(parts[4], -1.0),
            }
            self.samples.append(sample)
        except (OSError, subprocess.TimeoutExpired):
            return

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_sec)


def _with_repo_on_pythonpath(env: dict[str, str]) -> dict[str, str]:
    out = dict(env)
    parts = [p for p in out.get("PYTHONPATH", "").split(os.pathsep) if p]
    repo_str = str(REPO_ROOT)
    if repo_str not in parts:
        out["PYTHONPATH"] = os.pathsep.join([repo_str, *parts])
    return out


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    """Kill the subprocess and its entire POSIX process group so GPU workers cannot linger."""
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    if proc.poll() is None:
        proc.kill()


def _coerce_text(val: str | bytes | None) -> str:
    if val is None:
        return ""
    return val if isinstance(val, str) else val.decode("utf-8", errors="replace")


def _run_subprocess(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_sec: float,
    sample_gpu: bool = False,
    stream_output: bool = False,
) -> tuple[int, str, str, float, float | None, float | None, list[dict[str, float]]]:
    t0 = time.perf_counter()
    peak_nvsmi: float | None = None
    gpu_util_avg: float | None = None
    samples: list[dict[str, float]] = []

    def _exec_inner() -> tuple[int, str, str]:
        if not stream_output:
            with subprocess.Popen(
                list(cmd),
                cwd=cwd,
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=(os.name == "posix"),
            ) as proc_p:
                try:
                    out_txt, err_txt = proc_p.communicate(timeout=timeout_sec)
                    return proc_p.returncode, out_txt or "", err_txt or ""
                except subprocess.TimeoutExpired:
                    _terminate_process_group(proc_p)
                    try:
                        out_txt, err_txt = proc_p.communicate(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        out_txt, err_txt = "", ""
                    raise subprocess.TimeoutExpired(
                        cmd=list(cmd),
                        timeout=timeout_sec,
                        output=out_txt or "",
                        stderr=err_txt or "",
                    )

        lines: list[str] = []
        timed_out = threading.Event()
        with subprocess.Popen(
            list(cmd),
            cwd=cwd,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=(os.name == "posix"),
        ) as proc_p:
            assert proc_p.stdout is not None

            def _on_timeout() -> None:
                timed_out.set()
                _terminate_process_group(proc_p)

            timer = threading.Timer(timeout_sec, _on_timeout)
            timer.daemon = True
            timer.start()
            try:
                for line in proc_p.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    lines.append(line)
                    if timed_out.is_set() or (time.perf_counter() - t0) > timeout_sec:
                        timed_out.set()
                        _terminate_process_group(proc_p)
                        raise subprocess.TimeoutExpired(cmd=list(cmd), timeout=timeout_sec, output="".join(lines))
                if timed_out.is_set():
                    raise subprocess.TimeoutExpired(cmd=list(cmd), timeout=timeout_sec, output="".join(lines))
                rc_val = proc_p.wait(timeout=max(1.0, timeout_sec - (time.perf_counter() - t0)))
            finally:
                timer.cancel()
                _terminate_process_group(proc_p)
        return rc_val, "".join(lines), ""

    try:
        if sample_gpu:
            with _NvidiaSmiSampler() as sampler:
                rc_val, out_txt, err_txt = _exec_inner()
            peak_nvsmi = sampler.peak_mb
            gpu_util_avg = sampler.gpu_util_avg_pct
            samples = sampler.samples
        else:
            rc_val, out_txt, err_txt = _exec_inner()
        wall = time.perf_counter() - t0
        return rc_val, out_txt, err_txt, wall, peak_nvsmi, gpu_util_avg, samples
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - t0
        out = _coerce_text(exc.stdout)
        err = _coerce_text(exc.stderr)
        return 124, out, f"Timed out after {timeout_sec:.0f}s\n{err}".strip(), wall, None, None, []


def _build_ptopt_cmd(
    target_cmd: Sequence[str],
    *,
    telemetry_only: bool,
    telemetry_out: Path | None,
    profile: str | None,
    cache_dir: Path | None,
    extra_ptopt_args: Sequence[str],
    verbose: bool = False,
    fallback_report: bool = False,
) -> list[str]:
    cmd = [sys.executable, "-m", "ptopt.cli.main"]
    if telemetry_only:
        cmd.append("--telemetry-only")
    elif profile:
        cmd.extend(["--profile", profile])
    if verbose:
        cmd.append("--verbose")
    if fallback_report:
        cmd.append("--fallback-report")
    if telemetry_out is not None:
        cmd.extend(["--telemetry-out", str(telemetry_out)])
    if cache_dir is not None and not telemetry_only:
        cmd.extend(["--override", f"compile.cache_dir={cache_dir}"])
    if not telemetry_only and extra_ptopt_args:
        cmd.extend(_flatten_ptopt_args(extra_ptopt_args))
    cmd.append("--")
    cmd.extend(target_cmd)
    return cmd


def _prepare_cache_dir(cache_dir: Path, cache_state: str) -> None:
    if cache_state == "cold" and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)


def _write_run_artifacts(
    run_dir: Path,
    *,
    meta: Mapping[str, Any],
    stdout_text: str,
    stderr_text: str,
    tel_data: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    arm_label: str,
    exit_code: int,
    target_cmd: Sequence[str],
    config_snapshot: Mapping[str, Any] | None,
    debug_profile: Mapping[str, Any],
    diag_stderr: str | None,
    pure_wall_log: str | None,
    nvsmi_samples: Sequence[Mapping[str, float]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (run_dir / "stdout_stderr.log").write_text(
        f"=== STDOUT ===\n{stdout_text}\n\n=== STDERR ===\n{stderr_text}\n",
        encoding="utf-8",
    )
    (run_dir / "telemetry.json").write_text(json.dumps(tel_data, indent=2) + "\n", encoding="utf-8")
    (run_dir / "ptopt_events.json").write_text(json.dumps(list(events), indent=2) + "\n", encoding="utf-8")
    (run_dir / "ptopt_fallback.txt").write_text(
        format_fallback_report(events, profile=arm_label, command=target_cmd, exit_code=exit_code) + "\n",
        encoding="utf-8",
    )
    if config_snapshot is not None:
        (run_dir / "resolved_config.json").write_text(json.dumps(config_snapshot, indent=2) + "\n", encoding="utf-8")
    if debug_profile:
        (run_dir / "debug_profile.json").write_text(json.dumps(debug_profile, indent=2) + "\n", encoding="utf-8")
    if diag_stderr is not None:
        (run_dir / "diag_stderr.log").write_text(diag_stderr, encoding="utf-8")
    if pure_wall_log is not None:
        (run_dir / "pure_wall.log").write_text(pure_wall_log, encoding="utf-8")
    if nvsmi_samples:
        with (run_dir / "nvsmi_timeline.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["t_sec", "mem_used_mb", "gpu_util_pct", "mem_util_pct", "power_w", "temp_c"],
            )
            writer.writeheader()
            for row in nvsmi_samples:
                writer.writerow(row)


def _execute_arm(
    spec: RealTargetSpec,
    *,
    arm_label: str,
    target_cmd: Sequence[str],
    profile: str | None,
    telemetry_only: bool,
    cache_state: str,
    cache_dir: Path,
    extra_ptopt_args: Sequence[str],
    fp: HardwareFingerprint,
    run_artifact_dir: Path | None,
    live_progress: bool = False,
) -> RealRunRecord:
    """Execute timing pass + optional pure-wall pass + optional deep diagnostic pass."""
    _prepare_cache_dir(cache_dir, cache_state)
    base_env = dict(os.environ)
    base_env.update(spec.env)
    ptopt_env = _with_repo_on_pythonpath(base_env)

    total_passes = 1 + (1 if spec.measure_pure_wall else 0) + (1 if spec.diag_mode == "deep" else 0)
    pass_idx = 1

    if live_progress:
        print(f"  [pass {pass_idx}/{total_passes}] telemetry timing ...", end=" ", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        tel_path = Path(tmp) / "telemetry.json"
        cmd = _build_ptopt_cmd(
            target_cmd,
            telemetry_only=telemetry_only,
            telemetry_out=tel_path,
            profile=profile,
            cache_dir=cache_dir,
            extra_ptopt_args=extra_ptopt_args,
        )
        rc, stdout_txt, stderr_txt, wall_sec, _, _, _ = _run_subprocess(
            cmd,
            cwd=spec.cwd,
            env=ptopt_env,
            timeout_sec=spec.timeout_sec,
            sample_gpu=False,
            stream_output=spec.stream_output,
        )
        tel_data: dict[str, Any] = {}
        if tel_path.is_file():
            try:
                tel_data = json.loads(tel_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                tel_data = {}
    if live_progress:
        status_lbl = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"{status_lbl} ({wall_sec:.2f}s)", flush=True)

    pure_wall_sec: float | None = None
    nvsmi_peak_mb: float | None = None
    gpu_util_avg: float | None = None
    nvsmi_samples: list[dict[str, float]] = []
    pure_wall_log: str | None = None

    if spec.measure_pure_wall and rc == 0:
        pass_idx += 1
        if live_progress:
            print(f"  [pass {pass_idx}/{total_passes}] pure_wall + nvidia-smi ...", end=" ", flush=True)
        _prepare_cache_dir(cache_dir, cache_state)
        if telemetry_only:
            wall_cmd = list(target_cmd)
            wall_env = base_env
        else:
            wall_cmd = _build_ptopt_cmd(
                target_cmd,
                telemetry_only=False,
                telemetry_out=None,
                profile=profile,
                cache_dir=cache_dir,
                extra_ptopt_args=extra_ptopt_args,
            )
            wall_env = ptopt_env
        w_rc, w_out, w_err, pure_dt, sampled_mb, util_avg, samples = _run_subprocess(
            wall_cmd,
            cwd=spec.cwd,
            env=wall_env,
            timeout_sec=spec.timeout_sec,
            sample_gpu=True,
            stream_output=False,
        )
        pure_wall_log = f"CMD: {' '.join(wall_cmd)}\nRC: {w_rc}\nWALL_SEC: {pure_dt:.4f}\n\n=== STDOUT ===\n{w_out}\n\n=== STDERR ===\n{w_err}\n"
        if w_rc == 0:
            pure_wall_sec = round(pure_dt, 4)
            nvsmi_peak_mb = sampled_mb
            gpu_util_avg = util_avg
            nvsmi_samples = samples
        if live_progress:
            nvsmi_info = f", nvsmi={sampled_mb:.0f}MB" if sampled_mb is not None else ""
            util_info = f", gpu_util={util_avg:.1f}%" if util_avg is not None else ""
            print(f"{'OK' if w_rc == 0 else f'FAIL(rc={w_rc})'} ({pure_dt:.2f}s{nvsmi_info}{util_info})", flush=True)

    debug_profile: dict[str, Any] = {}
    diag_stderr: str | None = None
    if spec.diag_mode == "deep" and rc == 0:
        pass_idx += 1
        if live_progress:
            print(f"  [pass {pass_idx}/{total_passes}] deep_diag (profiler + TORCH_LOGS) ...", end=" ", flush=True)
        diag_cache = (
            cache_dir.parent / f"{cache_dir.name}_diag_cold"
            if cache_state == "cold"
            else cache_dir
        )
        _prepare_cache_dir(diag_cache, cache_state)
        diag_env = dict(ptopt_env)
        diag_env["PTOPT_DEBUG_PROFILE"] = "1"
        diag_env["PTOPT_TORCH_PROFILE"] = "1"
        diag_env.setdefault("PTOPT_TORCH_PROFILE_TOP_K", "20")
        diag_env.setdefault("PTOPT_TORCH_PROFILE_SHAPES", "1")
        if "TORCH_LOGS" not in diag_env:
            diag_env["TORCH_LOGS"] = _TORCH_LOGS_SPEC

        with tempfile.TemporaryDirectory() as dtmp:
            dtel_path = Path(dtmp) / "diag_telemetry.json"
            dcmd = _build_ptopt_cmd(
                target_cmd,
                telemetry_only=telemetry_only,
                telemetry_out=dtel_path,
                profile=profile,
                cache_dir=diag_cache,
                extra_ptopt_args=extra_ptopt_args,
                verbose=True,
                fallback_report=True,
            )
            d_rc, d_out, d_err, d_dt, _, _, _ = _run_subprocess(
                dcmd,
                cwd=spec.cwd,
                env=diag_env,
                timeout_sec=spec.timeout_sec,
                sample_gpu=False,
                stream_output=False,
            )
            diag_stderr = f"DIAG_RC: {d_rc}\nTORCH_LOGS: {diag_env.get('TORCH_LOGS')}\n\n=== STDERR ===\n{d_err}\n\n=== STDOUT ===\n{d_out}\n"
            if d_rc == 0 and dtel_path.is_file():
                try:
                    dtel_data = json.loads(dtel_path.read_text(encoding="utf-8"))
                    debug_profile = dict(dtel_data.get("debug_profile") or {})
                except json.JSONDecodeError:
                    debug_profile = {}
        if live_progress:
            print(f"{'OK' if d_rc == 0 else f'FAIL(rc={d_rc})'} ({d_dt:.2f}s)", flush=True)

    train_raw = tel_data.get("train_step_times_ms")
    if train_raw is None:
        train_raw = tel_data.get("step_times_ms") or []
    eval_raw = tel_data.get("eval_step_times_ms") or []
    if not train_raw and not eval_raw and tel_data.get("step_times_ms"):
        train_raw = tel_data["step_times_ms"]

    mem = tel_data.get("memory_mb") or {}
    combined_out = (stdout_txt + "\n" + stderr_txt).strip()
    extracted = extract_metrics_from_output(combined_out, spec.metric_regexes)
    err_msg = None if rc == 0 else ("\n".join(combined_out.splitlines()[-12:]) or f"exit={rc}")[-400:]
    events = list(tel_data.get("fallback_events") or [])

    if run_artifact_dir is not None:
        cfg_snap = None if telemetry_only else _resolve_config_snapshot(fp, profile, extra_ptopt_args)
        _write_run_artifacts(
            run_artifact_dir,
            meta={
                "target": spec.name,
                "arm": arm_label,
                "cache_state": cache_state,
                "cwd": str(spec.cwd),
                "target_cmd": list(target_cmd),
                "ptopt_cmd": cmd,
                "env_overrides": spec.env,
                "exit_code": rc,
                "end_to_end_sec": round(wall_sec, 4),
                "pure_wall_sec": pure_wall_sec,
                "nvsmi_peak_mb": nvsmi_peak_mb,
                "gpu_util_avg_pct": gpu_util_avg,
                "extracted_metrics": extracted,
            },
            stdout_text=stdout_txt,
            stderr_text=stderr_txt,
            tel_data=tel_data,
            events=events,
            arm_label=arm_label,
            exit_code=rc,
            target_cmd=target_cmd,
            config_snapshot=cfg_snap,
            debug_profile=debug_profile,
            diag_stderr=diag_stderr,
            pure_wall_log=pure_wall_log,
            nvsmi_samples=nvsmi_samples,
        )

    return RealRunRecord(
        target=spec.name,
        arm=arm_label,
        cache_state=cache_state,
        exit_code=rc,
        end_to_end_sec=round(wall_sec, 4),
        train_stats=compute_step_stats(train_raw),
        eval_stats=compute_step_stats(eval_raw),
        peak_allocated_mb=round(float(mem.get("peak_allocated") or 0.0), 2),
        peak_reserved_mb=round(float(mem.get("peak_reserved") or 0.0), 2),
        losses=[float(x) for x in (tel_data.get("losses") or [])],
        extracted_metrics=extracted,
        fallback_events=events,
        pure_wall_sec=pure_wall_sec,
        nvsmi_peak_mb=nvsmi_peak_mb,
        gpu_util_avg_pct=gpu_util_avg,
        debug_profile=debug_profile,
        artifact_dir=str(run_artifact_dir) if run_artifact_dir is not None else None,
        error=err_msg,
    )


def _run_arm_trials(spec: RealTargetSpec, **kwargs: Any) -> RealRunRecord:
    trials = max(1, spec.trials)
    runs = [_execute_arm(spec, **kwargs) for _ in range(trials)]
    runs.sort(key=lambda r: (r.exit_code != 0, r.end_to_end_sec))
    return runs[len(runs) // 2]


def _build_arm_plan(
    spec: RealTargetSpec,
    *,
    target_cache: Path,
    resolver: ConfigResolver,
    fp: HardwareFingerprint,
) -> list[dict[str, Any]]:
    wrap_cmd = spec.wrap_cmd or spec.cmd
    plan: list[dict[str, Any]] = [
        {
            "arm_label": "baseline",
            "target_cmd": spec.cmd,
            "profile": None,
            "telemetry_only": True,
            "cache_state": "none",
            "cache_dir": target_cache / "baseline",
            "extra_ptopt_args": (),
        }
    ]
    for prof in spec.profiles:
        states = _resolve_cache_states(prof, spec.cache_modes, resolver, fp)
        prof_cache = target_cache / prof
        for state in states:
            plan.append({
                "arm_label": prof,
                "target_cmd": wrap_cmd,
                "profile": prof,
                "telemetry_only": False,
                "cache_state": state,
                "cache_dir": prof_cache,
                "extra_ptopt_args": spec.ptopt_args,
            })
    if spec.native_cmd:
        plan.append({
            "arm_label": "native_opt",
            "target_cmd": spec.native_cmd,
            "profile": None,
            "telemetry_only": True,
            "cache_state": "none",
            "cache_dir": target_cache / "native_opt",
            "extra_ptopt_args": (),
        })
        if spec.test_conflict:
            conflict_prof = spec.conflict_profile
            states = _resolve_cache_states(conflict_prof, spec.cache_modes, resolver, fp)
            for state in states:
                plan.append({
                    "arm_label": f"conflict:{conflict_prof}",
                    "target_cmd": spec.native_cmd,
                    "profile": conflict_prof,
                    "telemetry_only": False,
                    "cache_state": state,
                    "cache_dir": target_cache / f"conflict_{conflict_prof}",
                    "extra_ptopt_args": spec.ptopt_args,
                })
    return plan


def run_real_target(
    spec: RealTargetSpec,
    *,
    fingerprint: HardwareFingerprint | None = None,
    cache_root: Path | None = None,
    bundle_dir: Path | None = None,
    live_progress: bool = False,
    on_record: Callable[[RealRunRecord, list[RealRunRecord]], None] | None = None,
) -> list[RealRunRecord]:
    """Run all configured arms (A/B/C/D) for a single external repository target."""
    if not spec.cwd.is_dir():
        raise FileNotFoundError(f"Target cwd does not exist: {spec.cwd}")
    fp = fingerprint or collect_fingerprint()
    resolver = ConfigResolver()
    records: list[RealRunRecord] = []

    def _run_inside_cache(base_cache: Path) -> None:
        target_cache = base_cache / spec.name
        plan = _build_arm_plan(spec, target_cache=target_cache, resolver=resolver, fp=fp)
        total = len(plan)
        for idx, item in enumerate(plan, start=1):
            arm_label = str(item["arm_label"])
            cache_state = str(item["cache_state"])
            run_art_dir: Path | None = None
            if bundle_dir is not None:
                slug = f"{idx:02d}_{_sanitize_slug(spec.name)}__{_sanitize_slug(arm_label)}__{_sanitize_slug(cache_state)}"
                run_art_dir = bundle_dir / "runs" / slug
            if live_progress:
                print(
                    f"\n>>> [{idx}/{total}] {spec.name} | arm={arm_label} | cache={cache_state}",
                    flush=True,
                )
            rec = _run_arm_trials(
                spec,
                arm_label=arm_label,
                target_cmd=item["target_cmd"],
                profile=item["profile"],
                telemetry_only=bool(item["telemetry_only"]),
                cache_state=cache_state,
                cache_dir=item["cache_dir"],
                extra_ptopt_args=item["extra_ptopt_args"],
                fp=fp,
                run_artifact_dir=run_art_dir,
                live_progress=live_progress,
            )
            records.append(rec)
            if on_record is not None:
                on_record(rec, list(records))

    if cache_root is not None:
        _run_inside_cache(cache_root)
    else:
        with tempfile.TemporaryDirectory() as tmp_cache:
            _run_inside_cache(Path(tmp_cache))
    return records


def _summarize_audit(events: Sequence[Mapping[str, Any]]) -> str:
    if not events:
        return "-"
    counts = {"APPLIED": 0, "SKIPPED": 0, "FALLBACK": 0}
    for ev in events:
        st = str(ev.get("status", "")).upper()
        if st in counts:
            counts[st] += 1
    return f"A:{counts['APPLIED']} S:{counts['SKIPPED']} F:{counts['FALLBACK']}"


def _evaluate_arm_parity(
    base: RealRunRecord,
    arm: RealRunRecord,
    *,
    parity_mode: str,
    metric_rel_tol: float,
) -> tuple[bool, float | None, str]:
    profile_key = arm.arm.split(":", 1)[1] if arm.arm.startswith("conflict:") else arm.arm
    if profile_key not in PARITY_THRESHOLDS:
        profile_key = "balanced"

    metric_issues: list[str] = []
    for k, arm_val in arm.extracted_metrics.items():
        if not math.isfinite(arm_val):
            metric_issues.append(f"{k}=non-finite")
            continue
        if k in base.extracted_metrics:
            b_val = base.extracted_metrics[k]
            rel = abs(arm_val - b_val) / max(abs(b_val), 1e-8)
            if rel > metric_rel_tol:
                metric_issues.append(f"{k} rel={rel:.4f}>{metric_rel_tol}")

    has_losses = bool(base.losses and arm.losses)
    if parity_mode == "exit_only":
        return True, None, "Exit code 0"
    if parity_mode == "metric":
        if metric_issues:
            return False, None, "; ".join(metric_issues)
        if not arm.extracted_metrics:
            return False, None, "No extracted metrics matched"
        return True, None, "Extracted metrics within tolerance"

    if has_losses:
        res = check_parity(base.losses, arm.losses, profile=profile_key)
        if parity_mode == "loss":
            return res.passed, res.max_rel_diff, res.reason
        if res.has_non_finite:
            return False, res.max_rel_diff, res.reason
        if not res.passed and arm.extracted_metrics and not metric_issues:
            return True, res.max_rel_diff, f"Metrics OK (batch loss rel={res.max_rel_diff})"
        if metric_issues:
            return False, res.max_rel_diff, "; ".join(metric_issues)
        return res.passed, res.max_rel_diff, res.reason

    if metric_issues:
        return False, None, "; ".join(metric_issues)
    return True, None, "Eval/inference completed cleanly"


def build_real_comparison_rows(
    records: Sequence[RealRunRecord],
    specs_by_name: Mapping[str, RealTargetSpec] | None = None,
) -> list[RealComparisonRow]:
    """Compare every non-baseline `RealRunRecord` against its target baseline."""
    baselines = {rec.target: rec for rec in records if rec.arm == "baseline"}
    rows: list[RealComparisonRow] = []
    for rec in records:
        if rec.arm == "baseline":
            continue
        base = baselines.get(rec.target)
        if base is None:
            continue

        base_stats = base.primary_step_stats()
        arm_stats = rec.primary_step_stats()
        kind = "train" if (rec.train_stats.steady_steps > 0 or base.train_stats.steady_steps > 0) else "eval"
        metrics_str = ", ".join(f"{k}={v:.4g}" for k, v in rec.extracted_metrics.items()) or "-"
        audit_str = _summarize_audit(rec.fallback_events)

        if base.exit_code != 0 or rec.exit_code != 0:
            err = rec.error or base.error or f"exit base={base.exit_code} arm={rec.exit_code}"
            rows.append(
                RealComparisonRow(
                    target=rec.target,
                    arm=rec.arm,
                    cache_state=rec.cache_state,
                    kind=kind,
                    base_step_ms=base_stats.median,
                    arm_step_ms=arm_stats.median,
                    arm_p95_ms=arm_stats.p95,
                    step_speedup=0.0,
                    eval_base_ms=base.eval_stats.median,
                    eval_arm_ms=rec.eval_stats.median,
                    eval_speedup=0.0,
                    base_e2e_s=base.end_to_end_sec,
                    arm_e2e_s=rec.end_to_end_sec,
                    e2e_speedup=0.0,
                    pure_wall_base_s=base.pure_wall_sec,
                    pure_wall_arm_s=rec.pure_wall_sec,
                    pure_wall_speedup=None,
                    base_vram_mb=base.peak_allocated_mb,
                    arm_vram_mb=rec.peak_allocated_mb,
                    vram_delta_mb=compute_vram_delta(base.peak_allocated_mb, rec.peak_allocated_mb),
                    max_rel_diff=None,
                    metrics_summary=metrics_str,
                    audit_summary=audit_str,
                    status="FAIL",
                    note=err[:120],
                )
            )
            continue

        spec = specs_by_name.get(rec.target) if specs_by_name else None
        parity_mode = spec.parity_mode if spec else "auto"
        metric_tol = spec.metric_rel_tol if spec else 0.05
        ok, max_rel, note = _evaluate_arm_parity(base, rec, parity_mode=parity_mode, metric_rel_tol=metric_tol)

        pure_sp = (
            compute_speedup(base.pure_wall_sec, rec.pure_wall_sec)
            if (base.pure_wall_sec and rec.pure_wall_sec)
            else None
        )
        rows.append(
            RealComparisonRow(
                target=rec.target,
                arm=rec.arm,
                cache_state=rec.cache_state,
                kind=kind,
                base_step_ms=base_stats.median,
                arm_step_ms=arm_stats.median,
                arm_p95_ms=arm_stats.p95,
                step_speedup=compute_speedup(base_stats.median, arm_stats.median),
                eval_base_ms=base.eval_stats.median,
                eval_arm_ms=rec.eval_stats.median,
                eval_speedup=compute_speedup(base.eval_stats.median, rec.eval_stats.median),
                base_e2e_s=base.end_to_end_sec,
                arm_e2e_s=rec.end_to_end_sec,
                e2e_speedup=compute_speedup(base.end_to_end_sec, rec.end_to_end_sec),
                pure_wall_base_s=base.pure_wall_sec,
                pure_wall_arm_s=rec.pure_wall_sec,
                pure_wall_speedup=pure_sp,
                base_vram_mb=base.peak_allocated_mb,
                arm_vram_mb=rec.peak_allocated_mb,
                vram_delta_mb=compute_vram_delta(base.peak_allocated_mb, rec.peak_allocated_mb),
                max_rel_diff=max_rel,
                metrics_summary=metrics_str,
                audit_summary=audit_str,
                status="PASS" if ok else "FAIL",
                note=note,
            )
        )
    return rows


def _derive_optimization_hints(records: Sequence[RealRunRecord]) -> list[str]:
    """Generate actionable optimization hints from phase shares, Dynamo stats, allocator, and audit events."""
    hints: list[str] = []
    for r in records:
        tag = f"{r.target} ({r.arm}/{r.cache_state})"
        pb = r.debug_profile.get("phase_breakdown") or {}
        shares = pb.get("steady_shares_pct") or {}
        if shares:
            if float(shares.get("data_other", 0.0)) >= 25.0:
                hints.append(
                    f"[BOTTLENECK:DATA] {tag}: Data/Other is {shares['data_other']}% of step time "
                    "-> try `--ptopt-arg=--override dataloader.cuda_prefetch=true` or increase DataLoader workers."
                )
            if float(shares.get("optim", 0.0)) >= 15.0:
                hints.append(
                    f"[BOTTLENECK:OPTIM] {tag}: Optimizer step takes {shares['optim']}% "
                    "-> check ptopt_events.json whether `optimizer.fusion` applied fused/foreach."
                )
            if float(shares.get("loss_clip", 0.0)) >= 15.0:
                hints.append(
                    f"[BOTTLENECK:LOSS] {tag}: Loss/Clip takes {shares['loss_clip']}% "
                    "-> check Hybrid FusedLinearCrossEntropy eligibility or functional loss fp32 cast cost."
                )

        dyn = (r.debug_profile.get("dynamo_compile") or {}).get("summary") or {}
        if dyn:
            if int(dyn.get("graph_break_count", 0)) > 0:
                hints.append(
                    f"[COMPILE:GRAPH_BREAK] {tag}: {dyn['graph_break_count']} graph breaks "
                    "(unique_graphs={}) -> inspect `diag_stderr.log` and `debug_profile.json`.".format(dyn.get("unique_graphs"))
                )
            if int(dyn.get("recompiles_count", 0)) > 0:
                hints.append(
                    f"[COMPILE:RECOMPILE] {tag}: {dyn['recompiles_count']} recompiles "
                    "-> consider `--ptopt-arg=--override compile.dynamic=true` or check shape variance in module_summary."
                )

        mem = r.debug_profile.get("cuda_memory_debug") or {}
        if mem:
            if int(mem.get("num_alloc_retries", 0)) > 0 or int(mem.get("num_ooms", 0)) > 0:
                hints.append(
                    f"[VRAM:PRESSURE] {tag}: alloc_retries={mem.get('num_alloc_retries')}, ooms={mem.get('num_ooms')} "
                    "-> test `runtime.activation_checkpointing=always` or lower `PTOPT_ACT_CKPT_VRAM_THRESHOLD`."
                )
            if float(mem.get("reserved_overhead_mb", 0.0)) >= 512.0:
                hints.append(
                    f"[VRAM:FRAGMENTATION] {tag}: reserved overhead is {mem.get('reserved_overhead_mb')} MB "
                    "-> inspect `expandable_segments` & allocator settings."
                )

        if r.gpu_util_avg_pct is not None and r.gpu_util_avg_pct < 60.0:
            hints.append(
                f"[GPU:UNDERUTILIZED] {tag}: avg GPU utilization is {r.gpu_util_avg_pct}% "
                "-> workload is likely launch-bound or CPU/DataLoader-bound (`train_cuda_graphs` / `cuda_prefetch` candidate)."
            )

        for ev in r.fallback_events:
            if str(ev.get("status", "")).upper() == "FALLBACK":
                hints.append(
                    f"[FALLBACK] {tag}: {ev.get('pass_name')} fell back ({ev.get('reason')}: {ev.get('error')})"
                )

    if not hints:
        hints.append("No major phase/compile/VRAM bottleneck flags triggered automatically; inspect Top Ops & debug_profile.json for kernel-level opportunities.")
    return hints


def format_live_record_card(
    rec: RealRunRecord,
    records: Sequence[RealRunRecord],
    specs_by_name: Mapping[str, RealTargetSpec] | None = None,
) -> str:
    """Format an immediate single-arm summary card printed as soon as that arm/profile completes."""
    stats = rec.primary_step_stats()
    metrics_str = ", ".join(f"{k}={v:.4g}" for k, v in rec.extracted_metrics.items()) or "-"
    pb = rec.debug_profile.get("phase_breakdown") or {}
    sh = pb.get("steady_shares_pct") or {}
    phase_str = (
        f"Data={sh.get('data_other', 0):.0f}% Fwd={sh.get('forward', 0):.0f}% "
        f"Loss={sh.get('loss_clip', 0):.0f}% Bwd={sh.get('backward', 0):.0f}% Opt={sh.get('optim', 0):.0f}%"
        if sh
        else "-"
    )
    dyn = (rec.debug_profile.get("dynamo_compile") or {}).get("summary") or {}
    dyn_str = (
        f"graphs={dyn.get('unique_graphs', 0)}, breaks={dyn.get('graph_break_count', 0)}, "
        f"recomp={dyn.get('recompiles_count', 0)}, ind_hit/miss={dyn.get('inductor_cache_hit', 0)}/{dyn.get('inductor_cache_miss', 0)}"
        if dyn
        else "-"
    )
    pure_str = f"{rec.pure_wall_sec:.2f}s" if rec.pure_wall_sec is not None else "-"
    nvsmi_str = f"{rec.nvsmi_peak_mb:.1f}MB" if rec.nvsmi_peak_mb is not None else "-"
    util_str = f"{rec.gpu_util_avg_pct:.1f}%" if rec.gpu_util_avg_pct is not None else "-"

    lines = ["-" * 78]
    if rec.arm == "baseline":
        lines.extend([
            f"[DONE] {rec.target} | BASELINE | rc={rec.exit_code}",
            f"  Step ms    : med={stats.median:.2f}, p95={stats.p95:.2f} | Eval med={rec.eval_stats.median:.2f}ms",
            f"  Wall/VRAM  : tel_e2e={rec.end_to_end_sec:.2f}s | pure_wall={pure_str} | alloc={rec.peak_allocated_mb:.1f}MB | nvsmi={nvsmi_str} | gpu_util={util_str}",
            f"  Metrics    : {metrics_str} | Phase%: {phase_str}",
        ])
    else:
        rows = build_real_comparison_rows(records, specs_by_name=specs_by_name)
        matched = next(
            (r for r in reversed(rows) if r.target == rec.target and r.arm == rec.arm and r.cache_state == rec.cache_state),
            None,
        )
        if matched is not None:
            wall_spd = f"{matched.pure_wall_speedup:.2f}x" if matched.pure_wall_speedup is not None else f"{matched.e2e_speedup:.2f}x(tel)"
            lines.extend([
                f"[DONE] {rec.target} | arm={rec.arm} | cache={rec.cache_state} | status={matched.status} ({matched.note})",
                f"  Speedup    : step={matched.step_speedup:.2f}x ({matched.base_step_ms:.2f}->{matched.arm_step_ms:.2f}ms) | wall={wall_spd} | pure_wall={pure_str}",
                f"  VRAM/GPU   : alloc={matched.base_vram_mb:.1f}->{matched.arm_vram_mb:.1f}MB ({matched.vram_delta_mb:+.1f}MB) | nvsmi={nvsmi_str} | gpu_util={util_str}",
                f"  Parity/Aud : loss_rel={matched.max_rel_diff if matched.max_rel_diff is not None else 'N/A'} | metrics={metrics_str} | audit={matched.audit_summary}",
                f"  Phase/Dyn  : {phase_str} | Dynamo: {dyn_str}",
            ])
        else:
            lines.append(f"[DONE] {rec.target} | arm={rec.arm} | cache={rec.cache_state} | rc={rec.exit_code} | metrics={metrics_str}")
    if rec.error:
        lines.append(f"  Error      : {rec.error}")
    if rec.artifact_dir:
        lines.append(f"  Artifacts  : {rec.artifact_dir}")
    lines.append("-" * 78)
    return "\n".join(lines)


def _render_real_report_text(
    *,
    records: Sequence[RealRunRecord],
    specs_by_name: Mapping[str, RealTargetSpec] | None,
    fingerprint: HardwareFingerprint | None,
    bundle_dir: Path | None,
    now: datetime.datetime,
    is_partial: bool = False,
) -> tuple[str, list[RealComparisonRow], bool]:
    """Render report text, comparison rows, and `any_fail` flag without writing `report/`."""
    comparisons = build_real_comparison_rows(records, specs_by_name=specs_by_name)
    any_fail = any(r.exit_code != 0 for r in records) or any(c.status == "FAIL" for c in comparisons)

    builder = TextReportBuilder(
        phase_title=(
            "REAL-WORLD GITHUB PIPELINE BENCHMARK & DEEP OPTIMIZATION REPORT (IN PROGRESS)"
            if is_partial
            else "REAL-WORLD GITHUB PIPELINE BENCHMARK & DEEP OPTIMIZATION REPORT"
        ),
        repo_root=REPO_ROOT,
        now=now,
    )
    meta: dict[str, object] = {
        "Targets": len({r.target for r in records}),
        "Total Runs": len(records),
        "Comparisons": len(comparisons),
        "Overall Status": "PASS" if not any_fail else "FAIL",
        "Bundle Directory": str(bundle_dir) if bundle_dir is not None else "(disabled)",
    }
    if fingerprint is not None:
        meta["Google Colab"] = fingerprint.is_colab
        meta["Backend / GPU"] = (
            f"{fingerprint.effective_backend} / {fingerprint.primary_gpu.name} ({fingerprint.primary_gpu.sm_name})"
            if fingerprint.primary_gpu
            else fingerprint.effective_backend
        )
        meta["PyTorch / Triton"] = f"{fingerprint.host.torch_version} / {fingerprint.host.triton_version or 'N/A'}"
    builder.add_header(meta)

    comp_rows = []
    for c in comparisons:
        wall_col = (
            f"{c.pure_wall_base_s:.2f}s->{c.pure_wall_arm_s:.2f}s ({c.pure_wall_speedup:.2f}x)"
            if c.pure_wall_speedup is not None
            else f"{c.base_e2e_s:.2f}s->{c.arm_e2e_s:.2f}s ({c.e2e_speedup:.2f}x)"
        )
        comp_rows.append([
            c.target,
            c.arm,
            c.cache_state,
            c.kind,
            f"{c.base_step_ms:.2f}->{c.arm_step_ms:.2f}",
            f"{c.step_speedup:.2f}x" if c.step_speedup > 0 else "N/A",
            wall_col,
            f"{c.base_vram_mb:.1f}->{c.arm_vram_mb:.1f} ({c.vram_delta_mb:+.1f})",
            c.max_rel_diff if c.max_rel_diff is not None else "N/A",
            c.metrics_summary,
            c.audit_summary,
            c.status,
        ])
    builder.add_table(
        "4-ARM REAL PIPELINE COMPARISON (Baseline vs ptopt / Native / Conflict)",
        [
            "Target",
            "Arm",
            "Cache",
            "Kind",
            "Step ms (B->A)",
            "Step Spd",
            "Wall E2E (B->A)",
            "VRAM MB (B->A/d)",
            "LossRel",
            "Stdout Metrics",
            "Audit",
            "Status",
        ],
        comp_rows,
    )

    # Phase breakdown table
    phase_rows = []
    for r in records:
        pb = r.debug_profile.get("phase_breakdown") or {}
        if not pb:
            continue
        cold = pb.get("step_0_cold") or {}
        sm = pb.get("steady_mean_ms") or {}
        sh = pb.get("steady_shares_pct") or {}
        phase_rows.append([
            r.target,
            f"{r.arm}/{r.cache_state}",
            f"{cold.get('total_ms', 0.0):.1f}",
            f"{sm.get('data_other_ms', 0.0):.2f} ({sh.get('data_other', 0.0):.0f}%)",
            f"{sm.get('forward_ms', 0.0):.2f} ({sh.get('forward', 0.0):.0f}%)",
            f"{sm.get('loss_clip_ms', 0.0):.2f} ({sh.get('loss_clip', 0.0):.0f}%)",
            f"{sm.get('backward_ms', 0.0):.2f} ({sh.get('backward', 0.0):.0f}%)",
            f"{sm.get('optim_ms', 0.0):.2f} ({sh.get('optim', 0.0):.0f}%)",
            f"{sm.get('total_ms', 0.0):.2f}",
        ])
    if phase_rows:
        builder.add_table(
            "PHASE BREAKDOWN COMPARISON (Cold Step0 & Steady ms / % Share)",
            ["Target", "Arm/Cache", "Step0 ms", "Data/Other", "Forward", "Loss/Clip", "Backward", "Optim", "Steady Tot"],
            phase_rows,
        )

    # Dynamo & torch.compile diagnostics table
    dyn_rows = []
    for r in records:
        dyn = (r.debug_profile.get("dynamo_compile") or {}).get("summary") or {}
        if not dyn:
            continue
        dyn_rows.append([
            r.target,
            f"{r.arm}/{r.cache_state}",
            f"{dyn.get('frames_ok', 0)}/{dyn.get('frames_total', 0)}",
            dyn.get("unique_graphs", 0),
            dyn.get("graph_break_count", 0),
            dyn.get("recompiles_count", 0),
            f"{dyn.get('inductor_cache_hit', 0)}/{dyn.get('inductor_cache_miss', 0)}",
            f"{dyn.get('aot_cache_hit', 0)}/{dyn.get('aot_cache_miss', 0)}",
        ])
    if dyn_rows:
        builder.add_table(
            "TORCH.COMPILE & DYNAMO DIAGNOSTICS",
            ["Target", "Arm/Cache", "Frames(OK/Tot)", "Graphs", "Breaks", "Recompiles", "Inductor(Hit/Miss)", "AOT(Hit/Miss)"],
            dyn_rows,
        )

    # CUDA memory & utilization table
    mem_rows = []
    for r in records:
        mem = r.debug_profile.get("cuda_memory_debug") or {}
        if not mem and r.nvsmi_peak_mb is None and r.peak_allocated_mb == 0.0:
            continue
        mem_rows.append([
            r.target,
            f"{r.arm}/{r.cache_state}",
            f"{r.peak_allocated_mb:.1f}/{r.peak_reserved_mb:.1f}",
            f"{mem.get('reserved_overhead_mb', 0.0):.1f}" if mem else "-",
            f"{mem.get('headroom_mb', 0.0):.1f}" if mem else "-",
            f"{mem.get('utilization_pct', 0.0):.1f}%" if mem else "-",
            f"{mem.get('num_alloc_retries', 0)}/{mem.get('num_ooms', 0)}" if mem else "-",
            f"{r.nvsmi_peak_mb:.1f}" if r.nvsmi_peak_mb is not None else "-",
            f"{r.gpu_util_avg_pct:.1f}%" if r.gpu_util_avg_pct is not None else "-",
        ])
    if mem_rows:
        builder.add_table(
            "CUDA MEMORY, ALLOCATOR & GPU UTILIZATION DIAGNOSTICS",
            ["Target", "Arm/Cache", "Alloc/Res MB", "Resv Ovh MB", "Headroom MB", "VRAM Util", "Retries/OOM", "NVSMI MB", "Avg GPU Util"],
            mem_rows,
        )

    # Phase 8 VRAM-Headroom Bandwidth Booster telemetry table
    bw_rows = []
    for r in records:
        bw = r.debug_profile.get("bandwidth_vram_booster") or {}
        if not bw:
            continue
        pf = bw.get("prefetch_stats") or {}
        ibw = bw.get("inductor_bw_tune") or {}
        fp8p = bw.get("fp8_cache_policy") or {}
        hr = bw.get("vram_headroom") or {}
        pf_depth = f"{pf.get('initial_depth', '-')}/{pf.get('max_queue_depth', '-')}/{pf.get('current_depth', '-')}" if pf else "-"
        bw_rows.append([
            r.target,
            f"{r.arm}/{r.cache_state}",
            pf_depth,
            f"{pf.get('batches_prefetched', 0)}/{pf.get('backoff_events', 0)}" if pf else "-",
            "on" if ibw.get("enabled") else ("off" if ibw else "-"),
            fp8p.get("mode", "-"),
            f"{hr.get('free_mb', 0.0):.1f} ({hr.get('used_ratio', 0.0):.1%})" if hr else "-",
        ])
    if bw_rows:
        builder.add_table(
            "VRAM HEADROOM & BANDWIDTH BOOSTER TELEMETRY",
            ["Target", "Arm/Cache", "Prefetch(I/M/C)", "Prefetched/Backoff", "Inductor BW", "FP8 Cache", "Free VRAM (Used%)"],
            bw_rows,
        )

    # Top CUDA / CPU ops hotspots per run
    hotspot_lines: list[str] = []
    for r in records:
        tp = r.debug_profile.get("torch_profiler") or {}
        if not tp:
            continue
        hotspot_lines.append(f"--- {r.target} [{r.arm}/{r.cache_state}] (total_cuda={tp.get('total_self_cuda_ms')}ms, total_cpu={tp.get('total_self_cpu_ms')}ms) ---")
        for op in (tp.get("top_cuda_ops") or [])[:8]:
            hotspot_lines.append(
                f"  CUDA {op.get('pct', 0):>5.1f}% | self={op.get('self_cuda_ms', 0):>8.3f}ms | calls={op.get('calls', 0):>5} | {op.get('name')}"
            )
        for op in (tp.get("top_cpu_ops") or [])[:8]:
            hotspot_lines.append(
                f"  CPU  {op.get('pct', 0):>5.1f}% | self={op.get('self_cpu_ms', 0):>8.3f}ms | calls={op.get('calls', 0):>5} | {op.get('name')}"
            )
    if hotspot_lines:
        builder.add_lines_section("TOP CUDA & CPU OPS HOTSPOTS", hotspot_lines)

    # Top-level module & shape trace
    mod_lines: list[str] = []
    for r in records:
        ms = r.debug_profile.get("module_summary") or {}
        if not ms:
            continue
        for cls_name, info in ms.items():
            mod_lines.append(
                f"{r.target} [{r.arm}/{r.cache_state}] {cls_name}: params={info.get('params')}, "
                f"train_calls={info.get('train_calls')}, eval_calls={info.get('eval_calls')}, "
                f"unique_shapes={len(info.get('input_shapes') or [])} -> {info.get('input_shapes')}"
            )
    if mod_lines:
        builder.add_lines_section("TOP-LEVEL MODULE & DYNAMIC SHAPE TRACE", mod_lines)

    # Full pass audit & artifact index
    audit_lines: list[str] = []
    for r in records:
        if r.artifact_dir:
            audit_lines.append(f"[ARTIFACTS] {r.target} ({r.arm}/{r.cache_state}) -> {r.artifact_dir}")
        if r.error:
            audit_lines.append(f"[ERROR] {r.target} ({r.arm}/{r.cache_state}): {r.error}")
        for ev in r.fallback_events:
            audit_lines.append(
                f"[{ev.get('status', 'INFO')}] {r.target} ({r.arm}/{r.cache_state}) -> {ev.get('pass_name')}: {ev.get('reason')}"
                + (f" | {ev.get('error')}" if ev.get("error") else "")
            )
    for c in comparisons:
        if c.status != "PASS":
            audit_lines.append(f"[FAIL] {c.target} ({c.arm}): {c.note}")
    builder.add_lines_section("FULL PASS AUDIT & ARTIFACT INDEX", audit_lines)

    builder.add_lines_section("ACTIONABLE OPTIMIZATION HINTS", _derive_optimization_hints(records))

    builder.add_verification_footer(
        status=(
            "IN_PROGRESS"
            if is_partial
            else ("PENDING_USER_REVIEW" if not any_fail else "FAILED_NEEDS_ATTENTION")
        ),
        note="Inspect bundle artifacts (debug_profile.json, diag_stderr.log, nvsmi_timeline.csv) to guide next optimization",
    )
    return builder.render(), comparisons, any_fail


def generate_real_benchmark_report(
    *,
    records: Sequence[RealRunRecord],
    specs_by_name: Mapping[str, RealTargetSpec] | None = None,
    fingerprint: HardwareFingerprint | None = None,
    report_name: str = "real-bench",
    report_dir: Path | None = None,
    bundle_dir: Path | None = None,
    now: datetime.datetime | None = None,
) -> tuple[Path, str, list[RealComparisonRow]]:
    """Render detailed optimization report into `report/HHMMSS_YYYYMMDD-<name>.txt` and optional bundle."""
    ts = now or datetime.datetime.now().astimezone()
    report_text, comparisons, _ = _render_real_report_text(
        records=records,
        specs_by_name=specs_by_name,
        fingerprint=fingerprint,
        bundle_dir=bundle_dir,
        now=ts,
        is_partial=False,
    )
    target_dir = Path(report_dir) if report_dir is not None else (REPO_ROOT / "report")
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / format_report_filename(report_name, now=ts)
    out_path.write_text(report_text, encoding="utf-8")
    if bundle_dir is not None:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "summary.txt").write_text(report_text, encoding="utf-8")
    return out_path, report_text, comparisons


def export_real_json(
    out_path: Path,
    records: Sequence[RealRunRecord],
    comparisons: Sequence[RealComparisonRow],
    *,
    fingerprint: HardwareFingerprint | None = None,
    hints: Sequence[str] = (),
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "fingerprint": fingerprint.to_dict() if fingerprint is not None else None,
        "records": [r.to_dict() for r in records],
        "comparisons": [c.to_dict() for c in comparisons],
        "optimization_hints": list(hints),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def export_real_csv(out_path: Path, comparisons: Sequence[RealComparisonRow]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(RealComparisonRow.__dataclass_fields__.keys())
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in comparisons:
            writer.writerow(row.to_dict())
    return out_path


def load_manifest_specs(manifest_path: Path) -> list[RealTargetSpec]:
    """Load target specifications from a user-supplied JSON manifest file."""
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = raw.get("targets", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest {manifest_path} must contain a non-empty list of targets.")

    specs: list[RealTargetSpec] = []
    for item in entries:
        if not isinstance(item, Mapping):
            raise ValueError("Each manifest entry must be a JSON object.")
        cwd_raw = Path(str(item["cwd"]))
        cwd = cwd_raw if cwd_raw.is_absolute() else (manifest_path.parent / cwd_raw).resolve()
        specs.append(
            RealTargetSpec(
                name=str(item["name"]),
                cwd=cwd,
                cmd=_normalize_cmd_tokens(item["cmd"]),
                wrap_cmd=_normalize_cmd_tokens(item["wrap_cmd"]) if item.get("wrap_cmd") else None,
                native_cmd=_normalize_cmd_tokens(item["native_cmd"]) if item.get("native_cmd") else None,
                test_conflict=bool(item.get("test_conflict", False)),
                conflict_profile=str(item.get("conflict_profile", "balanced")),
                profiles=tuple(str(p) for p in item.get("profiles", ("safe", "balanced"))),
                cache_modes=tuple(str(c) for c in item.get("cache_modes", ("auto",))),
                trials=int(item.get("trials", 1)),
                metric_regexes={str(k): str(v) for k, v in (item.get("metric_regex") or {}).items()},
                parity_mode=str(item.get("parity_mode", "auto")),
                metric_rel_tol=float(item.get("metric_rel_tol", 0.05)),
                measure_pure_wall=bool(item.get("measure_pure_wall", True)),
                diag_mode=str(item.get("diag_mode", "deep")),
                stream_output=bool(item.get("stream_output", False)),
                ptopt_args=tuple(str(a) for a in (item.get("ptopt_args") or ())),
                env={str(k): str(v) for k, v in (item.get("env") or {}).items()},
                timeout_sec=float(item.get("timeout_sec", 3600.0)),
            )
        )
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run A/B/C/D benchmarks on unmodified external PyTorch GitHub repositories.",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="JSON manifest describing external targets")
    parser.add_argument("--name", default=None, help="Benchmark target label (single-target mode)")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Working directory of external cloned repo")
    parser.add_argument("--cmd", default=None, help="Baseline command string to benchmark")
    parser.add_argument("--wrap-cmd", default=None, help="Optional command override for ptopt arms")
    parser.add_argument("--native-cmd", default=None, help="Optional Arm C command with repo native optimizations")
    parser.add_argument("--test-conflict", action="store_true", help="Also run Arm D (ptopt wrapping --native-cmd)")
    parser.add_argument("--conflict-profile", default="balanced", choices=["supersafe", "safe", "balanced", "aggressive"])
    parser.add_argument("--profiles", default="safe,balanced", help="Comma-separated ptopt profiles")
    parser.add_argument("--cache-modes", default="auto", help="Comma-separated cache states (auto,cold,warm,none)")
    parser.add_argument("--trials", type=int, default=1, help="Trials per arm (median E2E selected)")
    parser.add_argument("--metric-regex", action="append", default=[], help="Extract stdout metric via name=regex")
    parser.add_argument("--parity-mode", default="auto", choices=["auto", "loss", "metric", "exit_only"])
    parser.add_argument("--metric-rel-tol", type=float, default=0.05)
    parser.add_argument("--measure-pure-wall", action="store_true", dest="measure_pure_wall", help="Enable unhooked pure wall-clock + nvidia-smi timeline (default: on)")
    parser.add_argument("--no-pure-wall", action="store_false", dest="measure_pure_wall", help="Skip extra unhooked pure wall-clock pass")
    parser.set_defaults(measure_pure_wall=True)
    parser.add_argument("--diag-mode", default="deep", choices=["deep", "off"], help="Capture deep phase breakdown, torch.profiler ops, TORCH_LOGS & Dynamo counters (default: deep)")
    parser.add_argument("--stream", action="store_true", help="Stream wrapped command stdout/stderr live while capturing")
    parser.add_argument("--quiet", action="store_true", help="Suppress live per-profile progress cards (still prints final report)")
    parser.add_argument("--ptopt-arg", action="append", default=[], help="Extra flag passed to ptopt (e.g. --ptopt-arg=--no-compile)")
    parser.add_argument("--env", action="append", default=[], help="Extra subprocess environment variable KEY=VALUE")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Per-run timeout in seconds (default: 3600 = 1h)")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Persistent Inductor/Triton cache root")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "benchmark_results", help="Root folder saving per-run artifact bundles (default: benchmark_results/)")
    parser.add_argument("--no-save-bundle", action="store_true", help="Skip saving per-run artifact bundle folder")
    parser.add_argument("--report-name", default=None, help="Report suffix (default: real-<name> or real-bench)")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.manifest is not None:
        specs = load_manifest_specs(args.manifest)
    else:
        if not args.cmd:
            parser.error("Provide either --manifest or --cmd.")
        target_cwd = args.cwd.resolve()
        specs = [
            RealTargetSpec(
                name=args.name or target_cwd.name or "external",
                cwd=target_cwd,
                cmd=_normalize_cmd_tokens(args.cmd),
                wrap_cmd=_normalize_cmd_tokens(args.wrap_cmd) if args.wrap_cmd else None,
                native_cmd=_normalize_cmd_tokens(args.native_cmd) if args.native_cmd else None,
                test_conflict=bool(args.test_conflict),
                conflict_profile=args.conflict_profile,
                profiles=tuple(p.strip() for p in args.profiles.split(",") if p.strip()),
                cache_modes=tuple(c.strip() for c in args.cache_modes.split(",") if c.strip()),
                trials=args.trials,
                metric_regexes=parse_metric_specs(args.metric_regex),
                parity_mode=args.parity_mode,
                metric_rel_tol=args.metric_rel_tol,
                measure_pure_wall=bool(args.measure_pure_wall),
                diag_mode=args.diag_mode,
                stream_output=bool(args.stream),
                ptopt_args=tuple(args.ptopt_arg),
                env=parse_env_items(args.env),
                timeout_sec=args.timeout,
            )
        ]

    now = datetime.datetime.now().astimezone()
    fp = collect_fingerprint()
    default_rname = f"real-{specs[0].name}" if len(specs) == 1 else "real-bench"
    rname = args.report_name or default_rname
    bundle_dir: Path | None = None
    if not args.no_save_bundle:
        bundle_slug = format_report_filename(rname, now=now).removesuffix(".txt")
        bundle_dir = args.results_dir.resolve() / bundle_slug
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "env_hardware.json").write_text(
            json.dumps({
                "timestamp": now.isoformat(),
                "fingerprint": fp.to_dict(),
                "specs": [
                    {**asdict(s), "cwd": str(s.cwd)}
                    for s in specs
                ],
            }, indent=2) + "\n",
            encoding="utf-8",
        )

    specs_map = {s.name: s for s in specs}
    all_records: list[RealRunRecord] = []
    live_progress = not args.quiet

    def _on_arm_done(rec: RealRunRecord, _target_recs: list[RealRunRecord]) -> None:
        all_records.append(rec)
        if live_progress:
            print(format_live_record_card(rec, all_records, specs_by_name=specs_map), flush=True)
        if bundle_dir is not None:
            partial_txt, partial_cmp, _ = _render_real_report_text(
                records=all_records,
                specs_by_name=specs_map,
                fingerprint=fp,
                bundle_dir=bundle_dir,
                now=now,
                is_partial=True,
            )
            (bundle_dir / "summary.txt").write_text(partial_txt, encoding="utf-8")
            export_real_json(
                bundle_dir / "summary.json",
                all_records,
                partial_cmp,
                fingerprint=fp,
                hints=_derive_optimization_hints(all_records),
            )
            export_real_csv(bundle_dir / "comparisons.csv", partial_cmp)

    interrupted = False
    try:
        for spec in specs:
            run_real_target(
                spec,
                fingerprint=fp,
                cache_root=args.cache_dir,
                bundle_dir=bundle_dir,
                live_progress=live_progress,
                on_record=_on_arm_done,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[real_bench] Interrupted by user; saving partial results collected so far...", file=sys.stderr, flush=True)

    out_path, report_text, comparisons = generate_real_benchmark_report(
        records=all_records,
        specs_by_name=specs_map,
        fingerprint=fp,
        report_name=rname,
        report_dir=args.report_dir,
        bundle_dir=bundle_dir,
        now=now,
    )
    hints = _derive_optimization_hints(all_records)
    if bundle_dir is not None:
        export_real_json(bundle_dir / "summary.json", all_records, comparisons, fingerprint=fp, hints=hints)
        export_real_csv(bundle_dir / "comparisons.csv", comparisons)
    if args.json_out is not None:
        export_real_json(args.json_out, all_records, comparisons, fingerprint=fp, hints=hints)
    if args.csv_out is not None:
        export_real_csv(args.csv_out, comparisons)

    print(report_text, flush=True)
    print(f"[SAVED] Text report written to: {out_path}", flush=True)
    if bundle_dir is not None:
        print(f"[SAVED] Full benchmark artifact bundle written to: {bundle_dir}", flush=True)
    if interrupted:
        return 130
    any_fail = any(r.exit_code != 0 for r in all_records) or any(c.status == "FAIL" for c in comparisons)
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
