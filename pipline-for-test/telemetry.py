"""Telemetry schema, percentile statistics, speedup, VRAM delta, and parity checks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

TELEMETRY_SCHEMA_VERSION = 1

PARITY_THRESHOLDS: dict[str, float] = {
    "safe": 5e-3,
    "balanced": 2e-2,
    "aggressive": 3e-2,
}


@dataclass(frozen=True)
class StepStats:
    median: float
    p95: float
    min: float
    max: float
    std: float
    mean: float
    warmup_steps: int
    steady_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParityResult:
    passed: bool
    has_non_finite: bool
    max_abs_diff: float | None
    max_rel_diff: float | None
    mean_rel_diff: float | None
    threshold: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunRecord:
    """Normalized record for a single baseline or wrapped benchmark execution."""

    workload: str
    mode: str  # baseline | safe | balanced | aggressive
    cache_state: str  # none | cold | warm
    exit_code: int
    end_to_end_sec: float
    step_stats: StepStats
    throughput_value: float
    throughput_unit: str
    peak_allocated_mb: float
    peak_reserved_mb: float
    losses: list[float] = field(default_factory=list)
    fallback_events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
            "workload": self.workload,
            "mode": self.mode,
            "cache_state": self.cache_state,
            "exit_code": self.exit_code,
            "end_to_end_sec": self.end_to_end_sec,
            "warmup_steps": self.step_stats.warmup_steps,
            "steady_steps": self.step_stats.steady_steps,
            "step_time_ms": {
                "median": self.step_stats.median,
                "p95": self.step_stats.p95,
                "min": self.step_stats.min,
                "max": self.step_stats.max,
                "std": self.step_stats.std,
                "mean": self.step_stats.mean,
            },
            "throughput": {
                "value": self.throughput_value,
                "unit": self.throughput_unit,
            },
            "memory_mb": {
                "peak_allocated": self.peak_allocated_mb,
                "peak_reserved": self.peak_reserved_mb,
            },
            "losses": list(self.losses),
            "fallback_events": list(self.fallback_events),
            "error": self.error,
        }


def percentile(values: Sequence[float], pct: float) -> float:
    """Compute linear-interpolated percentile (0 <= pct <= 100)."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] + frac * (ordered[high] - ordered[low])


def compute_step_stats(step_times_ms: Sequence[float], warmup_steps: int | None = None) -> StepStats:
    """Split warmup steps and compute steady-state latency distribution."""
    n = len(step_times_ms)
    if n == 0:
        return StepStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    if warmup_steps is None:
        warmup = max(1, int(0.2 * n)) if n >= 5 else (1 if n >= 3 else 0)
    else:
        warmup = min(max(0, warmup_steps), max(0, n - 1))

    steady = [float(x) for x in step_times_ms[warmup:]]
    mean_v = sum(steady) / len(steady)
    var = sum((x - mean_v) ** 2 for x in steady) / len(steady)
    return StepStats(
        median=round(percentile(steady, 50.0), 3),
        p95=round(percentile(steady, 95.0), 3),
        min=round(min(steady), 3),
        max=round(max(steady), 3),
        std=round(math.sqrt(var), 3),
        mean=round(mean_v, 3),
        warmup_steps=warmup,
        steady_steps=len(steady),
    )


def compute_speedup(baseline_ms: float, wrapped_ms: float) -> float:
    """Return speedup factor (>1.0 means wrapped is faster)."""
    if baseline_ms <= 0.0 or wrapped_ms <= 0.0:
        return 0.0
    return round(baseline_ms / wrapped_ms, 3)


def compute_vram_delta(baseline_mb: float, wrapped_mb: float) -> float:
    """Return VRAM difference in MB (negative means wrapped uses less VRAM)."""
    return round(wrapped_mb - baseline_mb, 2)


def check_parity(
    baseline_losses: Sequence[float],
    wrapped_losses: Sequence[float],
    profile: str = "balanced",
) -> ParityResult:
    """Validate numerical parity between baseline and wrapped loss trajectories."""
    threshold = PARITY_THRESHOLDS.get(profile, PARITY_THRESHOLDS["balanced"])
    for val in (*baseline_losses, *wrapped_losses):
        if not math.isfinite(float(val)):
            return ParityResult(
                passed=False,
                has_non_finite=True,
                max_abs_diff=None,
                max_rel_diff=None,
                mean_rel_diff=None,
                threshold=threshold,
                reason="Non-finite (NaN/Inf) value detected",
            )

    count = min(len(baseline_losses), len(wrapped_losses))
    if count == 0:
        return ParityResult(
            passed=False,
            has_non_finite=False,
            max_abs_diff=None,
            max_rel_diff=None,
            mean_rel_diff=None,
            threshold=threshold,
            reason="No losses available for parity check",
        )

    abs_diffs: list[float] = []
    rel_diffs: list[float] = []
    for base_v, wrap_v in zip(baseline_losses[:count], wrapped_losses[:count]):
        diff = abs(float(wrap_v) - float(base_v))
        denom = max(abs(float(base_v)), 1e-8)
        abs_diffs.append(diff)
        rel_diffs.append(diff / denom)

    max_abs = round(max(abs_diffs), 6)
    max_rel = round(max(rel_diffs), 6)
    mean_rel = round(sum(rel_diffs) / len(rel_diffs), 6)
    passed = max_rel <= threshold
    reason = "Within tolerance" if passed else f"max_rel_diff={max_rel} > threshold={threshold}"
    return ParityResult(
        passed=passed,
        has_non_finite=False,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        mean_rel_diff=mean_rel,
        threshold=threshold,
        reason=reason,
    )


def merge_summary_and_telemetry(
    *,
    workload: str,
    mode: str,
    cache_state: str,
    exit_code: int,
    summary: Mapping[str, Any] | None,
    telemetry: Mapping[str, Any] | None,
    wall_sec: float,
    error: str | None = None,
) -> RunRecord:
    """Combine workload stdout summary and optional telemetry JSON into a normalized RunRecord."""
    summary_map = summary or {}
    tel_map = telemetry or {}

    raw_steps = tel_map.get("step_times_ms") or summary_map.get("step_times_ms") or []
    if not raw_steps and summary_map.get("step_ms_avg"):
        steps_count = int(summary_map.get("steps", 1) or 1)
        raw_steps = [float(summary_map["step_ms_avg"])] * steps_count

    stats = compute_step_stats(raw_steps)
    losses = list(summary_map.get("losses") or tel_map.get("losses") or [])

    mem = tel_map.get("memory_mb") or {}
    peak_alloc = float(summary_map.get("peak_allocated_mb") or mem.get("peak_allocated") or 0.0)
    peak_resv = float(summary_map.get("peak_reserved_mb") or mem.get("peak_reserved") or 0.0)

    throughput_val = float(summary_map.get("tokens_per_sec") or summary_map.get("throughput_samples_sec") or 0.0)
    throughput_unit = "tokens/sec" if "tokens_per_sec" in summary_map else "samples/sec"

    return RunRecord(
        workload=workload,
        mode=mode,
        cache_state=cache_state,
        exit_code=exit_code,
        end_to_end_sec=round(float(wall_sec if wall_sec > 0 else tel_map.get("end_to_end_sec", 0.0)), 4),
        step_stats=stats,
        throughput_value=round(throughput_val, 2),
        throughput_unit=throughput_unit,
        peak_allocated_mb=round(peak_alloc, 2),
        peak_reserved_mb=round(peak_resv, 2),
        losses=[float(x) for x in losses],
        fallback_events=list(tel_map.get("fallback_events") or []),
        error=error,
    )
