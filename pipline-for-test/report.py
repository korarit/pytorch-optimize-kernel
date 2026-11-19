"""Detailed Text Report formatter and optional JSON/CSV exporter for benchmark results."""

from __future__ import annotations

import csv
import datetime
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptopt.common.text_report import TextReportBuilder
from ptopt.detect import HardwareFingerprint
from telemetry import RunRecord, check_parity, compute_speedup, compute_vram_delta


@dataclass(frozen=True)
class ComparisonRow:
    """Single row comparing a wrapped run against its workload baseline."""

    workload: str
    mode: str
    cache_state: str
    base_median_ms: float
    wrap_median_ms: float
    wrap_p95_ms: float
    speedup: float
    base_e2e_s: float
    wrap_e2e_s: float
    base_vram_mb: float
    wrap_vram_mb: float
    vram_delta_mb: float
    max_rel_diff: float | None
    status: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_comparison_rows(records: Sequence[RunRecord]) -> list[ComparisonRow]:
    """Match each wrapped RunRecord with its baseline RunRecord and compute deltas."""
    baselines: dict[str, RunRecord] = {
        rec.workload: rec for rec in records if rec.mode == "baseline"
    }
    rows: list[ComparisonRow] = []
    for rec in records:
        if rec.mode == "baseline":
            continue
        base = baselines.get(rec.workload)
        if base is None:
            rows.append(
                ComparisonRow(
                    workload=rec.workload,
                    mode=rec.mode,
                    cache_state=rec.cache_state,
                    base_median_ms=0.0,
                    wrap_median_ms=rec.step_stats.median,
                    wrap_p95_ms=rec.step_stats.p95,
                    speedup=0.0,
                    base_e2e_s=0.0,
                    wrap_e2e_s=rec.end_to_end_sec,
                    base_vram_mb=0.0,
                    wrap_vram_mb=rec.peak_allocated_mb,
                    vram_delta_mb=0.0,
                    max_rel_diff=None,
                    status="SKIP",
                    note="Missing baseline run",
                )
            )
            continue

        if base.exit_code != 0 or rec.exit_code != 0:
            err_msg = rec.error or base.error or f"exit_code base={base.exit_code}, wrap={rec.exit_code}"
            rows.append(
                ComparisonRow(
                    workload=rec.workload,
                    mode=rec.mode,
                    cache_state=rec.cache_state,
                    base_median_ms=base.step_stats.median,
                    wrap_median_ms=rec.step_stats.median,
                    wrap_p95_ms=rec.step_stats.p95,
                    speedup=0.0,
                    base_e2e_s=base.end_to_end_sec,
                    wrap_e2e_s=rec.end_to_end_sec,
                    base_vram_mb=base.peak_allocated_mb,
                    wrap_vram_mb=rec.peak_allocated_mb,
                    vram_delta_mb=compute_vram_delta(base.peak_allocated_mb, rec.peak_allocated_mb),
                    max_rel_diff=None,
                    status="FAIL",
                    note=err_msg[:120],
                )
            )
            continue

        parity = check_parity(base.losses, rec.losses, profile=rec.mode)
        speedup = compute_speedup(base.step_stats.median, rec.step_stats.median)
        vram_delta = compute_vram_delta(base.peak_allocated_mb, rec.peak_allocated_mb)
        rows.append(
            ComparisonRow(
                workload=rec.workload,
                mode=rec.mode,
                cache_state=rec.cache_state,
                base_median_ms=base.step_stats.median,
                wrap_median_ms=rec.step_stats.median,
                wrap_p95_ms=rec.step_stats.p95,
                speedup=speedup,
                base_e2e_s=base.end_to_end_sec,
                wrap_e2e_s=rec.end_to_end_sec,
                base_vram_mb=base.peak_allocated_mb,
                wrap_vram_mb=rec.peak_allocated_mb,
                vram_delta_mb=vram_delta,
                max_rel_diff=parity.max_rel_diff,
                status="PASS" if parity.passed else "FAIL",
                note=parity.reason,
            )
        )
    return rows


def generate_benchmark_text_report(
    *,
    phase_title: str,
    report_name: str,
    records: Sequence[RunRecord],
    fingerprint: HardwareFingerprint | None = None,
    profile_configs: Mapping[str, Mapping[str, Any]] | None = None,
    report_dir: Path | None = None,
    repo_root: Path | None = None,
    now: datetime.datetime | None = None,
    footer_note: str = "Verify benchmark metrics and parity on Google Colab before advancing",
) -> tuple[Path, str]:
    """Generate a Detailed Text Report into `report/HHMMSS_YYYYMMDD-<name>.txt`."""
    root = repo_root or REPO_ROOT
    comparisons = build_comparison_rows(records)
    any_fail = any(r.exit_code != 0 for r in records) or any(c.status == "FAIL" for c in comparisons)

    builder = TextReportBuilder(
        phase_title=phase_title,
        repo_root=root,
        now=now or datetime.datetime.now().astimezone(),
    )

    header_meta: dict[str, object] = {
        "Total Runs": len(records),
        "Comparisons": len(comparisons),
        "Overall Status": "PASS" if not any_fail else "FAIL",
    }
    if fingerprint is not None:
        header_meta["Google Colab"] = fingerprint.is_colab
        header_meta["Backend / GPU"] = (
            f"{fingerprint.effective_backend} / {fingerprint.primary_gpu.name} ({fingerprint.primary_gpu.sm_name})"
            if fingerprint.primary_gpu
            else fingerprint.effective_backend
        )
    builder.add_header(header_meta)

    if fingerprint is not None:
        builder.add_kv_section("HARDWARE & ENVIRONMENT", {
            "os_arch": f"{fingerprint.host.os_name} / {fingerprint.host.cpu_arch}",
            "cores_ram_shm": (
                f"cores={fingerprint.host.physical_cores}/{fingerprint.host.logical_cores}, "
                f"ram={fingerprint.host.total_ram_mb}MB, shm={fingerprint.host.shm_total_mb}MB"
            ),
            "torch_triton": f"torch={fingerprint.host.torch_version}, triton={fingerprint.host.triton_version}",
            "vram_mb": fingerprint.vram_mb,
        })

    if profile_configs:
        cfg_rows = []
        for prof_name, cfg in profile_configs.items():
            cfg_rows.append([
                prof_name,
                cfg.get("runtime", {}).get("tf32"),
                f"{cfg.get('amp', {}).get('enabled')}/{cfg.get('amp', {}).get('dtype')}",
                f"{cfg.get('compile', {}).get('enabled')}/{cfg.get('compile', {}).get('mode')}",
                cfg.get("optimizer", {}).get("fused"),
                cfg.get("dataloader", {}).get("auto_tune"),
                cfg.get("kernels", {}).get("enabled"),
            ])
        builder.add_table(
            "CONFIG PROFILE SUMMARY",
            ["Profile", "TF32", "AMP", "Compile", "FusedOpt", "DL-Tune", "Kernels"],
            cfg_rows,
        )

    comp_rows = [
        [
            c.workload,
            c.mode,
            c.cache_state,
            f"{c.base_median_ms:.2f}",
            f"{c.wrap_median_ms:.2f}",
            f"{c.wrap_p95_ms:.2f}",
            f"{c.speedup:.2f}x",
            f"{c.base_e2e_s:.2f}s / {c.wrap_e2e_s:.2f}s",
            f"{c.base_vram_mb:.1f} / {c.wrap_vram_mb:.1f} ({c.vram_delta_mb:+.1f})",
            c.max_rel_diff if c.max_rel_diff is not None else "N/A",
            c.status,
        ]
        for c in comparisons
    ]
    builder.add_table(
        "BENCHMARK COMPARISON (Baseline vs Wrapped)",
        ["Workload", "Mode", "Cache", "Base ms", "Wrap ms", "Wrap P95", "Speedup", "E2E (B/W)", "VRAM MB (B/W/d)", "MaxRelDiff", "Status"],
        comp_rows,
    )

    raw_rows = [
        [
            r.workload,
            r.mode,
            r.cache_state,
            r.exit_code,
            r.step_stats.warmup_steps,
            r.step_stats.steady_steps,
            f"{r.step_stats.median:.2f}",
            f"{r.step_stats.p95:.2f}",
            f"{r.throughput_value:.1f} {r.throughput_unit}",
            f"{r.peak_allocated_mb:.1f}/{r.peak_reserved_mb:.1f}",
        ]
        for r in records
    ]
    builder.add_table(
        "RAW RUN DETAILS",
        ["Workload", "Mode", "Cache", "RC", "Warmup", "Steady", "Med ms", "P95 ms", "Throughput", "VRAM Alloc/Res MB"],
        raw_rows,
    )

    fb_lines: list[str] = []
    for r in records:
        if r.error:
            fb_lines.append(f"[ERROR] {r.workload} ({r.mode}/{r.cache_state}): {r.error}")
        for ev in r.fallback_events:
            fb_lines.append(
                f"[{ev.get('status', 'INFO')}] {r.workload} ({r.mode}) -> {ev.get('pass_name')}: {ev.get('reason')}"
                + (f" ({ev.get('error')})" if ev.get("error") else "")
            )
    for c in comparisons:
        if c.status != "PASS":
            fb_lines.append(f"[{c.status}] {c.workload} ({c.mode}): {c.note}")
    builder.add_lines_section("FALLBACK & ERRORS", fb_lines)

    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if not any_fail else "FAILED_NEEDS_ATTENTION",
        note=footer_note,
    )

    out_path = builder.write(report_name, report_dir=report_dir)
    return out_path, builder.render()


def export_json_report(path: Path, records: Sequence[RunRecord], comparisons: Sequence[ComparisonRow]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "records": [r.to_dict() for r in records],
        "comparisons": [c.to_dict() for c in comparisons],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def export_csv_report(path: Path, comparisons: Sequence[ComparisonRow]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ComparisonRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in comparisons:
            writer.writerow(row.to_dict())
    return path
