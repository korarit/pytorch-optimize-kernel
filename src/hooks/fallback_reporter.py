"""Fallback & pass-audit reporter for `--fallback-report` and `--save-report`."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ReportedEvent:
    """Normalized pass audit event accepted by fallback formatter and report builder."""

    pass_name: str
    status: str  # APPLIED | SKIPPED | FALLBACK | MIGRATED
    reason: str
    error: str | None = None


@dataclass(frozen=True)
class FallbackSummaryReport:
    """Categorized view of applied, skipped, and fallen-back optimization passes."""

    applied: tuple[ReportedEvent, ...]
    skipped: tuple[ReportedEvent, ...]
    fallbacks: tuple[ReportedEvent, ...]
    migrated: tuple[ReportedEvent, ...]


def coerce_event(raw: Any) -> ReportedEvent:
    """Convert a `FallbackEvent`, `ReportedEvent`, or dict mapping into `ReportedEvent`."""
    if isinstance(raw, ReportedEvent):
        return raw
    if isinstance(raw, Mapping):
        return ReportedEvent(
            pass_name=str(raw.get("pass_name", "unknown")),
            status=str(raw.get("status", "UNKNOWN")).upper(),
            reason=str(raw.get("reason", "")),
            error=(str(raw["error"]) if raw.get("error") is not None else None),
        )
    return ReportedEvent(
        pass_name=str(getattr(raw, "pass_name", "unknown")),
        status=str(getattr(raw, "status", "UNKNOWN")).upper(),
        reason=str(getattr(raw, "reason", "")),
        error=(str(getattr(raw, "error")) if getattr(raw, "error", None) is not None else None),
    )


def classify_events(events: Sequence[Any]) -> FallbackSummaryReport:
    """Group pass audit events into APPLIED, SKIPPED, FALLBACK, and MIGRATED buckets."""
    applied: list[ReportedEvent] = []
    skipped: list[ReportedEvent] = []
    fallbacks: list[ReportedEvent] = []
    migrated: list[ReportedEvent] = []

    for raw in events:
        ev = coerce_event(raw)
        if ev.status == "APPLIED":
            applied.append(ev)
        elif ev.status == "FALLBACK":
            fallbacks.append(ev)
        elif ev.status == "MIGRATED":
            migrated.append(ev)
        else:
            skipped.append(ev)

    return FallbackSummaryReport(
        applied=tuple(applied),
        skipped=tuple(skipped),
        fallbacks=tuple(fallbacks),
        migrated=tuple(migrated),
    )


def _format_event_line(ev: ReportedEvent, tag: str) -> str:
    err_part = f" | error={ev.error}" if ev.error else ""
    return f"  - [{tag}] {ev.pass_name}: {ev.reason}{err_part}"


def format_fallback_report(
    events: Sequence[Any],
    *,
    profile: str | None = None,
    command: Sequence[str] | None = None,
    exit_code: int | None = None,
) -> str:
    """Format clear human-readable pass summary separating APPLIED, SKIPPED, and FALLBACK."""
    grouped = classify_events(events)
    lines = ["[ptopt fallback/hook summary]"]
    meta_parts: list[str] = []
    if profile:
        meta_parts.append(f"profile={profile}")
    if exit_code is not None:
        meta_parts.append(f"exit_code={exit_code}")
    if command:
        meta_parts.append(f"cmd={' '.join(command)}")
    if meta_parts:
        lines.append("  " + " | ".join(meta_parts))

    lines.append(
        f"  Counts: APPLIED={len(grouped.applied)} | SKIPPED={len(grouped.skipped)} | "
        f"FALLBACK={len(grouped.fallbacks)}"
        + (f" | MIGRATED={len(grouped.migrated)}" if grouped.migrated else "")
    )

    if grouped.applied:
        lines.append("  Applied Passes:")
        for ev in grouped.applied:
            lines.append(_format_event_line(ev, "APPLIED"))

    if grouped.skipped:
        lines.append("  Skipped Passes (by arch/config):")
        for ev in grouped.skipped:
            lines.append(_format_event_line(ev, "SKIPPED (by arch/config)"))

    if grouped.fallbacks:
        lines.append("  Fallbacks (runtime error):")
        for ev in grouped.fallbacks:
            lines.append(_format_event_line(ev, "FALLBACK (runtime error)"))

    if grouped.migrated:
        lines.append("  Config Migrations:")
        for ev in grouped.migrated:
            lines.append(_format_event_line(ev, "MIGRATED"))

    if not events:
        lines.append("  (no pass events recorded — target may have exited before `import torch`)")

    return "\n".join(lines)


def write_events_json(events: Sequence[Any], out_path: str | Path) -> Path:
    """Serialize pass events as JSON list to `out_path` for parent CLI collection."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "pass_name": ev.pass_name,
            "status": ev.status,
            "reason": ev.reason,
            "error": ev.error,
        }
        for ev in (coerce_event(item) for item in events)
    ]
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def read_events_json(path: str | Path) -> list[ReportedEvent]:
    """Read serialized pass events back from `path`, returning [] if unavailable."""
    src_path = Path(path)
    if not src_path.is_file():
        return []
    try:
        raw = json.loads(src_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [coerce_event(item) for item in raw]
    except Exception:  # noqa: BLE001
        return []
    return []


def _sanitize_run_name(raw_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_name.strip()).strip("-")
    return cleaned or "session"


def build_run_text_report(
    events: Sequence[Any],
    *,
    run_name: str,
    fp: Any = None,
    profile: str | None = None,
    command: Sequence[str] | None = None,
    exit_code: int | None = None,
    report_dir: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Write structured pass/fallback report into `report/HHMMSS_YYYYMMDD-run-<name>.txt`."""
    from src.common.text_report import TextReportBuilder  # noqa: PLC0415

    root = repo_root or Path(__file__).resolve().parents[2]
    grouped = classify_events(events)
    clean_name = _sanitize_run_name(run_name)

    builder = TextReportBuilder(f"PTOPT RUN DIAGNOSTIC REPORT ({clean_name})", root)
    header_meta: dict[str, object] = {
        "Profile": profile or "default",
        "Exit Code": exit_code if exit_code is not None else "N/A",
        "Command": " ".join(command) if command else "N/A",
    }
    if fp is not None:
        gpu_name = fp.primary_gpu.name if fp.primary_gpu else "CPU-only"
        header_meta["GPU / Backend"] = f"{gpu_name} | {fp.effective_backend} (sm_{fp.sm_int})"
        header_meta["PyTorch / Triton"] = f"{fp.host.torch_version} | {fp.host.triton_version or 'N/A'}"
    builder.add_header(header_meta)

    rows = [
        [
            ev.pass_name,
            (
                "APPLIED"
                if ev.status == "APPLIED"
                else (
                    "FALLBACK (runtime error)"
                    if ev.status == "FALLBACK"
                    else ("MIGRATED" if ev.status == "MIGRATED" else "SKIPPED (by arch/config)")
                )
            ),
            ev.reason,
            ev.error or "-",
        ]
        for ev in (coerce_event(item) for item in events)
    ]
    builder.add_table(
        "Pass & Fallback Audit",
        ["Pass Name", "Status", "Reason", "Error"],
        rows,
    )
    builder.add_kv_section(
        "Summary",
        {
            "Applied": len(grouped.applied),
            "Skipped (arch/config)": len(grouped.skipped),
            "Fallback (runtime error)": len(grouped.fallbacks),
            "Migrated": len(grouped.migrated),
        },
    )

    target_dir = Path(report_dir) if report_dir is not None else (root / "report")
    return builder.write(name=f"run-{clean_name}", report_dir=target_dir)
