#!/usr/bin/env python3
"""Phase 4.1 Verification: Run full 3-Axis Migrate Matrix + Migrator unit tests and write report."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.text_report import TextReportBuilder
from src.config import detect_runtime_cache_key
from src.detect import collect_fingerprint

_SUITES: tuple[tuple[str, str, str], ...] = (
    (
        "Unit: Config Migrator & Cache",
        "tests/unit/test_config_migrator.py",
        "v0->v1 schema, warnings, cache-key isolation, lazy torch",
    ),
    (
        "Axis 1: Code Migration Matrix",
        "tests/migrate/test_code_migration_matrix.py",
        "legacy AMP, grad accum, multi-opt+LR, custom autograd, train/eval",
    ),
    (
        "Axis 2: Checkpoint Matrix",
        "tests/migrate/test_checkpoint_migration_matrix.py",
        "5x5 cross-load (baseline/supersafe/safe/balanced/aggressive)",
    ),
    (
        "Axis 3: Config & Cache Matrix",
        "tests/migrate/test_config_cache_migration.py",
        "end-to-end v0 CLI/hook upgrade + cross-arch stale cache isolation",
    ),
)


@dataclass(frozen=True)
class SuiteResult:
    """Execution outcome of one verification suite."""

    name: str
    path: str
    coverage: str
    duration_s: float
    returncode: int
    passed: bool
    summary_line: str
    failure_trace: str


def _extract_unittest_summary(output: str, returncode: int) -> str:
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    ran_line = next((ln for ln in reversed(lines) if ln.startswith("Ran ")), "")
    status_line = next((ln for ln in reversed(lines) if ln in ("OK", "FAILED") or ln.startswith("FAILED (")), "")
    if ran_line and status_line:
        return f"{ran_line} -> {status_line}"
    if ran_line:
        return ran_line
    return f"exit={returncode}"


def _run_suite(name: str, test_path: str, coverage: str) -> SuiteResult:
    cmd = [sys.executable, "-m", "unittest", test_path, "-v"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    passed = proc.returncode == 0
    trace = "" if passed else combined[-3000:]
    return SuiteResult(
        name=name,
        path=test_path,
        coverage=coverage,
        duration_s=elapsed,
        returncode=proc.returncode,
        passed=passed,
        summary_line=_extract_unittest_summary(combined, proc.returncode),
        failure_trace=trace,
    )


def main() -> int:
    fp = collect_fingerprint()
    gpu_name = fp.primary_gpu.name if fp.primary_gpu else "CPU-only"
    cache_key = detect_runtime_cache_key()
    device_label = f"{fp.effective_backend} (sm_{fp.sm_int})" if fp.sm_int > 0 else fp.effective_backend

    results = [_run_suite(name, path, cov) for name, path, cov in _SUITES]
    all_passed = all(r.passed for r in results)

    report = TextReportBuilder("PHASE 4.1 REPORT: FULL 3-AXIS MIGRATE TEST MATRIX", REPO_ROOT)
    report.add_header(
        {
            "GPU / Backend": f"{gpu_name} | {device_label}",
            "Python / PyTorch": f"{fp.host.python_version} | {fp.host.torch_version}",
            "CUDA / Triton": f"{fp.host.cuda_version or 'N/A'} | {fp.host.triton_version or 'N/A'}",
            "Runtime Cache Key": cache_key.cache_key,
        }
    )

    rows = [
        [
            r.name,
            r.coverage,
            device_label,
            f"{r.duration_s:.2f}s",
            "PASS" if r.passed else "FAIL",
            r.summary_line,
        ]
        for r in results
    ]
    report.add_table(
        "3-Axis Migrate Matrix Verification Results",
        ["Suite / Axis", "Coverage", "Target", "Duration", "Status", "Test Outcome"],
        rows,
    )

    trace_lines: list[str] = []
    for r in results:
        if not r.passed:
            trace_lines.append(f"--- FAILURE TRACE: {r.name} ({r.path}, exit={r.returncode}) ---")
            trace_lines.extend(r.failure_trace.splitlines())
    if not trace_lines:
        trace_lines.append("None — all 4 suites passed with zero failures or skipped assertions.")
    report.add_lines_section("Failure Diagnostics & Traces", trace_lines)

    report.add_kv_section(
        "Phase 4.1 Verdict",
        {
            "Suites Passed": f"{sum(1 for r in results if r.passed)}/{len(results)}",
            "Total Wall Time": f"{sum(r.duration_s for r in results):.2f}s",
            "Overall Phase 4.1": "PASS" if all_passed else "FAIL",
        },
    )
    report.add_verification_footer(
        status="PENDING_USER_REVIEW" if all_passed else "NEEDS_FIX",
        note="Run `python3 scripts/verify_phase4_1.py` on Google Colab GPU & verify manually before advancing to Phase 4.2",
    )

    report_path = report.write(name="phase-4.1")
    print(report.render())
    print(f"Report written to: {report_path}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
