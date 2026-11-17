#!/usr/bin/env python3
"""Phase 4.2 Verification: Run unit tests, `ptopt --doctor`, `--fallback-report`, `--save-report`, and write report."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.text_report import TextReportBuilder
from src.detect import collect_fingerprint


def _run_cmd(cmd: list[str]) -> tuple[int, str, str, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr, time.perf_counter() - t0


def main() -> int:
    fp = collect_fingerprint()
    gpu_name = fp.primary_gpu.name if fp.primary_gpu else "CPU-only"
    device_label = f"{fp.effective_backend} (sm_{fp.sm_int})" if fp.sm_int > 0 else fp.effective_backend

    rows: list[list[str]] = []
    traces: list[str] = []
    all_passed = True

    # 1. Unit tests (`tests/unit/test_doctor_and_fallback.py` executed by path to avoid site-packages shadowing)
    rc_u, out_u, err_u, dt_u = _run_cmd([sys.executable, str(REPO_ROOT / "tests/unit/test_doctor_and_fallback.py"), "-v"])
    ok_u = rc_u == 0
    all_passed &= ok_u
    ran_line = next((ln.strip() for ln in (out_u + "\n" + err_u).splitlines() if ln.strip().startswith("Ran ")), f"exit={rc_u}")
    rows.append(["Unit: Doctor & Fallback", "tests/unit/test_doctor_and_fallback.py", f"{dt_u:.2f}s", "PASS" if ok_u else "FAIL", ran_line])
    if not ok_u:
        traces.append(f"--- FAILURE: test_doctor_and_fallback.py ---\n{(out_u + err_u)[-2000:]}")

    # 2. Live `ptopt --doctor` + targeted mock profiles
    rc_d, out_d, err_d, dt_d = _run_cmd([sys.executable, "-m", "src.cli.main", "--doctor"])
    ok_d = rc_d == 0 and "[Diagnostic Checks]" in out_d and "[Recommended Profile]" in out_d
    all_passed &= ok_d
    rows.append(["CLI: ptopt --doctor (live)", device_label, f"{dt_d:.2f}s", "PASS" if ok_d else "FAIL", "Checks + recommendation printed"])
    if not ok_d:
        traces.append(f"--- FAILURE: ptopt --doctor ---\n{out_d}\n{err_d}")

    rc_t4, out_t4, _err_t4, dt_t4 = _run_cmd([sys.executable, "-m", "src.cli.main", "--target-arch", "low_shm_t4", "--doctor"])
    ok_t4 = rc_t4 == 0 and "[WARN   ] /dev/shm Shared Memory" in out_t4 and "Profile : balanced" in out_t4
    all_passed &= ok_t4
    rows.append(["CLI: --target-arch low_shm_t4", "Colab T4 512MB /dev/shm simulation", f"{dt_t4:.2f}s", "PASS" if ok_t4 else "FAIL", "Recommended balanced + WARN low /dev/shm"])

    # 3. End-to-end `--fallback-report` + `--save-report`
    with tempfile.TemporaryDirectory() as tmp:
        save_dir = Path(tmp) / "run_reports"
        rc_f, out_f, err_f, dt_f = _run_cmd([
            sys.executable,
            "-m",
            "src.cli.main",
            "--profile",
            "safe",
            "--fallback-report",
            "--save-report",
            str(save_dir),
            "--run-name",
            "phase42_probe",
            "--",
            sys.executable,
            "-c",
            "import torch; y = torch.nn.Linear(8, 4)(torch.randn(2, 8)).sum().item(); print(f'sum={y:.4f}')",
        ])
        saved_files = list(save_dir.glob("*-run-phase42_probe.txt"))
        ok_f = (
            rc_f == 0
            and "[ptopt fallback/hook summary]" in err_f
            and "SKIPPED (by arch/config)" in err_f
            and len(saved_files) == 1
        )
        all_passed &= ok_f
        rows.append([
            "CLI: --fallback-report --save-report",
            "End-to-end audit stderr + run-*.txt file",
            f"{dt_f:.2f}s",
            "PASS" if ok_f else "FAIL",
            f"saved={saved_files[0].name if saved_files else 'none'}",
        ])
        if not ok_f:
            traces.append(f"--- FAILURE: fallback/save-report ---\nSTDOUT:\n{out_f}\nSTDERR:\n{err_f}")

    # 4. Documentation check (`ai-docs-dev/USER-GUIDE.md`)
    guide_path = REPO_ROOT / "ai-docs-dev" / "USER-GUIDE.md"
    guide_text = guide_path.read_text(encoding="utf-8") if guide_path.is_file() else ""
    ok_g = all(
        sec in guide_text
        for sec in ("--doctor", "--fallback-report", "Google Colab", "Adding a New GPU Rule", "Adding a New Optimization Pass", "Adding a Custom Triton Kernel")
    )
    all_passed &= ok_g
    rows.append(["Docs: ai-docs-dev/USER-GUIDE.md", "CLI, Profiles, Colab, GPU rule, Pass & Kernel", "0.00s", "PASS" if ok_g else "FAIL", f"{len(guide_text)} bytes"])

    report = TextReportBuilder("PHASE 4.2 REPORT: DIAGNOSTICS (--doctor), FALLBACK REPORTER & DOCS", REPO_ROOT)
    report.add_header({
        "GPU / Backend": f"{gpu_name} | {device_label}",
        "Python / PyTorch": f"{fp.host.python_version} | {fp.host.torch_version}",
        "CUDA / Triton": f"{fp.host.cuda_version or 'N/A'} | {fp.host.triton_version or 'N/A'}",
    })
    report.add_table(
        "Phase 4.2 Verification Checks",
        ["Check", "Scope", "Duration", "Status", "Details"],
        rows,
    )
    report.add_lines_section("Live `ptopt --doctor` Snapshot", out_d.strip().splitlines())
    report.add_lines_section(
        "Failure Diagnostics",
        traces if traces else ["None — all Phase 4.2 diagnostics, fallback reporting, and docs passed cleanly."],
    )
    report.add_kv_section(
        "Phase 4.2 Verdict",
        {
            "Checks Passed": f"{sum(1 for r in rows if r[3] == 'PASS')}/{len(rows)}",
            "Overall Phase 4.2": "PASS" if all_passed else "FAIL",
        },
    )
    report.add_verification_footer(
        status="PENDING_USER_REVIEW" if all_passed else "NEEDS_FIX",
        note="Verify `ptopt --doctor`, `--fallback-report`, and USER-GUIDE.md on Google Colab before advancing to Phase 4.3",
    )

    out_path = report.write(name="phase-4.2")
    print(report.render())
    print(f"Report written to: {out_path}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
