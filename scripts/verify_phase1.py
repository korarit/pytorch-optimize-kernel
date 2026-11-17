#!/usr/bin/env python3
"""Phase 1 consolidated verification script: runs full Phase 1 unit tests + live --explain across profiles."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.explain import build_mock_fingerprint, format_explain_text
from src.common.text_report import TextReportBuilder
from src.config import ConfigResolver
from src.detect import collect_fingerprint


def _run_all_unit_tests(repo_root: Path) -> tuple[str, int, str]:
    unit_dir = repo_root / "tests" / "unit"
    if importlib.util.find_spec("pytest") is not None:
        cmd = [sys.executable, "-m", "pytest", str(unit_dir), "-v"]
        runner = "pytest"
    else:
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests/unit", "-v"]
        runner = "unittest"
    res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    return runner, res.returncode, (res.stdout + "\n" + res.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 1 (1.1 + 1.2 + 1.3) and generate consolidated text report.")
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()

    live_fp = collect_fingerprint()
    resolver = ConfigResolver()
    runner, rc, test_output = _run_all_unit_tests(REPO_ROOT)

    profile_rows = []
    for prof in ("safe", "balanced", "aggressive"):
        cfg = resolver.resolve(live_fp, profile=prof, env={}).config
        profile_rows.append([
            prof,
            cfg.runtime.tf32,
            f"{cfg.amp.enabled}/{cfg.amp.dtype}",
            cfg.amp.grad_scaler,
            f"{cfg.compile.enabled}/{cfg.compile.mode}",
            cfg.optimizer.fused,
            cfg.attention.flash_sdp,
            cfg.dataloader.auto_tune,
            cfg.kernels.enabled,
        ])

    balanced_res = resolver.resolve(live_fp, profile="balanced", env={})
    explain_live_text = format_explain_text(live_fp, balanced_res)

    mock_rows = []
    for preset in ("t4", "l4", "a100", "h100", "cpu", "mps"):
        mfp = build_mock_fingerprint(preset)
        cfg = resolver.resolve(mfp, profile="balanced", env={}).config
        mock_rows.append([
            preset,
            mfp.effective_backend,
            mfp.primary_gpu.sm_name if mfp.primary_gpu else "N/A",
            cfg.runtime.tf32,
            f"{cfg.amp.enabled}/{cfg.amp.dtype}",
            cfg.amp.grad_scaler,
            cfg.compile.enabled,
        ])

    builder = TextReportBuilder(
        phase_title="PHASE 1 CONSOLIDATED REPORT: HARDWARE PROBE, CONFIG ENGINE & --EXPLAIN",
        repo_root=REPO_ROOT,
    )
    builder.add_header({
        "Google Colab": live_fp.is_colab,
        "Live Backend": live_fp.effective_backend,
        "Live GPU": live_fp.primary_gpu.name if live_fp.primary_gpu else "None",
        "Test Runner": f"{runner} (exit_code={rc})",
    })
    builder.add_table(
        "LIVE HOST PROFILES SUMMARY",
        ["Profile", "TF32", "AMP", "Scaler", "Compile", "FusedOpt", "FlashSDP", "DL-Tune", "Kernels"],
        profile_rows,
    )
    builder.add_table(
        "MOCK TARGET ARCH COMPARISON (balanced)",
        ["Preset", "Backend", "SM", "TF32", "AMP", "Scaler", "Compile"],
        mock_rows,
    )
    builder.add_lines_section("LIVE --EXPLAIN OUTPUT (balanced)", explain_live_text.splitlines())
    builder.add_lines_section(f"FULL PHASE 1 UNIT TEST OUTPUT ({runner})", test_output.splitlines())
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if rc == 0 else "FAILED_NEEDS_ATTENTION",
        note="Verify Phase 1 consolidated report on Google Colab before starting Phase 2",
    )

    out_path = builder.write("phase-1", report_dir=args.report_dir)
    print(builder.render())
    print(f"[SAVED] Text report written to: {out_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
