#!/usr/bin/env python3
"""Phase 1.1 verification script: runs live hardware probe + unit tests and writes `report/HHMMSS_YYYYMMDD-phase-1.1.txt`."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.text_report import TextReportBuilder
from src.detect import collect_fingerprint


def _run_unit_tests(repo_root: Path) -> tuple[str, int, str]:
    """Run tests/unit/test_detect.py via pytest if available, else unittest."""
    test_file = repo_root / "tests" / "unit" / "test_detect.py"
    if importlib.util.find_spec("pytest") is not None:
        cmd = [sys.executable, "-m", "pytest", str(test_file), "-v"]
        runner = "pytest"
    else:
        cmd = [sys.executable, "-m", "unittest", "tests/unit/test_detect.py", "-v"]
        runner = "unittest"

    res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    combined = (res.stdout + "\n" + res.stderr).strip()
    return runner, res.returncode, combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 1.1 Hardware Probe and generate text report.")
    parser.add_argument("--report-dir", type=Path, default=None, help="Output report directory (default: <repo>/report)")
    args = parser.parse_args()

    fp = collect_fingerprint()
    runner, rc, test_output = _run_unit_tests(REPO_ROOT)

    builder = TextReportBuilder(
        phase_title="PHASE 1.1: HARDWARE & RUNTIME PROBE VERIFICATION",
        repo_root=REPO_ROOT,
    )
    builder.add_header({
        "Google Colab": fp.is_colab,
        "Effective Backend": fp.effective_backend,
        "Test Runner": f"{runner} (exit_code={rc})",
    })
    builder.add_kv_section("HOST FINGERPRINT", fp.host.to_dict())

    gpu_rows = [
        [
            dev.index,
            dev.name,
            dev.sm_name,
            f"{dev.total_vram_mb} MB",
            dev.sm_count,
            dev.bf16_supported,
            dev.tf32_supported,
            dev.fp8_supported,
            dev.flash_sdp_supported,
        ]
        for dev in fp.gpu.devices
    ]
    builder.add_kv_section("GPU PROBE SUMMARY", {
        "backend": fp.gpu.backend,
        "device_count": fp.gpu.device_count,
        "selected_index": fp.gpu.selected_index,
        "driver_version": fp.gpu.driver_version,
        "cuda_visible_devices": fp.gpu.cuda_visible_devices,
        "probe_errors": fp.gpu.probe_errors or "none",
    })
    builder.add_table(
        "GPU DEVICES",
        ["Idx", "Name", "SM", "VRAM", "SMs", "BF16", "TF32", "FP8", "FlashSDP"],
        gpu_rows,
    )
    builder.add_lines_section(f"UNIT TEST OUTPUT ({runner})", test_output.splitlines())
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if rc == 0 else "FAILED_NEEDS_ATTENTION",
        note="Verify detected Colab GPU/Host values & unit test results manually before Phase 1.2",
    )

    out_path = builder.write("phase-1.1", report_dir=args.report_dir)
    print(builder.render())
    print(f"[SAVED] Text report written to: {out_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
