#!/usr/bin/env python3
"""Phase 2.1 verification script: tests ptopt CLI & sitecustomize injection and writes `report/HHMMSS_YYYYMMDD-phase-2.1.txt`."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptopt.cli.main import main as ptopt_main
from ptopt.common.text_report import TextReportBuilder
from ptopt.detect import collect_fingerprint


def _run_tests(repo_root: Path) -> tuple[str, int, str]:
    test_file = repo_root / "tests" / "unit" / "test_cli_hooks.py"
    if importlib.util.find_spec("pytest") is not None:
        cmd = [sys.executable, "-m", "pytest", str(test_file), "-v"]
        runner = "pytest"
    else:
        cmd = [sys.executable, "-m", "unittest", "tests/unit/test_cli_hooks.py", "-v"]
        runner = "unittest"
    res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    return runner, res.returncode, (res.stdout + "\n" + res.stderr).strip()


def _probe_ptopt_live() -> dict[str, object]:
    """Run a live inline PyTorch snippet through `ptopt` to verify zero-code hook injection."""
    snippet = (
        "import json, os, torch; "
        "from ptopt.hooks import get_hook_manager; "
        "m = get_hook_manager(); "
        "print(json.dumps({"
        "'ptopt_active': os.environ.get('PTOPT_ACTIVE'), "
        "'bootstrapped': m.bootstrapped, "
        "'torch_patched': m.torch_patched, "
        "'profile': m.config.profile if m.config else None, "
        "'cuda': torch.cuda.is_available()"
        "}))"
    )
    # Run via subprocess invoking `python -m src.cli.main` so stdout is cleanly captured.
    res = subprocess.run(
        [sys.executable, "-m", "ptopt.cli.main", "--profile", "safe", "--", sys.executable, "-c", snippet],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return {"exit_code": res.returncode, "stderr": res.stderr.strip()}
    last_line = res.stdout.strip().splitlines()[-1]
    payload = json.loads(last_line)
    payload["exit_code"] = res.returncode
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 2.1 CLI & Hook Injection.")
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()

    fp = collect_fingerprint()
    runner, rc, test_output = _run_tests(REPO_ROOT)
    live_probe = _probe_ptopt_live()

    # Also verify direct ptopt_main exit code 0 on a no-op command
    passthrough_rc = ptopt_main(["--profile", "safe", "--", sys.executable, "-c", "pass"])
    overall_rc = 0 if (rc == 0 and passthrough_rc == 0 and live_probe.get("exit_code") == 0) else 1

    builder = TextReportBuilder(
        phase_title="PHASE 2.1: CLI LAUNCHER & SITECUSTOMIZE HOOK INJECTION",
        repo_root=REPO_ROOT,
    )
    builder.add_header({
        "Google Colab": fp.is_colab,
        "Backend": fp.effective_backend,
        "GPU": fp.primary_gpu.name if fp.primary_gpu else "None",
        "Test Runner": f"{runner} (exit_code={rc})",
    })
    builder.add_kv_section("LIVE PTOPT INJECTION PROBE", {
        "passthrough_exit_code": passthrough_rc,
        **live_probe,
    })
    builder.add_lines_section(f"UNIT TEST OUTPUT ({runner})", test_output.splitlines())
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if overall_rc == 0 else "FAILED_NEEDS_ATTENTION",
        note="Verify ptopt injection works cleanly on Google Colab without interfering with Colab runtime",
    )

    out_path = builder.write("phase-2.1", report_dir=args.report_dir)
    print(builder.render())
    print(f"[SAVED] Text report written to: {out_path}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
