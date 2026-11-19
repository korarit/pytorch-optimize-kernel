"""Main CLI launcher (`ptopt`) that injects configuration and hooks around unmodified PyTorch commands."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ptopt.cli.doctor import format_doctor_text, run_doctor
from ptopt.cli.explain import (
    available_mock_targets,
    build_config_snapshot_dict,
    build_mock_fingerprint,
    format_explain_text,
    format_show_config_text,
)
from ptopt.cli.overrides import parse_override_items
from ptopt.config import ConfigResolver, ConfigValidationError, OptimizeConfig, migrate_config_dict
from ptopt.detect import collect_fingerprint
from ptopt.hooks.fallback_reporter import build_run_text_report, read_events_json
from ptopt.passes.memory_allocator import apply_prelaunch_env

_VALUE_FLAGS = frozenset({
    "--profile",
    "--amp-dtype",
    "--compile-mode",
    "--dump-config",
    "--save-report",
    "--run-name",
    "--target-arch",
    "--mock-profile",
    "--override",
})
_BOOL_FLAGS = frozenset({
    "--no-compile",
    "--no-amp",
    "--no-dataloader-tune",
    "--no-kernels",
    "--strict",
    "--verbose",
    "--fallback-report",
    "--explain",
    "--show-config",
    "--doctor",
    "-h",
    "--help",
})


@dataclass(frozen=True)
class ParsedCLI:
    """Result of splitting `ptopt` options from the wrapped target command."""

    ptopt_args: argparse.Namespace
    command: list[str]


def _split_argv(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split `argv` into (ptopt_tokens, target_command_tokens) safely."""
    tokens = list(argv)
    if "--" in tokens:
        sep = tokens.index("--")
        return tokens[:sep], tokens[sep + 1 :]

    ptopt_tokens: list[str] = []
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok in _BOOL_FLAGS:
            ptopt_tokens.append(tok)
            idx += 1
            continue
        if any(tok.startswith(flag + "=") for flag in _VALUE_FLAGS):
            ptopt_tokens.append(tok)
            idx += 1
            continue
        if tok in _VALUE_FLAGS:
            ptopt_tokens.append(tok)
            if idx + 1 < len(tokens):
                ptopt_tokens.append(tokens[idx + 1])
                idx += 2
            else:
                idx += 1
            continue
        break
    return ptopt_tokens, tokens[idx:]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptopt",
        description="Zero-code PyTorch optimization wrapper for training and inference.",
    )
    parser.add_argument("--profile", default=None, choices=["supersafe", "safe", "balanced", "aggressive"])
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--compile-mode", default=None, choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--no-amp", action="store_true", help="Disable Auto Mixed Precision")
    parser.add_argument("--amp-dtype", default=None, choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--no-dataloader-tune", action="store_true", help="Disable DataLoader auto-tuning")
    parser.add_argument("--no-kernels", action="store_true", help="Disable custom Triton/CUDA kernels")
    parser.add_argument("--strict", action="store_true", help="Fail fast instead of falling back on hook errors")
    parser.add_argument("--verbose", action="store_true", help="Print runtime hook warnings")
    parser.add_argument("--fallback-report", action="store_true", help="Print pass/fallback summary on exit")
    parser.add_argument("--save-report", type=Path, default=None, help="Directory to write report/HHMMSS_YYYYMMDD-run-<name>.txt")
    parser.add_argument("--run-name", default=None, help="Name suffix used with --save-report")
    parser.add_argument("--explain", action="store_true", help="Print hardware detection, rule trace, and config")
    parser.add_argument("--show-config", action="store_true", help="Print resolved JSON config only")
    parser.add_argument("--doctor", action="store_true", help="Diagnose GPU/PyTorch/Triton/shm readiness and recommend profile")
    parser.add_argument("--dump-config", type=Path, default=None, help="Save config snapshot JSON to path")
    parser.add_argument("--target-arch", default=None, help=f"Mock arch ({', '.join(available_mock_targets())})")
    parser.add_argument("--mock-profile", default=None, help="Alias for --target-arch")
    parser.add_argument("--override", action="append", default=[], help="Override config via section.field=value")
    return parser


def _collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.no_compile:
        overrides["compile.enabled"] = False
    if args.compile_mode is not None:
        overrides["compile.mode"] = args.compile_mode
    if args.no_amp:
        overrides["amp.enabled"] = False
    if args.amp_dtype is not None:
        overrides["amp.dtype"] = args.amp_dtype
    if args.no_dataloader_tune:
        overrides["dataloader.auto_tune"] = False
    if args.no_kernels:
        overrides["kernels.enabled"] = False
    overrides.update(parse_override_items(args.override, example="section.field=value"))
    return overrides


def parse_ptopt_argv(argv: Sequence[str]) -> ParsedCLI:
    """Parse ptopt flags and normalize the wrapped target command."""
    ptopt_tokens, cmd_tokens = _split_argv(argv)
    parser = _build_parser()
    ns = parser.parse_args(ptopt_tokens)

    cmd = list(cmd_tokens)
    if cmd and (cmd[0] == "-m" or cmd[0].endswith(".py")):
        cmd = [sys.executable, *cmd]
    return ParsedCLI(ptopt_args=ns, command=cmd)


def build_child_env(
    config_dict: Mapping[str, Any],
    *,
    strict: bool = False,
    verbose: bool = False,
    fallback_report: bool = False,
    events_out: str | Path | None = None,
    base_env: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Construct environment variables injecting bootstrap `sitecustomize.py` and config."""
    root = repo_root or Path(__file__).resolve().parents[2]
    bootstrap_dir = root / "ptopt" / "hooks" / "bootstrap"

    env = dict(os.environ if base_env is None else base_env)
    existing_pp = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    ordered_pp: list[str] = []
    for candidate in (str(bootstrap_dir), str(root), *existing_pp):
        if candidate not in ordered_pp:
            ordered_pp.append(candidate)

    env["PYTHONPATH"] = os.pathsep.join(ordered_pp)
    env["PTOPT_ACTIVE"] = "1"
    migrated_cfg = migrate_config_dict(config_dict, emit_warnings=verbose).config
    env["PTOPT_CONFIG_JSON"] = json.dumps(migrated_cfg.to_dict())
    env["PTOPT_STRICT"] = "1" if strict else "0"
    env["PTOPT_VERBOSE"] = "1" if verbose else "0"
    env["PTOPT_FALLBACK_REPORT"] = "1" if fallback_report else "0"
    if events_out is not None:
        env["PTOPT_EVENTS_OUT"] = str(events_out)
    apply_prelaunch_env(migrated_cfg, env)
    return env


def _launch_subprocess(cmd: Sequence[str], env: Mapping[str, str]) -> int:
    """Run wrapped target command forwarding signals and returning exact exit code."""
    proc = subprocess.Popen(list(cmd), env=dict(env))

    def _forward_sig(signum: int, _frame: Any) -> None:
        if proc.poll() is None:
            proc.send_signal(signum)

    prev_int = signal.signal(signal.SIGINT, _forward_sig)
    prev_term = signal.signal(signal.SIGTERM, _forward_sig)
    try:
        return proc.wait()
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `ptopt`."""
    parsed = parse_ptopt_argv(sys.argv[1:] if argv is None else argv)
    args = parsed.ptopt_args
    mock_key = args.mock_profile or args.target_arch

    fp = build_mock_fingerprint(mock_key) if mock_key else collect_fingerprint()
    overrides = _collect_overrides(args)
    result = ConfigResolver().resolve(fp, profile=args.profile, overrides=overrides)

    if args.dump_config is not None:
        args.dump_config.parent.mkdir(parents=True, exist_ok=True)
        args.dump_config.write_text(
            json.dumps(build_config_snapshot_dict(fp, result), indent=2) + "\n",
            encoding="utf-8",
        )

    if args.doctor:
        doc_report = run_doctor(fp=fp, cache_dir=result.config.compile.cache_dir, report_dir=args.save_report)
        print(format_doctor_text(doc_report), end="")
        if not parsed.command:
            return 0

    if args.explain or args.show_config:
        if args.show_config and not args.explain:
            print(format_show_config_text(result), end="")
        else:
            print(format_explain_text(fp, result), end="")
        if not parsed.command:
            return 0

    if not parsed.command:
        _build_parser().print_help()
        return 2

    events_tmp: Path | None = None
    if args.save_report is not None:
        fd, raw_tmp = tempfile.mkstemp(prefix="ptopt_events_", suffix=".json")
        os.close(fd)
        events_tmp = Path(raw_tmp)

    try:
        child_env = build_child_env(
            result.config.to_dict(),
            strict=args.strict,
            verbose=args.verbose,
            fallback_report=args.fallback_report,
            events_out=events_tmp,
        )
        rc = _launch_subprocess(parsed.command, child_env)
        if args.save_report is not None and events_tmp is not None:
            events = read_events_json(events_tmp)
            default_name = (
                Path(parsed.command[1]).stem
                if len(parsed.command) > 1 and parsed.command[1].endswith(".py")
                else "ptopt"
            )
            saved_path = build_run_text_report(
                events,
                run_name=args.run_name or default_name,
                fp=fp,
                profile=result.config.profile,
                command=parsed.command,
                exit_code=rc,
                report_dir=args.save_report,
            )
            print(f"[ptopt] Saved fallback report: {saved_path}", file=sys.stderr)
        return rc
    finally:
        if events_tmp is not None and events_tmp.exists():
            try:
                events_tmp.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
