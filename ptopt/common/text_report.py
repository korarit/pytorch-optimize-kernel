"""Shared plain-text report builder saving reports to `report/HHMMSS_YYYYMMDD-<name>.txt`."""

from __future__ import annotations

import datetime
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _read_git_meta(repo_root: Path) -> tuple[str, str]:
    """Return (branch, short_commit) if git metadata is readable, else ('N/A', 'N/A')."""
    if not (repo_root / ".git").exists():
        return "N/A", "N/A"
    meta: list[str] = []
    for args in (
        ["rev-parse", "--abbrev-ref", "HEAD"],
        ["rev-parse", "--short", "HEAD"],
    ):
        try:
            res = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            meta.append(res.stdout.strip() if res.returncode == 0 and res.stdout.strip() else "N/A")
        except (OSError, subprocess.TimeoutExpired):
            meta.append("N/A")
    return meta[0], meta[1]


def format_report_filename(name: str, now: datetime.datetime | None = None) -> str:
    """Build report filename following `HHMMSS_YYYYMMDD-<name>.txt`."""
    ts = now or datetime.datetime.now().astimezone()
    clean_name = name.strip().lstrip("-").removesuffix(".txt")
    if not clean_name:
        raise ValueError("Report name must not be empty.")
    return f"{ts.strftime('%H%M%S_%Y%m%d')}-{clean_name}.txt"


@dataclass
class TextReportBuilder:
    """Constructs structured plain-text reports and writes them to `report/`."""

    phase_title: str
    repo_root: Path
    now: datetime.datetime = field(default_factory=lambda: datetime.datetime.now().astimezone())
    _sections: list[str] = field(default_factory=list)

    def add_header(self, extra_meta: Mapping[str, object] | None = None) -> "TextReportBuilder":
        """Append the standard report header block."""
        sep = "=" * 78
        branch, commit = _read_git_meta(self.repo_root)
        ts_str = self.now.strftime("%Y-%m-%d %H:%M:%S %Z").strip() or self.now.isoformat()
        lines = [
            sep,
            f"KERNEL-PYTOUCH-OPTIMIZE | {self.phase_title}",
            sep,
            f"Timestamp          : {ts_str}",
            f"Repo Root          : {self.repo_root}",
            f"Git Branch/Commit  : {branch} @ {commit}",
        ]
        if extra_meta:
            for key, value in extra_meta.items():
                lines.append(f"{str(key):<19}: {value}")
        self._sections.append("\n".join(lines))
        return self

    def add_kv_section(self, title: str, items: Mapping[str, object]) -> "TextReportBuilder":
        """Append a key-value section."""
        lines = [f"[{title}]", "-" * 78]
        for key, value in items.items():
            lines.append(f"{str(key):<22}: {value}")
        self._sections.append("\n".join(lines))
        return self

    def add_lines_section(self, title: str, lines_iter: Iterable[str]) -> "TextReportBuilder":
        """Append a free-form lines section."""
        body = list(lines_iter)
        lines = [f"[{title}]", "-" * 78]
        lines.extend(body if body else ["(none)"])
        self._sections.append("\n".join(lines))
        return self

    def add_table(self, title: str, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> "TextReportBuilder":
        """Append an aligned plain-text table section."""
        str_headers = [str(h) for h in headers]
        str_rows = [[str(cell) for cell in row] for row in rows]
        widths = [len(h) for h in str_headers]
        for row in str_rows:
            for idx, cell in enumerate(row):
                if idx < len(widths):
                    widths[idx] = max(widths[idx], len(cell))

        def _fmt_row(vals: Sequence[str]) -> str:
            return " | ".join(val.ljust(widths[idx]) for idx, val in enumerate(vals))

        lines = [f"[{title}]", "-" * 78, _fmt_row(str_headers), "-+-".join("-" * w for w in widths)]
        if str_rows:
            lines.extend(_fmt_row(row) for row in str_rows)
        else:
            lines.append("(no rows)")
        self._sections.append("\n".join(lines))
        return self

    def add_verification_footer(
        self,
        status: str = "PENDING_USER_REVIEW",
        note: str = "Run on Google Colab & verify manually before advancing",
    ) -> "TextReportBuilder":
        """Append the mandatory Human Verification Gate footer."""
        lines = [
            "[VERIFICATION STATUS]",
            "-" * 78,
            f"STATUS             : {status} ({note})",
            "=" * 78,
        ]
        self._sections.append("\n".join(lines))
        return self

    def render(self) -> str:
        """Render the complete report as a single UTF-8 string."""
        return "\n\n".join(self._sections) + "\n"

    def write(self, name: str, report_dir: Path | None = None) -> Path:
        """Write the rendered report into `report/HHMMSS_YYYYMMDD-<name>.txt`."""
        target_dir = report_dir or (self.repo_root / "report")
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / format_report_filename(name=name, now=self.now)
        out_path.write_text(self.render(), encoding="utf-8")
        return out_path
