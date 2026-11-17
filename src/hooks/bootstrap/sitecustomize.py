"""Bootstrap sitecustomize.py injected via PYTHONPATH by `ptopt`."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_repo_on_sys_path() -> None:
    """Ensure package root containing `src/` is importable even from arbitrary working directories."""
    # <repo_root>/src/hooks/bootstrap/sitecustomize.py -> parents[3] == <repo_root>
    pkg_root = Path(__file__).resolve().parents[3]
    if (pkg_root / "src").is_dir():
        root_str = str(pkg_root)
        if root_str not in sys.path:
            sys.path.insert(1, root_str)


def _init() -> None:
    _ensure_repo_on_sys_path()
    try:
        from src.hooks.manager import bootstrap_from_env, chain_load_sitecustomize  # noqa: PLC0415

        chain_load_sitecustomize(__file__)
        if os.environ.get("PTOPT_ACTIVE") == "1":
            bootstrap_from_env()
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("PTOPT_STRICT") == "1":
            raise
        if os.environ.get("PTOPT_VERBOSE") == "1":
            print(f"[ptopt] bootstrap warning: {exc}", file=sys.stderr)


_init()
