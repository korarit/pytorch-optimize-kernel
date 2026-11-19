"""Shared `--override` / `--set` CLI value and list-field parser."""

from __future__ import annotations

from typing import Any, Sequence

from ptopt.config import ConfigValidationError

_LIST_KEYS = frozenset({"kernels.allow_list"})


def parse_override_value(raw: str, *, key: str | None = None) -> Any:
    """Parse scalar or list override values (e.g. booleans, ints, None, comma-separated lists)."""
    stripped = raw.strip()
    low = stripped.lower()
    if key in _LIST_KEYS:
        if low in ("", "[]", "none", "null"):
            return []
        inner = stripped[1:-1] if (stripped.startswith("[") and stripped.endswith("]")) else stripped
        items: list[str] = []
        for token in inner.split(","):
            tok = token.strip().strip("'\"")
            if tok:
                items.append(tok)
        return items
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    try:
        return int(stripped)
    except ValueError:
        return raw


def parse_override_items(items: Sequence[str], *, example: str = "section.field=value") -> dict[str, Any]:
    """Parse sequence of `key=value` CLI override strings into a nested override dict."""
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigValidationError(f"Invalid --override '{item}'; expected {example}.")
        key_raw, val = item.split("=", 1)
        key = key_raw.strip()
        parsed[key] = parse_override_value(val, key=key)
    return parsed
