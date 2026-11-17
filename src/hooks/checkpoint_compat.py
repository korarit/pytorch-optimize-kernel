"""Checkpoint state_dict compatibility hooks (e.g. transparent `_orig_mod.` normalization)."""

from __future__ import annotations

import functools
from collections import OrderedDict
from typing import Any, Mapping

_ORIG_MOD_TOKEN = "_orig_mod."
_COMPAT_INSTALLED = False


def strip_orig_mod_prefix(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Return a state_dict copy with `_orig_mod.` prefixes stripped from all parameter keys."""
    cleaned: OrderedDict[str, Any] = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace(_ORIG_MOD_TOKEN, "") if isinstance(key, str) else key
        cleaned[new_key] = value
    metadata = getattr(state_dict, "_metadata", None)
    if isinstance(metadata, dict):
        cleaned._metadata = {  # type: ignore[attr-defined]
            (k.replace(_ORIG_MOD_TOKEN, "") if isinstance(k, str) else k): v
            for k, v in metadata.items()
        }
    return cleaned


def add_orig_mod_prefix(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Prefix keys with `_orig_mod.` when loading into a compiled wrapper module."""
    prefixed: OrderedDict[str, Any] = OrderedDict()
    for key, value in state_dict.items():
        if isinstance(key, str) and not key.startswith(_ORIG_MOD_TOKEN):
            prefixed[_ORIG_MOD_TOKEN + key] = value
        else:
            prefixed[key] = value
    return prefixed


def normalize_state_dict_for_module(state_dict: Mapping[str, Any], target_module: Any) -> Mapping[str, Any]:
    """Normalize `_orig_mod.` prefixes to match whether `target_module` is compiled or unwrapped."""
    keys = [k for k in state_dict.keys() if isinstance(k, str)]
    if not keys:
        return state_dict

    has_prefix = any(_ORIG_MOD_TOKEN in k for k in keys)
    is_compiled_target = hasattr(target_module, "_orig_mod")

    if is_compiled_target and not has_prefix:
        return add_orig_mod_prefix(state_dict)
    if not is_compiled_target and has_prefix:
        return strip_orig_mod_prefix(state_dict)
    return state_dict


def _ensure_contiguous_state_dict(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Ensure 4D channels_last tensors in state_dict are serialized in standard contiguous layout."""
    out: OrderedDict[str, Any] = OrderedDict()
    for key, val in state_dict.items():
        if hasattr(val, "dim") and hasattr(val, "is_contiguous") and val.dim() == 4 and not val.is_contiguous():
            out[key] = val.contiguous()
        else:
            out[key] = val
    metadata = getattr(state_dict, "_metadata", None)
    if isinstance(metadata, dict):
        out._metadata = dict(metadata)  # type: ignore[attr-defined]
    return out


def install_checkpoint_compat_hooks(torch_mod: Any) -> bool:
    """Install idempotent state_dict/load_state_dict compatibility hooks on `torch.nn.Module`."""
    global _COMPAT_INSTALLED  # noqa: PLW0603
    if _COMPAT_INSTALLED or torch_mod is None or not hasattr(torch_mod, "nn"):
        return False

    nn_module = torch_mod.nn.Module
    orig_state_dict = nn_module.state_dict
    orig_load_state_dict = nn_module.load_state_dict

    @functools.wraps(orig_state_dict)
    def _compat_state_dict(mod_self: Any, *args: Any, **kwargs: Any) -> Any:
        sd = orig_state_dict(mod_self, *args, **kwargs)
        if hasattr(mod_self, "_orig_mod") or any(isinstance(k, str) and _ORIG_MOD_TOKEN in k for k in sd.keys()):
            sd = strip_orig_mod_prefix(sd)
        if getattr(mod_self, "_ptopt_channels_last_done", False):
            sd = _ensure_contiguous_state_dict(sd)
        return sd

    @functools.wraps(orig_load_state_dict)
    def _compat_load_state_dict(mod_self: Any, state_dict: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
        normalized = normalize_state_dict_for_module(state_dict, mod_self)
        return orig_load_state_dict(mod_self, normalized, *args, **kwargs)

    nn_module.state_dict = _compat_state_dict  # type: ignore[method-assign]
    nn_module.load_state_dict = _compat_load_state_dict  # type: ignore[method-assign]
    nn_module._ptopt_orig_state_dict = orig_state_dict  # type: ignore[attr-defined]
    nn_module._ptopt_orig_load_state_dict = orig_load_state_dict  # type: ignore[attr-defined]
    _COMPAT_INSTALLED = True
    return True


def uninstall_checkpoint_compat_hooks(torch_mod: Any) -> None:
    """Restore original `nn.Module.state_dict` and `load_state_dict` (for unit tests)."""
    global _COMPAT_INSTALLED  # noqa: PLW0603
    if not _COMPAT_INSTALLED or torch_mod is None or not hasattr(torch_mod, "nn"):
        return
    nn_module = torch_mod.nn.Module
    if hasattr(nn_module, "_ptopt_orig_state_dict"):
        nn_module.state_dict = nn_module._ptopt_orig_state_dict  # type: ignore[method-assign]
        delattr(nn_module, "_ptopt_orig_state_dict")
    if hasattr(nn_module, "_ptopt_orig_load_state_dict"):
        nn_module.load_state_dict = nn_module._ptopt_orig_load_state_dict  # type: ignore[method-assign]
        delattr(nn_module, "_ptopt_orig_load_state_dict")
    _COMPAT_INSTALLED = False
