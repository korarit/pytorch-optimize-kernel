"""Pass 3.1c: Safe `channels_last` memory format conversion for Conv2d vision models."""

from __future__ import annotations

import functools
import os
import sys
from typing import TYPE_CHECKING, Any

from src.config import OptimizeConfig

if TYPE_CHECKING:
    from src.hooks.manager import HookManager

_CHANNELS_LAST_INSTALLED = False


def has_conv2d(module: Any, torch_mod: Any) -> bool:
    """Return True if `module` contains at least one `nn.Conv2d` layer."""
    conv2d_cls = getattr(getattr(torch_mod, "nn", None), "Conv2d", None)
    if conv2d_cls is None or not hasattr(module, "modules"):
        return False
    return any(isinstance(sub, conv2d_cls) for sub in module.modules())


def _convert_tensor_channels_last(obj: Any, torch_mod: Any) -> Any:
    tensor_cls = getattr(torch_mod, "Tensor", None)
    channels_last_fmt = getattr(torch_mod, "channels_last", None)
    if tensor_cls is None or channels_last_fmt is None:
        return obj
    if isinstance(obj, tensor_cls) and obj.dim() == 4:
        return obj.contiguous(memory_format=channels_last_fmt)
    if isinstance(obj, tuple):
        return tuple(_convert_tensor_channels_last(item, torch_mod) for item in obj)
    return obj


def convert_module_channels_last(module: Any, torch_mod: Any) -> bool:
    """Convert a Conv2d module and attach a 4D input pre-hook once; returns True if applied."""
    if getattr(module, "_ptopt_channels_last_done", False):
        return False
    if not has_conv2d(module, torch_mod):
        return False

    channels_last_fmt = getattr(torch_mod, "channels_last", None)
    if channels_last_fmt is None:
        return False

    module.to(memory_format=channels_last_fmt)

    def _input_pre_hook(_mod: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(_convert_tensor_channels_last(inp, torch_mod) for inp in inputs)

    if hasattr(module, "register_forward_pre_hook"):
        module.register_forward_pre_hook(_input_pre_hook)
    module._ptopt_channels_last_done = True
    return True


def install_channels_last_hook(torch_mod: Any) -> bool:
    """Intercept top-level Module forward calls to convert Conv2d models on first execution."""
    global _CHANNELS_LAST_INSTALLED  # noqa: PLW0603
    if _CHANNELS_LAST_INSTALLED or torch_mod is None or not hasattr(torch_mod, "nn"):
        return False

    nn_module = torch_mod.nn.Module
    attr_name = "__call__"
    orig_call = getattr(nn_module, attr_name)
    depth = [0]

    @functools.wraps(orig_call)
    def _channels_last_call(mod_self: Any, *args: Any, **kwargs: Any) -> Any:
        depth[0] += 1
        try:
            if depth[0] == 1 and not getattr(mod_self, "_ptopt_channels_last_checked", False):
                mod_self._ptopt_channels_last_checked = True
                convert_module_channels_last(mod_self, torch_mod)
            return orig_call(mod_self, *args, **kwargs)
        finally:
            depth[0] -= 1

    setattr(nn_module, attr_name, _channels_last_call)
    nn_module._ptopt_orig_channels_last_attr = attr_name  # type: ignore[attr-defined]
    nn_module._ptopt_orig_channels_last_call = orig_call  # type: ignore[attr-defined]
    _CHANNELS_LAST_INSTALLED = True
    return True


def uninstall_channels_last_hook(torch_mod: Any) -> None:
    """Restore original `nn.Module` call wrapper (for unit tests)."""
    global _CHANNELS_LAST_INSTALLED  # noqa: PLW0603
    if not _CHANNELS_LAST_INSTALLED or torch_mod is None or not hasattr(torch_mod, "nn"):
        return
    nn_module = torch_mod.nn.Module
    if hasattr(nn_module, "_ptopt_orig_channels_last_call"):
        attr = getattr(nn_module, "_ptopt_orig_channels_last_attr")
        setattr(nn_module, attr, getattr(nn_module, "_ptopt_orig_channels_last_call"))
        delattr(nn_module, "_ptopt_orig_channels_last_attr")
        delattr(nn_module, "_ptopt_orig_channels_last_call")
    _CHANNELS_LAST_INSTALLED = False


def apply_channels_last_pass(
    _cfg: OptimizeConfig,
    _manager: "HookManager",
    torch_mod: Any = None,
) -> str:
    """Enable Conv2d channels_last pass when `PTOPT_CHANNELS_LAST=1`."""
    if os.environ.get("PTOPT_CHANNELS_LAST") != "1":
        return "SKIPPED: opt-in only (set PTOPT_CHANNELS_LAST=1)"
    tmod = torch_mod if torch_mod is not None else sys.modules.get("torch")
    if tmod is None:
        return "SKIPPED: torch not loaded"
    installed = install_channels_last_hook(tmod)
    return "channels_last Conv2d hook installed" if installed else "channels_last hook already active"
