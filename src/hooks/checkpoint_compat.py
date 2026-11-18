"""Checkpoint state_dict compatibility hooks (e.g. transparent `_orig_mod.` normalization)."""

from __future__ import annotations

import functools
import weakref
from collections import OrderedDict
from typing import Any, Mapping

_ORIG_MOD_TOKEN = "_orig_mod."
_COMPAT_INSTALLED = False
_LIVE_OPTIMIZERS: weakref.WeakSet[Any] = weakref.WeakSet()


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
    metadata = getattr(state_dict, "_metadata", None)
    if isinstance(metadata, dict):
        prefixed._metadata = {  # type: ignore[attr-defined]
            (
                _ORIG_MOD_TOKEN + k
                if isinstance(k, str) and k and not k.startswith(_ORIG_MOD_TOKEN)
                else k
            ): v
            for k, v in metadata.items()
        }
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
    """Ensure 4D/5D channels_last tensors in state_dict are serialized in standard contiguous layout."""
    out: OrderedDict[str, Any] = OrderedDict()
    for key, val in state_dict.items():
        if hasattr(val, "dim") and hasattr(val, "is_contiguous") and val.dim() in (4, 5) and not val.is_contiguous():
            out[key] = val.contiguous()
        else:
            out[key] = val
    metadata = getattr(state_dict, "_metadata", None)
    if isinstance(metadata, dict):
        out._metadata = dict(metadata)  # type: ignore[attr-defined]
    return out


def _ensure_contiguous_optimizer_state_dict(state_dict: Mapping[str, Any], opt_self: Any) -> dict[str, Any]:
    """Normalize 4D/5D optimizer state buffers to contiguous, move injected CUDA `step` to CPU, and strip flags."""
    out = dict(state_dict)
    injected_fused = bool(getattr(opt_self, "_ptopt_injected_fused", False))
    injected_foreach = bool(getattr(opt_self, "_ptopt_injected_foreach", False))
    injected_capturable = bool(getattr(opt_self, "_ptopt_injected_capturable", False))

    raw_state = state_dict.get("state")
    if isinstance(raw_state, Mapping):
        clean_state: dict[Any, Any] = {}
        for param_id, p_state in raw_state.items():
            if isinstance(p_state, Mapping):
                clean_p_state: dict[Any, Any] = {}
                for sk, sv in p_state.items():
                    if hasattr(sv, "dim") and hasattr(sv, "is_contiguous") and sv.dim() in (4, 5) and not sv.is_contiguous():
                        clean_p_state[sk] = sv.contiguous()
                    elif injected_capturable and sk == "step" and getattr(sv, "is_cuda", False) and hasattr(sv, "cpu"):
                        clean_p_state[sk] = sv.cpu()
                    else:
                        clean_p_state[sk] = sv
                clean_state[param_id] = clean_p_state
            else:
                clean_state[param_id] = p_state
        out["state"] = clean_state

    raw_groups = state_dict.get("param_groups")
    if isinstance(raw_groups, list):
        if injected_fused or injected_foreach or injected_capturable:
            clean_groups: list[Any] = []
            for grp in raw_groups:
                if isinstance(grp, Mapping):
                    grp_copy = dict(grp)
                    if injected_fused and "fused" in grp_copy:
                        grp_copy["fused"] = False
                    if injected_foreach and "foreach" in grp_copy:
                        grp_copy["foreach"] = None
                    if injected_capturable and "capturable" in grp_copy:
                        grp_copy["capturable"] = False
                    clean_groups.append(grp_copy)
                else:
                    clean_groups.append(grp)
            out["param_groups"] = clean_groups
    return out


def sync_optimizer_state_memory_format(opt_self: Any, torch_mod: Any) -> int:
    """Align 4D optimizer state buffers (exp_avg/exp_avg_sq/momentum) with parameter layouts."""
    tensor_cls = getattr(torch_mod, "Tensor", None)
    channels_last_fmt = getattr(torch_mod, "channels_last", None)
    contiguous_fmt = getattr(torch_mod, "contiguous_format", None)
    if tensor_cls is None or contiguous_fmt is None:
        return 0

    opt_state = getattr(opt_self, "state", None)
    if not isinstance(opt_state, Mapping):
        return 0

    aligned = 0
    for grp in getattr(opt_self, "param_groups", ()):
        if not isinstance(grp, Mapping):
            continue
        for param in grp.get("params", ()):
            if not isinstance(param, tensor_cls) or param.dim() != 4:
                continue
            target_fmt = (
                channels_last_fmt
                if (
                    channels_last_fmt is not None
                    and param.is_contiguous(memory_format=channels_last_fmt)
                    and not param.is_contiguous()
                )
                else contiguous_fmt
            )
            p_grad = getattr(param, "grad", None)
            if (
                isinstance(p_grad, tensor_cls)
                and p_grad.dim() == 4
                and p_grad.shape == param.shape
                and p_grad.stride() != param.stride()
            ):
                param.grad = p_grad.contiguous(memory_format=target_fmt)
                aligned += 1

            p_state = opt_state.get(param)
            if not isinstance(p_state, dict):
                continue
            for sk, sv in list(p_state.items()):
                if (
                    isinstance(sv, tensor_cls)
                    and sv.dim() == 4
                    and sv.shape == param.shape
                    and sv.stride() != param.stride()
                ):
                    p_state[sk] = sv.contiguous(memory_format=target_fmt)
                    aligned += 1
    return aligned


def sync_all_live_optimizers(torch_mod: Any) -> int:
    """Align 4D optimizer state buffers across all live tracked optimizers."""
    total = 0
    for opt in list(_LIVE_OPTIMIZERS):
        total += sync_optimizer_state_memory_format(opt, torch_mod)
    return total


def install_checkpoint_compat_hooks(torch_mod: Any) -> bool:
    """Install idempotent state_dict/load_state_dict compatibility hooks on `nn.Module` and `optim.Optimizer`."""
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

    opt_cls = getattr(getattr(torch_mod, "optim", None), "Optimizer", None)
    if opt_cls is not None and hasattr(opt_cls, "state_dict"):
        orig_opt_init = opt_cls.__init__
        orig_opt_sd = opt_cls.state_dict
        orig_opt_load_sd = getattr(opt_cls, "load_state_dict", None)

        @functools.wraps(orig_opt_init)
        def _compat_opt_init(opt_self: Any, *args: Any, **kwargs: Any) -> None:
            orig_opt_init(opt_self, *args, **kwargs)
            try:
                _LIVE_OPTIMIZERS.add(opt_self)
            except TypeError:
                pass

        @functools.wraps(orig_opt_sd)
        def _compat_opt_state_dict(opt_self: Any, *args: Any, **kwargs: Any) -> Any:
            sd = orig_opt_sd(opt_self, *args, **kwargs)
            return _ensure_contiguous_optimizer_state_dict(sd, opt_self)

        opt_cls._ptopt_orig_init = orig_opt_init  # type: ignore[attr-defined]
        opt_cls.__init__ = _compat_opt_init  # type: ignore[method-assign]
        opt_cls._ptopt_orig_state_dict = orig_opt_sd  # type: ignore[attr-defined]
        opt_cls.state_dict = _compat_opt_state_dict  # type: ignore[method-assign]

        if callable(orig_opt_load_sd):
            @functools.wraps(orig_opt_load_sd)
            def _compat_opt_load_state_dict(
                opt_self: Any,
                state_dict: Mapping[str, Any],
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                res = orig_opt_load_sd(opt_self, state_dict, *args, **kwargs)
                try:
                    _LIVE_OPTIMIZERS.add(opt_self)
                except TypeError:
                    pass
                if getattr(opt_self, "_ptopt_injected_fused", False):
                    default_foreach = getattr(opt_self, "defaults", {}).get("foreach")
                    for grp in getattr(opt_self, "param_groups", ()):
                        if isinstance(grp, dict) and "fused" in grp:
                            grp["fused"] = True
                            if "foreach" in grp:
                                grp["foreach"] = default_foreach
                elif getattr(opt_self, "_ptopt_injected_foreach", False):
                    for grp in getattr(opt_self, "param_groups", ()):
                        if isinstance(grp, dict) and "foreach" in grp:
                            grp["foreach"] = True
                if getattr(opt_self, "_ptopt_injected_capturable", False):
                    for grp in getattr(opt_self, "param_groups", ()):
                        if isinstance(grp, dict) and "capturable" in grp:
                            grp["capturable"] = True

                # Normalize `step` tensor device across CPU<->CUDA checkpoint cross-loads.
                # PyTorch CUDA fused or capturable optimizers keep `step` on param.device; standard ones keep it on CPU.
                opt_state = getattr(opt_self, "state", None)
                tensor_cls = getattr(torch_mod, "Tensor", None)
                if isinstance(opt_state, dict) and tensor_cls is not None:
                    for grp in getattr(opt_self, "param_groups", ()):
                        if not isinstance(grp, dict):
                            continue
                        needs_param_dev = bool(grp.get("capturable", False) or grp.get("fused", False))
                        for param in grp.get("params", ()):
                            ps = opt_state.get(param)
                            if not isinstance(ps, dict):
                                continue
                            step_t = ps.get("step")
                            if isinstance(step_t, tensor_cls) and hasattr(param, "device"):
                                target_dev = param.device if needs_param_dev else "cpu"
                                if str(step_t.device) != str(target_dev):
                                    ps["step"] = step_t.to(device=target_dev)

                sync_optimizer_state_memory_format(opt_self, torch_mod)
                return res

            opt_cls._ptopt_orig_load_state_dict = orig_opt_load_sd  # type: ignore[attr-defined]
            opt_cls.load_state_dict = _compat_opt_load_state_dict  # type: ignore[method-assign]

    _COMPAT_INSTALLED = True
    return True


def uninstall_checkpoint_compat_hooks(torch_mod: Any) -> None:
    """Restore original `nn.Module` and `optim.Optimizer` state_dict hooks (for unit tests)."""
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
    opt_cls = getattr(getattr(torch_mod, "optim", None), "Optimizer", None)
    if opt_cls is not None:
        if hasattr(opt_cls, "_ptopt_orig_init"):
            opt_cls.__init__ = opt_cls._ptopt_orig_init  # type: ignore[method-assign]
            delattr(opt_cls, "_ptopt_orig_init")
        if hasattr(opt_cls, "_ptopt_orig_state_dict"):
            opt_cls.state_dict = opt_cls._ptopt_orig_state_dict  # type: ignore[method-assign]
            delattr(opt_cls, "_ptopt_orig_state_dict")
        if hasattr(opt_cls, "_ptopt_orig_load_state_dict"):
            opt_cls.load_state_dict = opt_cls._ptopt_orig_load_state_dict  # type: ignore[method-assign]
            delattr(opt_cls, "_ptopt_orig_load_state_dict")
    _COMPAT_INSTALLED = False
