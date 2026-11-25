"""Pass 3.3a: Safe Optimizer Fusion (`fused=True` / `foreach=True`) for CUDA optimizers."""

from __future__ import annotations

import functools
import inspect
import os
import sys
from typing import TYPE_CHECKING, Any

from ptopt.config import OptimizeConfig

if TYPE_CHECKING:
    from ptopt.hooks.manager import HookManager

_OPTIM_FUSION_INSTALLED = False
_TARGET_OPTIMIZERS = (
    "Adam",
    "AdamW",
    "SGD",
    "RMSprop",
    "Adagrad",
    "NAdam",
    "RAdam",
    "Adamax",
    "Adadelta",
    "ASGD",
)


def _iter_subclasses(base_cls: Any) -> list[Any]:
    seen: set[int] = set()
    out: list[Any] = []
    stack = [base_cls]
    while stack:
        cls = stack.pop()
        for sub in getattr(cls, "__subclasses__", lambda: ())():
            cid = id(sub)
            if cid not in seen:
                seen.add(cid)
                out.append(sub)
                stack.append(sub)
    return out


def _materialize_param_groups(params: Any) -> tuple[Any, list[Any]]:
    """Materialize generator/iterable params once and return (materialized_params, flat_tensors)."""
    if isinstance(params, (list, tuple)):
        items = list(params)
    else:
        items = list(params)

    flat_tensors: list[Any] = []
    if items and isinstance(items[0], dict):
        materialized_groups: list[dict[str, Any]] = []
        for group in items:
            grp = dict(group)
            grp_params = list(grp.get("params", []))
            grp["params"] = grp_params
            flat_tensors.extend(grp_params)
            materialized_groups.append(grp)
        return materialized_groups, flat_tensors

    flat_tensors.extend(items)
    return items, flat_tensors


def check_fused_eligibility(tensors: list[Any], torch_mod: Any) -> tuple[bool, str]:
    """Return (eligible, reason) for `fused=True` optimizer execution."""
    cuda_avail = False
    try:
        cuda_avail = bool(getattr(torch_mod, "cuda", None) and torch_mod.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_avail = False

    if not cuda_avail:
        return False, "CUDA unavailable"
    if not tensors:
        return False, "No parameters"

    for p in tensors:
        if not getattr(p, "is_cuda", False):
            return False, "Non-CUDA parameter tensor detected"
        is_fp = getattr(p, "is_floating_point", None)
        if callable(is_fp) and not is_fp():
            return False, "Non-floating-point parameter tensor detected"
        is_complex = getattr(p, "is_complex", None)
        if callable(is_complex) and is_complex():
            return False, "Complex parameter tensor not supported by fused optimizer"
    return True, "All parameters eligible for CUDA fused optimizer"


def install_optimizer_fusion_hook(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any,
) -> bool:
    """Install idempotent __init__ hooks on Adam, AdamW, and SGD."""
    global _OPTIM_FUSION_INSTALLED  # noqa: PLW0603
    if _OPTIM_FUSION_INSTALLED or torch_mod is None:
        return False

    optim_mod = getattr(torch_mod, "optim", None)
    if optim_mod is None:
        return False

    candidate_classes: list[tuple[str, Any]] = []
    seen_cls_ids: set[int] = set()
    for cls_name in _TARGET_OPTIMIZERS:
        opt_cls = getattr(optim_mod, cls_name, None)
        if opt_cls is not None and id(opt_cls) not in seen_cls_ids:
            seen_cls_ids.add(id(opt_cls))
            candidate_classes.append((cls_name, opt_cls))

    opt_base = getattr(optim_mod, "Optimizer", None)
    if opt_base is not None:
        for sub_cls in _iter_subclasses(opt_base):
            if id(sub_cls) not in seen_cls_ids and "__init__" in getattr(sub_cls, "__dict__", {}):
                seen_cls_ids.add(id(sub_cls))
                candidate_classes.append((getattr(sub_cls, "__name__", "CustomOptimizer"), sub_cls))

    def _try_patch_opt_cls(cls_name: str, opt_cls: Any) -> bool:
        if opt_cls is None or "_ptopt_orig_init" in getattr(opt_cls, "__dict__", {}):
            return False
        orig_init = opt_cls.__init__
        if getattr(orig_init, "_ptopt_is_fusion_init", False):
            return False
        try:
            sig = inspect.signature(orig_init)
        except (TypeError, ValueError):
            return False
        supports_fused = "fused" in sig.parameters
        supports_foreach = "foreach" in sig.parameters
        supports_capturable = "capturable" in sig.parameters
        if not (supports_fused or supports_foreach or supports_capturable):
            return False

        def _make_wrapped_init(
            target_cls_name: str,
            base_init: Any,
            has_fused: bool,
            has_foreach: bool,
            has_capturable: bool,
        ) -> Any:
            @functools.wraps(base_init)
            def _wrapped_init(opt_self: Any, params: Any, *args: Any, **kwargs: Any) -> None:
                mat_params, tensors = _materialize_param_groups(params)
                user_set_fused = "fused" in kwargs
                user_set_foreach = "foreach" in kwargs
                user_set_capturable = "capturable" in kwargs

                injected_kwargs = dict(kwargs)
                applied_mode: str | None = None
                applied_capturable = False

                if not user_set_fused and not user_set_foreach:
                    if cfg.optimizer.fused and has_fused:
                        ok, reason = check_fused_eligibility(tensors, torch_mod)
                        if ok:
                            injected_kwargs["fused"] = True
                            applied_mode = "fused=True"
                        else:
                            manager.record_event(
                                f"optimizer.{target_cls_name}",
                                "SKIPPED",
                                f"fused=True skipped ({reason}); using foreach/default",
                            )
                            if cfg.optimizer.foreach and has_foreach:
                                injected_kwargs["foreach"] = True
                                applied_mode = "foreach=True"
                    elif cfg.optimizer.foreach and has_foreach:
                        injected_kwargs["foreach"] = True
                        applied_mode = "foreach=True"

                # Guardrail #5a: opt-in capturable=True only via PTOPT_OPTIM_CAPTURABLE=1
                if (
                    not user_set_capturable
                    and has_capturable
                    and os.environ.get("PTOPT_OPTIM_CAPTURABLE") == "1"
                ):
                    ok_cap, _ = check_fused_eligibility(tensors, torch_mod)
                    if ok_cap:
                        injected_kwargs["capturable"] = True
                        applied_capturable = True

                try:
                    base_init(opt_self, mat_params, *args, **injected_kwargs)
                    if applied_mode == "fused=True":
                        opt_self._ptopt_injected_fused = True
                    elif applied_mode == "foreach=True":
                        opt_self._ptopt_injected_foreach = True
                    if applied_capturable:
                        opt_self._ptopt_injected_capturable = True
                    if applied_mode is not None or applied_capturable:
                        parts = ([applied_mode] if applied_mode else []) + (
                            ["capturable=True"] if applied_capturable else []
                        )
                        manager.record_event(f"optimizer.{target_cls_name}", "APPLIED", ", ".join(parts))
                except Exception as exc:  # noqa: BLE001
                    if (applied_mode is None and not applied_capturable) or manager.strict:
                        raise
                    manager.record_event(
                        f"optimizer.{target_cls_name}",
                        "FALLBACK",
                        f"{applied_mode or 'capturable=True'} init failed; reverted to user defaults",
                        str(exc),
                    )
                    base_init(opt_self, mat_params, *args, **kwargs)

            _wrapped_init._ptopt_is_fusion_init = True
            return _wrapped_init

        opt_cls.__init__ = _make_wrapped_init(
            cls_name, orig_init, supports_fused, supports_foreach, supports_capturable
        )
        opt_cls._ptopt_orig_init = orig_init
        patched_classes.append((opt_cls, orig_init))
        return True

    patched_classes: list[tuple[Any, Any]] = []
    for cls_name, opt_cls in candidate_classes:
        _try_patch_opt_cls(cls_name, opt_cls)

    if opt_base is not None:
        orig_init_subclass = opt_base.__dict__.get("__init_subclass__")

        @classmethod  # type: ignore[misc]
        def _ptopt_opt_init_subclass(cls: Any, **kwargs: Any) -> None:
            if orig_init_subclass is not None:
                fn = orig_init_subclass.__func__ if hasattr(orig_init_subclass, "__func__") else orig_init_subclass
                fn(cls, **kwargs)
            else:
                super(opt_base, cls).__init_subclass__(**kwargs)
            if "__init__" in getattr(cls, "__dict__", {}):
                _try_patch_opt_cls(getattr(cls, "__name__", "CustomOptimizer"), cls)

        opt_base._ptopt_fusion_orig_init_subclass = orig_init_subclass
        opt_base._ptopt_fusion_has_init_subclass = "__init_subclass__" in opt_base.__dict__
        opt_base.__init_subclass__ = _ptopt_opt_init_subclass

    if not patched_classes:
        if opt_base is not None and hasattr(opt_base, "_ptopt_fusion_orig_init_subclass"):
            if opt_base._ptopt_fusion_has_init_subclass:
                opt_base.__init_subclass__ = opt_base._ptopt_fusion_orig_init_subclass
            elif "__init_subclass__" in opt_base.__dict__:
                delattr(opt_base, "__init_subclass__")
            delattr(opt_base, "_ptopt_fusion_orig_init_subclass")
            delattr(opt_base, "_ptopt_fusion_has_init_subclass")
        return False
    optim_mod._ptopt_fusion_patched_classes = patched_classes
    _OPTIM_FUSION_INSTALLED = True
    return True


def uninstall_optimizer_fusion_hook(torch_mod: Any) -> None:
    """Restore original optimizer __init__ methods (for unit tests)."""
    global _OPTIM_FUSION_INSTALLED  # noqa: PLW0603
    if not _OPTIM_FUSION_INSTALLED or torch_mod is None:
        return
    optim_mod = getattr(torch_mod, "optim", None)
    if optim_mod is not None:
        opt_base = getattr(optim_mod, "Optimizer", None)
        if opt_base is not None and hasattr(opt_base, "_ptopt_fusion_orig_init_subclass"):
            if getattr(opt_base, "_ptopt_fusion_has_init_subclass", False):
                opt_base.__init_subclass__ = opt_base._ptopt_fusion_orig_init_subclass
            elif "__init_subclass__" in getattr(opt_base, "__dict__", {}):
                delattr(opt_base, "__init_subclass__")
            delattr(opt_base, "_ptopt_fusion_orig_init_subclass")
            delattr(opt_base, "_ptopt_fusion_has_init_subclass")
        for opt_cls, orig_init in getattr(optim_mod, "_ptopt_fusion_patched_classes", ()):
            try:
                opt_cls.__init__ = orig_init
                if "_ptopt_orig_init" in getattr(opt_cls, "__dict__", {}):
                    delattr(opt_cls, "_ptopt_orig_init")
            except Exception:  # noqa: BLE001
                pass
        if hasattr(optim_mod, "_ptopt_fusion_patched_classes"):
            delattr(optim_mod, "_ptopt_fusion_patched_classes")
        for cls_name in _TARGET_OPTIMIZERS:
            opt_cls = getattr(optim_mod, cls_name, None)
            if opt_cls is not None and "_ptopt_orig_init" in getattr(opt_cls, "__dict__", {}):
                opt_cls.__init__ = opt_cls._ptopt_orig_init
                delattr(opt_cls, "_ptopt_orig_init")
    _OPTIM_FUSION_INSTALLED = False


def apply_optimizer_fusion_pass(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any = None,
) -> str:
    """Enable optimizer fusion pass when `cfg.optimizer.fused` or `foreach` is enabled."""
    if not cfg.optimizer.fused and not cfg.optimizer.foreach:
        return "SKIPPED: optimizer.fused and foreach disabled"
    tmod = torch_mod if torch_mod is not None else sys.modules.get("torch")
    if tmod is None:
        return "SKIPPED: torch not loaded"
    installed = install_optimizer_fusion_hook(cfg, manager, tmod)
    return "Optimizer fusion hook installed" if installed else "Optimizer fusion hook already active"
