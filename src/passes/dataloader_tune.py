"""Pass 3.2: Safe DataLoader auto-tuning with /dev/shm, CUDA-tensor, and pickle-ability guards."""

from __future__ import annotations

import functools
import inspect
import io
import os
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from src.config import OptimizeConfig

if TYPE_CHECKING:
    from src.hooks.manager import HookManager

_DATALOADER_HOOK_INSTALLED = False


def read_shm_total_mb() -> float | None:
    """Read /dev/shm total size in MB, or None if unavailable."""
    shm_path = Path("/dev/shm")
    if not shm_path.exists():
        return None
    try:
        st = os.statvfs(shm_path)
        return round((st.f_frsize * st.f_blocks) / (1024.0 * 1024.0), 1)
    except OSError:
        return None


def has_cuda_tensors(dataset: Any, torch_mod: Any) -> bool:
    """Detect whether a TensorDataset-like object already stores tensors on CUDA."""
    tensor_cls = getattr(torch_mod, "Tensor", None)
    if tensor_cls is None:
        return False
    tensors = getattr(dataset, "tensors", None)
    if isinstance(tensors, (tuple, list)):
        for item in tensors:
            if isinstance(item, tensor_cls) and getattr(item, "is_cuda", False):
                return True
    return False


def is_picklable_fast(obj: Any, torch_mod: Any = None) -> tuple[bool, str | None]:
    """Check picklability of dataset/collate_fn without serializing large Tensor storage."""
    if obj is None:
        return True, None
    tensor_cls = getattr(torch_mod, "Tensor", None) if torch_mod is not None else None

    class _FastCheckPickler(pickle.Pickler):
        def persistent_id(self, value: Any) -> Any:
            if tensor_cls is not None and isinstance(value, tensor_cls):
                return f"tensor:{id(value)}"
            return None

    buf = io.BytesIO()
    try:
        _FastCheckPickler(buf, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def decide_dataloader_params(
    *,
    cfg: OptimizeConfig,
    dataset: Any,
    collate_fn: Any,
    worker_init_fn: Any,
    orig_num_workers: int,
    orig_pin_memory: bool,
    orig_persistent_workers: bool,
    orig_prefetch_factor: Any,
    torch_mod: Any,
    shm_mb: float | None,
    cpu_cores: int | None,
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Compute safe tuned DataLoader parameters and audit notes."""
    notes: list[tuple[str, str]] = []
    cuda_avail = False
    try:
        cuda_avail = bool(getattr(torch_mod, "cuda", None) and torch_mod.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_avail = False

    cuda_ds = has_cuda_tensors(dataset, torch_mod)
    if cuda_ds:
        notes.append(("SKIPPED", "Dataset holds CUDA tensors; keeping DataLoader settings unchanged"))
        return {
            "num_workers": orig_num_workers,
            "pin_memory": orig_pin_memory,
            "persistent_workers": orig_persistent_workers,
            "prefetch_factor": orig_prefetch_factor,
            "bumped": False,
        }, notes

    new_pin_memory = orig_pin_memory or bool(cfg.dataloader.pin_memory and cuda_avail)
    new_workers = orig_num_workers
    new_persistent = orig_persistent_workers
    new_prefetch = orig_prefetch_factor
    bumped = False

    # Only auto-bump when user kept default num_workers == 0
    if orig_num_workers == 0:
        target = cfg.dataloader.num_workers
        if target is None:
            cores = cpu_cores or os.cpu_count() or 2
            target = min(4, max(2, cores // 2))

        if target > 0:
            if shm_mb is not None and shm_mb < 1024.0:
                notes.append(("SKIPPED", f"/dev/shm={shm_mb}MB < 1024MB: keeping num_workers=0"))
                target = 0
            elif shm_mb is not None and shm_mb < 2048.0 and target > 2:
                notes.append(("APPLIED", f"/dev/shm={shm_mb}MB < 2048MB: capping num_workers {target} -> 2"))
                target = 2

        if target > 0:
            for label, obj in (("dataset", dataset), ("collate_fn", collate_fn), ("worker_init_fn", worker_init_fn)):
                ok, err = is_picklable_fast(obj, torch_mod=torch_mod)
                if not ok:
                    notes.append(("FALLBACK", f"Unpicklable {label} ({err}); keeping num_workers=0"))
                    target = 0
                    break

        if target > 0:
            new_workers = int(target)
            bumped = True
            if cfg.dataloader.persistent_workers:
                new_persistent = True
            if cfg.dataloader.prefetch_factor is not None:
                new_prefetch = int(cfg.dataloader.prefetch_factor)
            notes.append((
                "APPLIED",
                f"Tuned DataLoader: num_workers 0->{new_workers}, pin_memory={new_pin_memory}, persistent={new_persistent}",
            ))
    else:
        if new_pin_memory != orig_pin_memory:
            notes.append(("APPLIED", f"Preserved user num_workers={orig_num_workers}; enabled pin_memory=True"))

    # Keep PyTorch invariant: persistent_workers and prefetch_factor require num_workers > 0
    if new_workers == 0:
        new_persistent = False
        new_prefetch = None

    return {
        "num_workers": new_workers,
        "pin_memory": new_pin_memory,
        "persistent_workers": new_persistent,
        "prefetch_factor": new_prefetch,
        "bumped": bumped,
    }, notes


def install_dataloader_hook(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any,
    shm_probe_fn: Callable[[], float | None] | None = None,
) -> bool:
    """Install idempotent DataLoader.__init__ and __iter__ fallback hooks."""
    global _DATALOADER_HOOK_INSTALLED  # noqa: PLW0603
    if _DATALOADER_HOOK_INSTALLED or torch_mod is None:
        return False

    data_mod = getattr(getattr(torch_mod, "utils", None), "data", None)
    dl_cls = getattr(data_mod, "DataLoader", None)
    if dl_cls is None:
        return False

    orig_init = dl_cls.__init__
    orig_iter = dl_cls.__iter__
    sig = inspect.signature(orig_init)
    shm_reader = shm_probe_fn or read_shm_total_mb

    @functools.wraps(orig_init)
    def _tuned_init(dl_self: Any, *args: Any, **kwargs: Any) -> None:
        try:
            bound = sig.bind(dl_self, *args, **kwargs)
            bound.apply_defaults()
            b_args = bound.arguments
            orig_workers = int(b_args.get("num_workers", 0))
            orig_pin = bool(b_args.get("pin_memory", False))
            orig_persist = bool(b_args.get("persistent_workers", False))
            orig_prefetch = b_args.get("prefetch_factor", None)

            tuned, notes = decide_dataloader_params(
                cfg=cfg,
                dataset=b_args.get("dataset"),
                collate_fn=b_args.get("collate_fn"),
                worker_init_fn=b_args.get("worker_init_fn"),
                orig_num_workers=orig_workers,
                orig_pin_memory=orig_pin,
                orig_persistent_workers=orig_persist,
                orig_prefetch_factor=orig_prefetch,
                torch_mod=torch_mod,
                shm_mb=shm_reader(),
                cpu_cores=os.cpu_count(),
            )
            for status, reason in notes:
                manager.record_event("dataloader.auto_tune", status, reason)

            b_args["num_workers"] = tuned["num_workers"]
            b_args["pin_memory"] = tuned["pin_memory"]
            b_args["persistent_workers"] = tuned["persistent_workers"]
            b_args["prefetch_factor"] = tuned["prefetch_factor"]
            dl_self._ptopt_orig_dl = {
                "num_workers": orig_workers,
                "pin_memory": orig_pin,
                "persistent_workers": orig_persist,
                "prefetch_factor": orig_prefetch,
                "bumped": tuned["bumped"],
            }
            orig_init(*bound.args, **bound.kwargs)
            return
        except Exception as exc:  # noqa: BLE001
            if manager.strict:
                raise
            manager.record_event("dataloader.auto_tune", "FALLBACK", "DataLoader.__init__ tuning fallback", str(exc))
            orig_init(dl_self, *args, **kwargs)

    @functools.wraps(orig_iter)
    def _safe_iter(dl_self: Any) -> Any:
        try:
            return orig_iter(dl_self)
        except Exception as exc:  # noqa: BLE001
            orig_meta = getattr(dl_self, "_ptopt_orig_dl", None)
            if not orig_meta or not orig_meta.get("bumped"):
                raise
            manager.record_event(
                "dataloader.auto_tune",
                "FALLBACK",
                f"Worker iterator failed ({type(exc).__name__}); reverting to num_workers=0",
                str(exc),
            )
            dl_self.num_workers = orig_meta["num_workers"]
            dl_self.pin_memory = orig_meta["pin_memory"]
            dl_self.persistent_workers = orig_meta["persistent_workers"]
            dl_self.prefetch_factor = orig_meta["prefetch_factor"]
            orig_meta["bumped"] = False
            return orig_iter(dl_self)

    dl_cls.__init__ = _tuned_init  # type: ignore[method-assign]
    dl_cls.__iter__ = _safe_iter  # type: ignore[method-assign]
    dl_cls._ptopt_orig_init = orig_init  # type: ignore[attr-defined]
    dl_cls._ptopt_orig_iter = orig_iter  # type: ignore[attr-defined]
    _DATALOADER_HOOK_INSTALLED = True
    return True


def uninstall_dataloader_hook(torch_mod: Any) -> None:
    """Restore original DataLoader.__init__ and __iter__ (for unit tests)."""
    global _DATALOADER_HOOK_INSTALLED  # noqa: PLW0603
    if not _DATALOADER_HOOK_INSTALLED or torch_mod is None:
        return
    dl_cls = getattr(getattr(getattr(torch_mod, "utils", None), "data", None), "DataLoader", None)
    if dl_cls is not None:
        if hasattr(dl_cls, "_ptopt_orig_init"):
            dl_cls.__init__ = dl_cls._ptopt_orig_init  # type: ignore[method-assign]
            delattr(dl_cls, "_ptopt_orig_init")
        if hasattr(dl_cls, "_ptopt_orig_iter"):
            dl_cls.__iter__ = dl_cls._ptopt_orig_iter  # type: ignore[method-assign]
            delattr(dl_cls, "_ptopt_orig_iter")
    _DATALOADER_HOOK_INSTALLED = False


def apply_dataloader_tune_pass(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any = None,
) -> str:
    """Enable DataLoader auto-tuning pass when `cfg.dataloader.auto_tune` is True."""
    if not cfg.dataloader.auto_tune:
        return "SKIPPED: dataloader.auto_tune is disabled"
    tmod = torch_mod if torch_mod is not None else sys.modules.get("torch")
    if tmod is None:
        return "SKIPPED: torch not loaded"
    installed = install_dataloader_hook(cfg, manager, tmod)
    return "DataLoader auto-tune hook installed" if installed else "DataLoader hook already active"
