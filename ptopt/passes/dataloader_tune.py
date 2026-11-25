"""Pass 3.2: Safe DataLoader auto-tuning with /dev/shm, CUDA-tensor, and pickle-ability guards."""

from __future__ import annotations

import copy
import functools
import inspect
import io
import os
import pickle
import sys
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ptopt.config import OptimizeConfig
from ptopt.passes.memory_allocator import probe_cuda_vram_headroom

if TYPE_CHECKING:
    from ptopt.hooks.manager import HookManager

_DATALOADER_HOOK_INSTALLED = False


class _PTOPTWorkerInitWrapper:
    """Picklable worker_init_fn wrapper that bounds OpenMP/MKL/intra-op threads per worker."""

    def __init__(self, base_fn: Callable[[int], Any] | None) -> None:
        self.base_fn = base_fn

    def __call__(self, worker_id: int) -> Any:
        # Overwrite inherited OMP/MKL settings from fork() so workers don't oversubscribe cores
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        try:
            tmod = sys.modules.get("torch")
            set_threads = getattr(tmod, "set_num_threads", None) if tmod is not None else None
            if callable(set_threads):
                set_threads(1)
        except Exception:  # noqa: BLE001
            pass
        if self.base_fn is not None:
            return self.base_fn(worker_id)
        return None


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


def _iter_wrapped_datasets(dataset: Any, *, max_depth: int = 4) -> list[Any]:
    """Unwrap `Subset` and `ConcatDataset` containers safely."""
    out: list[Any] = []
    stack: list[tuple[Any, int]] = [(dataset, 0)]
    seen: set[int] = set()
    while stack:
        cur, depth = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        out.append(cur)
        if depth >= max_depth:
            continue
        inner = getattr(cur, "dataset", None)
        if inner is not None:
            stack.append((inner, depth + 1))
        sub_list = getattr(cur, "datasets", None)
        if isinstance(sub_list, (list, tuple)):
            for ds in sub_list:
                stack.append((ds, depth + 1))
    return out


def has_cuda_tensors(dataset: Any, torch_mod: Any) -> bool:
    """Detect whether a TensorDataset (or Subset/ConcatDataset wrapper) holds CUDA tensors."""
    tensor_cls = getattr(torch_mod, "Tensor", None)
    if tensor_cls is None or dataset is None:
        return False
    for ds in _iter_wrapped_datasets(dataset):
        tensors = getattr(ds, "tensors", None)
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


def _move_batch_to_cuda(obj: Any, torch_mod: Any, device: Any, tracked_tensors: list[Any]) -> Any:
    """Recursively stage CPU tensors in a batch onto CUDA (`non_blocking=True`), leaving custom/CUDA items intact."""
    tensor_cls = getattr(torch_mod, "Tensor", None)
    if tensor_cls is not None and isinstance(obj, tensor_cls):
        dev = getattr(obj, "device", None)
        if not getattr(obj, "is_cuda", False) and getattr(dev, "type", "") == "cpu":
            dst = obj.to(device=device, non_blocking=True)
            tracked_tensors.append(dst)
            return dst
        return obj
    if isinstance(obj, tuple):
        moved = [_move_batch_to_cuda(v, torch_mod, device, tracked_tensors) for v in obj]
        if hasattr(obj, "_fields"):
            return type(obj)(*moved)
        return tuple(moved)
    if isinstance(obj, list):
        return [_move_batch_to_cuda(v, torch_mod, device, tracked_tensors) for v in obj]
    if type(obj) is dict:
        return {k: _move_batch_to_cuda(v, torch_mod, device, tracked_tensors) for k, v in obj.items()}
    if isinstance(obj, dict):
        # Preserve dict subclasses (e.g. HuggingFace BatchEncoding, EasyDict)
        try:
            cloned = copy.copy(obj)
            for k, v in list(obj.items()):
                cloned[k] = _move_batch_to_cuda(v, torch_mod, device, tracked_tensors)
            return cloned
        except Exception:  # noqa: BLE001
            return {k: _move_batch_to_cuda(v, torch_mod, device, tracked_tensors) for k, v in obj.items()}
    return obj


def compute_adaptive_prefetch_depth(
    torch_mod: Any,
    *,
    max_depth: int | None = None,
    backoff_ratio: float | None = None,
) -> tuple[int, dict[str, float] | None]:
    """Choose GPU prefetch queue depth (0..4) dynamically from live VRAM headroom."""
    raw_override = os.environ.get("PTOPT_CUDA_PREFETCH_DEPTH")
    headroom = probe_cuda_vram_headroom(torch_mod)
    if raw_override is not None:
        try:
            return max(0, min(4, int(raw_override))), headroom
        except ValueError:
            pass

    if max_depth is None:
        try:
            max_depth = max(1, min(4, int(os.environ.get("PTOPT_CUDA_PREFETCH_MAX_DEPTH", "3"))))
        except ValueError:
            max_depth = 3

    if backoff_ratio is None:
        try:
            backoff_ratio = float(os.environ.get("PTOPT_CUDA_PREFETCH_BACKOFF_RATIO", "0.78"))
        except ValueError:
            backoff_ratio = 0.78

    if headroom is None:
        return 1, None

    used = float(headroom["used_ratio"])
    if used >= backoff_ratio:
        return 0, headroom
    if used < 0.50:
        return min(max_depth, 3), headroom
    if used < 0.68:
        return min(max_depth, 2), headroom
    return 1, headroom


def _get_or_init_prefetch_stats(torch_mod: Any) -> dict[str, Any]:
    stats = getattr(torch_mod, "_ptopt_prefetch_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "loaders_wrapped": 0,
            "batches_prefetched": 0,
            "batches_served": 0,
            "initial_depth": 0,
            "current_depth": 0,
            "max_queue_depth": 0,
            "backoff_events": 0,
            "last_vram_used_ratio": None,
            "last_vram_free_mb": None,
        }
        try:
            torch_mod._ptopt_prefetch_stats = stats
        except Exception:  # noqa: BLE001
            pass
    return stats


class _AsyncCudaPrefetchIter:
    """Adaptive multi-batch GPU prefetch ring queue using a dedicated CUDA copy stream with VRAM back-off."""

    def __init__(
        self,
        inner_iter: Any,
        torch_mod: Any,
        *,
        device: Any = None,
        stream: Any = None,
        manager: "HookManager | None" = None,
        max_depth: int | None = None,
    ) -> None:
        self._iter = inner_iter
        self._torch = torch_mod
        self._manager = manager
        self._max_depth = max_depth
        cuda_api = getattr(torch_mod, "cuda", None)
        if device is not None:
            self._device = device
        elif cuda_api is not None and hasattr(cuda_api, "current_device"):
            self._device = torch_mod.device("cuda", cuda_api.current_device())
        else:
            self._device = "cuda"
        if stream is not None:
            self._stream = stream
        else:
            cache = getattr(cuda_api, "_ptopt_prefetch_streams", None)
            if not isinstance(cache, dict):
                cache = {}
                try:
                    cuda_api._ptopt_prefetch_streams = cache
                except Exception:  # noqa: BLE001
                    pass
            dev_key = str(self._device)
            cached_stream = cache.get(dev_key)
            if cached_stream is None:
                cached_stream = cuda_api.Stream(device=self._device)
                cache[dev_key] = cached_stream
            self._stream = cached_stream
        self._queue: deque[tuple[Any, list[Any], Any]] = deque()
        self._exhausted = False
        self._backoff_logged = False
        self._stats = _get_or_init_prefetch_stats(torch_mod)
        self._stats["loaders_wrapped"] += 1

        init_depth, headroom = compute_adaptive_prefetch_depth(torch_mod, max_depth=self._max_depth)
        self._current_depth = init_depth
        self._stats["initial_depth"] = init_depth
        self._stats["current_depth"] = init_depth
        if headroom is not None:
            self._stats["last_vram_used_ratio"] = headroom["used_ratio"]
            self._stats["last_vram_free_mb"] = headroom["free_mb"]
        if init_depth > 0:
            self._fill_queue(init_depth)
        elif manager is not None:
            self._record_backoff(headroom)

    def _record_backoff(self, headroom: dict[str, float] | None) -> None:
        if self._backoff_logged:
            return
        self._backoff_logged = True
        self._stats["backoff_events"] += 1
        if self._manager is not None:
            pct = f"{headroom['used_ratio']:.1%}" if headroom is not None else "high"
            free_mb = f"{headroom['free_mb']:.1f}MB" if headroom is not None else "N/A"
            self._manager.record_event(
                "dataloader.cuda_prefetch",
                "FALLBACK",
                f"VRAM pressure ({pct} used, {free_mb} free): backed off GPU prefetch queue depth -> 0",
            )

    def _stage_one_batch(self) -> tuple[Any, list[Any], Any] | None:
        if self._exhausted:
            return None
        try:
            raw = next(self._iter)
        except StopIteration:
            self._exhausted = True
            return None

        tracked: list[Any] = []
        copy_event: Any = None
        try:
            with self._torch.cuda.stream(self._stream):
                staged = _move_batch_to_cuda(raw, self._torch, self._device, tracked)
                event_cls = getattr(getattr(self._torch, "cuda", None), "Event", None)
                if callable(event_cls) and tracked:
                    copy_event = event_cls()
                    copy_event.record(self._stream)
            self._stats["batches_prefetched"] += 1
            return staged, tracked, copy_event
        except Exception:  # noqa: BLE001
            return raw, [], None

    def _fill_queue(self, target_depth: int) -> None:
        while not self._exhausted and len(self._queue) < target_depth:
            item = self._stage_one_batch()
            if item is None:
                break
            self._queue.append(item)
        if len(self._queue) > int(self._stats["max_queue_depth"]):
            self._stats["max_queue_depth"] = len(self._queue)

    def __iter__(self) -> "_AsyncCudaPrefetchIter":
        return self

    def __next__(self) -> Any:
        if self._queue:
            cur_item, cur_tensors, cur_event = self._queue.popleft()
        elif self._exhausted:
            raise StopIteration
        else:
            on_demand = self._stage_one_batch()
            if on_demand is None:
                raise StopIteration
            cur_item, cur_tensors, cur_event = on_demand

        cuda_api = getattr(self._torch, "cuda", None)
        if cuda_api is not None and cur_tensors:
            try:
                cur_stream = cuda_api.current_stream(self._device)
            except TypeError:
                cur_stream = cuda_api.current_stream()
            wait_event_fn = getattr(cur_stream, "wait_event", None)
            if cur_event is not None and callable(wait_event_fn):
                wait_event_fn(cur_event)
            else:
                cur_stream.wait_stream(self._stream)
            for tensor in cur_tensors:
                record_fn = getattr(tensor, "record_stream", None)
                if callable(record_fn):
                    record_fn(cur_stream)
        self._stats["batches_served"] += 1
        served = int(self._stats["batches_served"])

        if served <= 2 or served % 8 == 0:
            new_depth, headroom = compute_adaptive_prefetch_depth(self._torch, max_depth=self._max_depth)
            self._current_depth = new_depth
            self._stats["current_depth"] = new_depth
            if headroom is not None:
                self._stats["last_vram_used_ratio"] = headroom["used_ratio"]
                self._stats["last_vram_free_mb"] = headroom["free_mb"]
            if new_depth == 0:
                self._record_backoff(headroom)
            else:
                self._backoff_logged = False
                self._fill_queue(new_depth)
        elif self._current_depth > 0:
            self._fill_queue(self._current_depth)
        return cur_item


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
        data_ns = getattr(getattr(torch_mod, "utils", None), "data", None)
        iterable_ds_cls = getattr(data_ns, "IterableDataset", None) if data_ns is not None else None
        tensor_ds_cls = getattr(data_ns, "TensorDataset", None) if data_ns is not None else None
        unwrapped_ds = _iter_wrapped_datasets(dataset)
        if iterable_ds_cls is not None and any(isinstance(d, iterable_ds_cls) for d in unwrapped_ds):
            notes.append(("SKIPPED", "IterableDataset detected; keeping num_workers=0 to avoid duplicate streams"))
            target = 0
        elif tensor_ds_cls is not None and unwrapped_ds and all(
            isinstance(d, tensor_ds_cls) or hasattr(d, "dataset") or hasattr(d, "datasets") for d in unwrapped_ds
        ) and any(isinstance(d, tensor_ds_cls) for d in unwrapped_ds):
            notes.append(("SKIPPED", "In-memory TensorDataset detected; keeping num_workers=0 to avoid IPC overhead"))
            target = 0
        else:
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


def _effective_prefetch_max_depth(cfg: OptimizeConfig) -> int | None:
    """Default to 3-batch lookahead when bandwidth_boost is enabled, 1 otherwise (unless env overrides)."""
    if os.environ.get("PTOPT_CUDA_PREFETCH_DEPTH") or os.environ.get("PTOPT_CUDA_PREFETCH_MAX_DEPTH"):
        return None
    return 3 if getattr(cfg.runtime, "bandwidth_boost", False) else 1


def _detect_dataloader_remainder(
    dataset: Any,
    batch_size: Any,
    drop_last: bool,
    batch_sampler: Any,
) -> tuple[bool, int | None, int | None]:
    """Return `(has_remainder, dataset_len, batch_size)` for map-style DataLoaders."""
    eff_bs = batch_size
    eff_drop_last = drop_last
    ds_or_sampler = dataset
    if batch_sampler is not None:
        eff_drop_last = bool(getattr(batch_sampler, "drop_last", eff_drop_last))
        bs_attr = getattr(batch_sampler, "batch_size", None)
        if bs_attr is not None:
            eff_bs = bs_attr
        sampler_attr = getattr(batch_sampler, "sampler", None)
        if sampler_attr is not None:
            ds_or_sampler = sampler_attr
    if eff_drop_last or not isinstance(eff_bs, int) or eff_bs <= 1:
        return False, None, eff_bs if isinstance(eff_bs, int) else None
    try:
        total_len = len(ds_or_sampler) if ds_or_sampler is not None else len(dataset)
    except (TypeError, NotImplementedError, AttributeError):
        return False, None, eff_bs
    if total_len > eff_bs and (total_len % eff_bs) != 0:
        return True, total_len, eff_bs
    return False, total_len, eff_bs


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

            ds_obj = b_args.get("dataset")
            is_cuda_ds = has_cuda_tensors(ds_obj, torch_mod)
            dl_self._ptopt_cuda_ds = is_cuda_ds
            has_rem, ds_len, eff_bs = _detect_dataloader_remainder(
                ds_obj,
                b_args.get("batch_size"),
                bool(b_args.get("drop_last", False)),
                b_args.get("batch_sampler"),
            )
            dl_self._ptopt_dl_has_remainder = bool(has_rem)
            dl_self._ptopt_dl_remainder_info = (ds_len, eff_bs)
            if has_rem:
                try:
                    torch_mod._ptopt_dataloader_has_remainder = True
                    torch_mod._ptopt_dataloader_remainder_info = (ds_len, eff_bs)
                except Exception:  # noqa: BLE001
                    pass

            if cfg.dataloader.auto_tune:
                tuned, notes = decide_dataloader_params(
                    cfg=cfg,
                    dataset=ds_obj,
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
                if tuned["bumped"] and int(tuned["num_workers"]) > 0:
                    orig_wif = b_args.get("worker_init_fn")
                    if not isinstance(orig_wif, _PTOPTWorkerInitWrapper):
                        b_args["worker_init_fn"] = _PTOPTWorkerInitWrapper(orig_wif)
                dl_self._ptopt_orig_dl = {
                    "num_workers": orig_workers,
                    "pin_memory": orig_pin,
                    "persistent_workers": orig_persist,
                    "prefetch_factor": orig_prefetch,
                    "bumped": tuned["bumped"],
                }

            cuda_ok = False
            try:
                cuda_ok = bool(getattr(torch_mod, "cuda", None) and torch_mod.cuda.is_available())
            except Exception:  # noqa: BLE001
                cuda_ok = False
            prefetch_req = bool(getattr(cfg.dataloader, "cuda_prefetch", False)) or os.environ.get("PTOPT_CUDA_PREFETCH") == "1"
            if prefetch_req and cuda_ok and not is_cuda_ds:
                if not b_args.get("pin_memory", False):
                    b_args["pin_memory"] = True
                dl_self._ptopt_cuda_prefetch = True
                max_d = _effective_prefetch_max_depth(cfg)
                init_depth, headroom = compute_adaptive_prefetch_depth(torch_mod, max_depth=max_d)
                hr_info = (
                    f"depth={init_depth}, free_vram={headroom['free_mb']:.1f}MB ({headroom['used_ratio']:.1%} used)"
                    if headroom is not None
                    else f"depth={init_depth}"
                )
                manager.record_event(
                    "dataloader.cuda_prefetch",
                    "APPLIED",
                    f"Adaptive CUDA copy-stream prefetch enabled ({hr_info})",
                )
            elif prefetch_req and not cuda_ok:
                manager.record_event("dataloader.cuda_prefetch", "SKIPPED", "CUDA unavailable for cuda_prefetch")

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
            torch_mod._ptopt_dataloader_has_remainder = bool(getattr(dl_self, "_ptopt_dl_has_remainder", False))
            if hasattr(dl_self, "_ptopt_dl_remainder_info"):
                torch_mod._ptopt_dataloader_remainder_info = dl_self._ptopt_dl_remainder_info
        except Exception:  # noqa: BLE001
            pass
        try:
            base_it = orig_iter(dl_self)
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
            base_it = orig_iter(dl_self)

        cuda_ok = False
        try:
            cuda_ok = bool(getattr(torch_mod, "cuda", None) and torch_mod.cuda.is_available())
        except Exception:  # noqa: BLE001
            cuda_ok = False
        prefetch_active = (
            getattr(dl_self, "_ptopt_cuda_prefetch", False)
            or bool(getattr(cfg.dataloader, "cuda_prefetch", False))
            or os.environ.get("PTOPT_CUDA_PREFETCH") == "1"
        )
        if prefetch_active and cuda_ok and not getattr(dl_self, "_ptopt_cuda_ds", False):
            # When active automatically via ptopt CLI (and not forced by PTOPT_CUDA_PREFETCH=1),
            # guard CPU-only scripts on GPU hosts by requiring CUDA context initialization first.
            is_init_fn = getattr(getattr(torch_mod, "cuda", None), "is_initialized", None)
            if (
                os.environ.get("PTOPT_ACTIVE") == "1"
                and os.environ.get("PTOPT_CUDA_PREFETCH") != "1"
                and callable(is_init_fn)
                and not is_init_fn()
            ):
                if not getattr(dl_self, "_ptopt_prefetch_cpu_skipped", False):
                    dl_self._ptopt_prefetch_cpu_skipped = True
                    manager.record_event(
                        "dataloader.cuda_prefetch",
                        "SKIPPED",
                        "CUDA context uninitialized at DataLoader.__iter__; keeping CPU batches",
                    )
                return base_it
            max_d = _effective_prefetch_max_depth(cfg)
            return _AsyncCudaPrefetchIter(base_it, torch_mod, manager=manager, max_depth=max_d)
        return base_it

    dl_cls.__init__ = _tuned_init  # type: ignore[method-assign]
    dl_cls.__iter__ = _safe_iter  # type: ignore[method-assign]
    dl_cls._ptopt_orig_init = orig_init  # type: ignore[attr-defined]
    dl_cls._ptopt_orig_iter = orig_iter  # type: ignore[attr-defined]

    # Auto-promote pinned CPU -> CUDA `.to(device)` / `.cuda()` calls to non_blocking=True
    tensor_cls = getattr(torch_mod, "Tensor", None)
    cuda_avail = False
    try:
        cuda_avail = bool(getattr(torch_mod, "cuda", None) and torch_mod.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_avail = False

    if cuda_avail and (cfg.dataloader.pin_memory or getattr(cfg.dataloader, "cuda_prefetch", False)) and tensor_cls is not None and hasattr(tensor_cls, "to"):
        orig_to = tensor_cls.to
        compiler_ns = getattr(torch_mod, "compiler", None)

        def _dynamo_compiling() -> bool:
            fn = getattr(compiler_ns, "is_compiling", None) if compiler_ns is not None else None
            return bool(fn()) if callable(fn) else False

        def _is_cuda_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
            dev = kwargs.get("device")
            if dev is None and len(args) == 1:
                dev = args[0]
            if isinstance(dev, str):
                return dev.startswith("cuda")
            return getattr(dev, "type", None) == "cuda"

        def _should_promote_non_blocking(t_self: Any, args: tuple[Any, ...], kwargs: dict[str, Any], cuda_target: bool) -> bool:
            if not cuda_target or "non_blocking" in kwargs or len(args) > 1:
                return False
            if getattr(t_self, "is_cuda", True) or _dynamo_compiling():
                return False
            dev = getattr(t_self, "device", None)
            return dev is not None and dev.type == "cpu" and t_self.is_pinned()

        @functools.wraps(orig_to)
        def _pinned_async_to(t_self: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(t_self, "is_cuda", False) or _dynamo_compiling():
                return orig_to(t_self, *args, **kwargs)
            if _should_promote_non_blocking(t_self, args, kwargs, _is_cuda_arg(args, kwargs)):
                kwargs["non_blocking"] = True
            return orig_to(t_self, *args, **kwargs)

        tensor_cls._ptopt_orig_to = orig_to  # type: ignore[attr-defined]
        tensor_cls.to = _pinned_async_to  # type: ignore[method-assign]

        if hasattr(tensor_cls, "cuda"):
            orig_cuda = tensor_cls.cuda

            @functools.wraps(orig_cuda)
            def _pinned_async_cuda(t_self: Any, *args: Any, **kwargs: Any) -> Any:
                if getattr(t_self, "is_cuda", False) or _dynamo_compiling():
                    return orig_cuda(t_self, *args, **kwargs)
                if _should_promote_non_blocking(t_self, args, kwargs, True):
                    kwargs["non_blocking"] = True
                return orig_cuda(t_self, *args, **kwargs)

            tensor_cls._ptopt_orig_cuda = orig_cuda  # type: ignore[attr-defined]
            tensor_cls.cuda = _pinned_async_cuda  # type: ignore[method-assign]

    _DATALOADER_HOOK_INSTALLED = True
    return True


def uninstall_dataloader_hook(torch_mod: Any) -> None:
    """Restore original DataLoader.__init__/__iter__ and Tensor async transfer hooks (for unit tests)."""
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
    tensor_cls = getattr(torch_mod, "Tensor", None)
    if tensor_cls is not None:
        if hasattr(tensor_cls, "_ptopt_orig_to"):
            tensor_cls.to = tensor_cls._ptopt_orig_to  # type: ignore[method-assign]
            delattr(tensor_cls, "_ptopt_orig_to")
        if hasattr(tensor_cls, "_ptopt_orig_cuda"):
            tensor_cls.cuda = tensor_cls._ptopt_orig_cuda  # type: ignore[method-assign]
            delattr(tensor_cls, "_ptopt_orig_cuda")
    for attr in ("_ptopt_dataloader_has_remainder", "_ptopt_dataloader_remainder_info"):
        if hasattr(torch_mod, attr):
            delattr(torch_mod, attr)
    _DATALOADER_HOOK_INSTALLED = False


def apply_dataloader_tune_pass(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any = None,
) -> str:
    """Enable DataLoader auto-tuning / CUDA stream prefetch pass when enabled."""
    if not cfg.dataloader.auto_tune and not getattr(cfg.dataloader, "cuda_prefetch", False):
        return "SKIPPED: dataloader.auto_tune is disabled"
    tmod = torch_mod if torch_mod is not None else sys.modules.get("torch")
    if tmod is None:
        return "SKIPPED: torch not loaded"
    installed = install_dataloader_hook(cfg, manager, tmod)
    return "DataLoader auto-tune hook installed" if installed else "DataLoader hook already active"
