"""Unified `nn.Module.__call__` dispatcher and cyclic-safe nested tensor inspection helpers."""

from __future__ import annotations

import functools
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator

ModuleCallHandler = Callable[[Any, tuple[Any, ...], dict[str, Any], Callable[..., Any]], Any]


def iter_nested_tensors(
    obj: Any,
    tensor_cls: Any,
    *,
    max_depth: int = 6,
    max_items: int = 64,
) -> Iterator[Any]:
    """Yield tensors inside nested tuple/list/dict inputs safely (cyclic & depth bounded)."""
    if tensor_cls is None or obj is None:
        return
    if isinstance(obj, tensor_cls):
        yield obj
        return

    seen_ids: set[int] = set()
    yielded = 0

    def _walk(cur: Any, depth: int) -> Iterator[Any]:
        nonlocal yielded
        if cur is None or yielded >= max_items:
            return
        if isinstance(cur, tensor_cls):
            yielded += 1
            yield cur
            return
        if depth >= max_depth:
            return
        if isinstance(cur, (tuple, list)):
            cid = id(cur)
            if cid in seen_ids:
                return
            seen_ids.add(cid)
            for item in cur:
                yield from _walk(item, depth + 1)
                if yielded >= max_items:
                    return
        elif isinstance(cur, dict):
            cid = id(cur)
            if cid in seen_ids:
                return
            seen_ids.add(cid)
            for val in cur.values():
                yield from _walk(val, depth + 1)
                if yielded >= max_items:
                    return

    yield from _walk(obj, 0)


def has_cuda_context(mod_self: Any, args: tuple[Any, ...], kwargs: dict[str, Any], torch_mod: Any) -> bool:
    """Return True if any module parameter or (possibly nested) input tensor resides on CUDA."""
    tensor_cls = getattr(torch_mod, "Tensor", None)
    if tensor_cls is not None:
        for item in (*args, *kwargs.values()):
            for tensor in iter_nested_tensors(item, tensor_cls):
                if getattr(tensor, "is_cuda", False):
                    return True
    if hasattr(mod_self, "parameters"):
        try:
            first_param = next(mod_self.parameters(), None)
            if first_param is not None and getattr(first_param, "is_cuda", False):
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def has_requires_grad_tensor(args: tuple[Any, ...], kwargs: dict[str, Any], torch_mod: Any) -> bool:
    """Return True if any input tensor (including nested dicts/lists/tuples) requires gradients."""
    tensor_cls = getattr(torch_mod, "Tensor", None)
    if tensor_cls is None:
        return False
    for item in (*args, *kwargs.values()):
        for tensor in iter_nested_tensors(item, tensor_cls):
            if getattr(tensor, "requires_grad", False):
                return True
    return False


def build_shape_signature(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    torch_mod: Any,
    *,
    include_device: bool = True,
) -> tuple[Any, ...]:
    """Build hashable shape+dtype(+device) signature without per-step string formatting."""
    tensor_cls = getattr(torch_mod, "Tensor", None)
    if tensor_cls is None:
        return ()
    sig: list[Any] = []
    for item in (*args, *kwargs.values()):
        for tensor in iter_nested_tensors(item, tensor_cls):
            shape = tuple(tensor.shape)
            dtype = getattr(tensor, "dtype", None)
            if include_device:
                sig.append((shape, dtype, getattr(tensor, "device", None)))
            else:
                sig.append((shape, dtype))
    return tuple(sig)


class ModuleCallDispatcher:
    """Single top-level `nn.Module.__call__` hook multiplexing all optimization passes."""

    def __init__(self, torch_mod: Any) -> None:
        self.torch = torch_mod
        self._nn_mod = torch_mod.nn.Module
        self._orig_call = getattr(self._nn_mod, "__call__")
        self._handlers: dict[str, tuple[int, ModuleCallHandler]] = {}
        self._chain: Callable[..., Any] = self._orig_call
        self._tls = threading.local()
        self._unified_fn: Callable[..., Any] | None = None
        self._installed = False

    @property
    def _depth(self) -> int:
        return int(getattr(self._tls, "depth", 0))

    @_depth.setter
    def _depth(self, val: int) -> None:
        self._tls.depth = max(0, int(val))

    @property
    def _recompute_depth(self) -> int:
        return int(getattr(self._tls, "recompute_depth", 0))

    @_recompute_depth.setter
    def _recompute_depth(self, val: int) -> None:
        self._tls.recompute_depth = max(0, int(val))

    def _rebuild_chain(self) -> None:
        ordered = sorted(self._handlers.items(), key=lambda kv: kv[1][0])
        nxt: Callable[..., Any] = self._orig_call
        for _name, (_order, handler) in reversed(ordered):
            down = nxt

            def _step(
                mod_self: Any,
                *args: Any,
                _h: ModuleCallHandler = handler,
                _down: Callable[..., Any] = down,
                **kwargs: Any,
            ) -> Any:
                return _h(mod_self, args, kwargs, _down)

            nxt = _step
        self._chain = nxt

    def _ensure_installed(self) -> None:
        if self._installed:
            return
        dispatcher = self
        orig_call = self._orig_call
        is_compiling_fn = getattr(getattr(self.torch, "compiler", None), "is_compiling", None)

        @functools.wraps(orig_call)
        def _unified_mod_call(mod_self: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                (callable(is_compiling_fn) and is_compiling_fn())
                or dispatcher._depth > 0
                or dispatcher._recompute_depth > 0
            ):
                return orig_call(mod_self, *args, **kwargs)
            dispatcher._depth += 1
            try:
                return dispatcher._chain(mod_self, *args, **kwargs)
            finally:
                dispatcher._depth -= 1

        self._unified_fn = _unified_mod_call
        self._nn_mod._ptopt_dispatcher_orig_call = orig_call  # type: ignore[attr-defined]
        setattr(self._nn_mod, "__call__", _unified_mod_call)
        self._installed = True

    def register(self, name: str, order: int, handler: ModuleCallHandler) -> None:
        self._handlers[name] = (order, handler)
        self._rebuild_chain()
        self._ensure_installed()

    def unregister(self, name: str) -> None:
        if name not in self._handlers:
            return
        self._handlers.pop(name, None)
        if self._handlers:
            self._rebuild_chain()
            return
        self._chain = self._orig_call
        if (
            self._installed
            and hasattr(self._nn_mod, "_ptopt_dispatcher_orig_call")
            and getattr(self._nn_mod, "__call__", None) is self._unified_fn
        ):
            setattr(self._nn_mod, "__call__", self._orig_call)
            delattr(self._nn_mod, "_ptopt_dispatcher_orig_call")
        self._installed = False
        if getattr(self.torch, "_ptopt_module_dispatcher", None) is self:
            delattr(self.torch, "_ptopt_module_dispatcher")


def get_module_call_dispatcher(torch_mod: Any, *, create: bool = True) -> ModuleCallDispatcher | None:
    """Return the `ModuleCallDispatcher` bound to `torch_mod`, creating it if requested."""
    if torch_mod is None or not hasattr(torch_mod, "nn"):
        return None
    disp = getattr(torch_mod, "_ptopt_module_dispatcher", None)
    if disp is None and create:
        disp = ModuleCallDispatcher(torch_mod)
        torch_mod._ptopt_module_dispatcher = disp
    return disp


def register_module_call_handler(
    torch_mod: Any,
    name: str,
    order: int,
    handler: ModuleCallHandler,
) -> bool:
    """Register a top-level `nn.Module.__call__` handler on the unified dispatcher."""
    disp = get_module_call_dispatcher(torch_mod, create=True)
    if disp is None:
        return False
    disp.register(name, order, handler)
    return True


def unregister_module_call_handler(torch_mod: Any, name: str) -> None:
    """Remove a handler from the unified dispatcher and restore `Module.__call__` if empty."""
    disp = get_module_call_dispatcher(torch_mod, create=False)
    if disp is not None:
        disp.unregister(name)


@contextmanager
def recompute_dispatch_guard(torch_mod: Any) -> Iterator[None]:
    """Suppress top-level `nn.Module.__call__` pass wrappers during backward recomputation."""
    disp = get_module_call_dispatcher(torch_mod, create=False)
    if disp is None:
        yield
        return
    disp._recompute_depth += 1
    try:
        yield
    finally:
        disp._recompute_depth = max(0, disp._recompute_depth - 1)


def resolve_dataparallel_target(mod_self: Any, torch_mod: Any) -> tuple[Any, bool, str | None]:
    """Unwrap single-GPU `nn.DataParallel` so compilation targets `.module` without `_get_device_index` breaks."""
    dp_cls = getattr(getattr(torch_mod, "nn", None), "DataParallel", None)
    if dp_cls is None or not isinstance(mod_self, dp_cls):
        return mod_self, False, None
    inner = getattr(mod_self, "module", None)
    dev_ids = getattr(mod_self, "device_ids", None)
    if inner is None:
        return None, True, "DataParallel missing inner .module; skipping compile"
    if isinstance(dev_ids, (list, tuple)) and len(dev_ids) > 1:
        return None, True, f"Multi-GPU DataParallel (devices={len(dev_ids)}) skipped for compile safety"
    return inner, True, f"Unwrapped single-GPU DataParallel -> {type(inner).__name__}"
