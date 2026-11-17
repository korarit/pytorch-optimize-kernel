"""Central pass registration wiring Phase 3 passes into HookManager."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.passes.channels_last import apply_channels_last_pass
from src.passes.memory_allocator import apply_memory_allocator_pass
from src.passes.runtime_flags import apply_runtime_flags_pass

if TYPE_CHECKING:
    from src.hooks.manager import HookManager


def register_builtin_passes(manager: "HookManager") -> None:
    """Register all active optimization passes in execution order."""
    if os.environ.get("PTOPT_DISABLE_BUILTIN_PASSES") == "1":
        return
    manager.register_pass("runtime.memory_allocator", apply_memory_allocator_pass)
    manager.register_pass("runtime.flags", apply_runtime_flags_pass)
    manager.register_pass("runtime.channels_last", apply_channels_last_pass)
