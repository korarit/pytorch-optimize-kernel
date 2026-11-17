"""Central pass registration wiring Phase 3 passes into HookManager."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.passes.auto_amp import apply_auto_amp_pass
from src.passes.auto_compile import apply_auto_compile_pass
from src.passes.channels_last import apply_channels_last_pass
from src.passes.dataloader_tune import apply_dataloader_tune_pass
from src.passes.inference_opt import apply_inference_opt_pass
from src.passes.memory_allocator import apply_memory_allocator_pass
from src.passes.optimizer_fusion import apply_optimizer_fusion_pass
from src.passes.runtime_flags import apply_runtime_flags_pass

if TYPE_CHECKING:
    from src.hooks.manager import HookManager


def register_builtin_passes(manager: "HookManager") -> None:
    """Register all active optimization passes in execution order."""
    if os.environ.get("PTOPT_DISABLE_BUILTIN_PASSES") == "1":
        return
    manager.register_pass("runtime.memory_allocator", apply_memory_allocator_pass)
    manager.register_pass("runtime.flags", apply_runtime_flags_pass)
    manager.register_pass("dataloader.auto_tune", apply_dataloader_tune_pass)
    manager.register_pass("optimizer.fusion", apply_optimizer_fusion_pass)
    # Compile wraps inner _wrapped_call_impl inside channels_last + auto_amp + inference_opt
    manager.register_pass("compile.auto", apply_auto_compile_pass)
    manager.register_pass("runtime.channels_last", apply_channels_last_pass)
    manager.register_pass("amp.auto", apply_auto_amp_pass)
    manager.register_pass("inference.opt", apply_inference_opt_pass)
