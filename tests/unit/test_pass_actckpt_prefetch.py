"""Unit tests for Phase 5.5: Auto Activation Checkpointing & Dedicated CUDA Copy Stream Prefetch."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import OptimizeConfig
from src.hooks import HookManager
from src.passes.activation_checkpoint import (
    discover_checkpoint_blocks,
    install_activation_checkpoint_hook,
    uninstall_activation_checkpoint_hook,
)
from src.passes.dataloader_tune import _AsyncCudaPrefetchIter, install_dataloader_hook, uninstall_dataloader_hook


class _CustomNonTensorTag:
    def __init__(self, label: str) -> None:
        self.label = label


class _DummyCudaTensor:
    """Minimal tensor stub tracking `.to(non_blocking=True)` and `.record_stream(...)` calls."""

    def __init__(self, val: int, *, device_type: str = "cpu") -> None:
        self.val = val
        self.device = SimpleNamespace(type=device_type)
        self.is_cuda = device_type == "cuda"
        self.recorded_streams: list[object] = []
        self.moved_non_blocking = False

    def to(self, device: object = None, non_blocking: bool = False) -> "_DummyCudaTensor":
        del device
        moved = _DummyCudaTensor(self.val, device_type="cuda")
        moved.moved_non_blocking = bool(non_blocking)
        return moved

    def record_stream(self, stream: object) -> None:
        self.recorded_streams.append(stream)


class _DummyStream:
    def __init__(self, name: str) -> None:
        self.name = name
        self.waited_on: list[object] = []

    def wait_stream(self, other: object) -> None:
        self.waited_on.append(other)


class _DummyStreamCtx:
    def __init__(self, cuda_ns: "_DummyCudaNamespace", stream: _DummyStream) -> None:
        self._cuda = cuda_ns
        self._stream = stream
        self._prev: _DummyStream | None = None

    def __enter__(self) -> _DummyStream:
        self._prev = self._cuda._active_stream
        self._cuda._active_stream = self._stream
        return self._stream

    def __exit__(self, *_exc: object) -> None:
        self._cuda._active_stream = self._prev


class _DummyCudaNamespace:
    def __init__(self) -> None:
        self._main = _DummyStream("main")
        self._active_stream: _DummyStream | None = self._main
        self.created_streams: list[_DummyStream] = []

    def current_device(self) -> int:
        return 0

    def Stream(self, device: object = None) -> _DummyStream:
        del device
        s = _DummyStream(f"copy_{len(self.created_streams)}")
        self.created_streams.append(s)
        return s

    def stream(self, s: _DummyStream) -> _DummyStreamCtx:
        return _DummyStreamCtx(self, s)

    def current_stream(self, device: object = None) -> _DummyStream:
        del device
        return self._main


class _MiniBlock(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class _StackedModel(nn.Module):
    def __init__(self, dim: int = 128, n_blocks: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_MiniBlock(dim) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


class TestActivationCheckpointAndPrefetch(unittest.TestCase):
    """Tests for Phase 5.5 Auto Activation Checkpointing & Dedicated CUDA Copy Stream Prefetcher."""

    def test_activation_checkpoint_always_gradient_parity_and_eval_skip(self) -> None:
        torch.manual_seed(42)
        ref_model = _StackedModel(dim=128, n_blocks=3).train()
        ckpt_model = _StackedModel(dim=128, n_blocks=3).train()
        ckpt_model.load_state_dict(ref_model.state_dict())

        # Discover composite blocks (>= 60k params each for test speed)
        found = discover_checkpoint_blocks(ckpt_model, torch, min_block_params=60_000)
        self.assertEqual(len(found), 3)

        cfg = OptimizeConfig(profile="balanced")
        cfg.runtime.activation_checkpointing = "always"
        mgr = HookManager()

        x_ref = torch.randn(4, 128, requires_grad=True)
        x_ckpt = x_ref.detach().clone().requires_grad_(True)
        loss_ref = ref_model(x_ref).pow(2).mean()
        loss_ref.backward()

        orig_ckpt = torch.utils.checkpoint.checkpoint
        ckpt_calls = 0

        def _spy_ckpt(fn: object, *args: object, **kwargs: object) -> object:
            nonlocal ckpt_calls
            ckpt_calls += 1
            self.assertFalse(kwargs.get("use_reentrant", True))
            return orig_ckpt(fn, *args, **kwargs)  # type: ignore[arg-type]

        with patch.dict(os.environ, {"PTOPT_ACT_CKPT_MIN_PARAMS": "60000"}), patch.object(
            torch.utils.checkpoint, "checkpoint", side_effect=_spy_ckpt
        ):
            rt = install_activation_checkpoint_hook(cfg, mgr, torch)
            assert rt is not None
            try:
                loss_ckpt = ckpt_model(x_ckpt).pow(2).mean()
                loss_ckpt.backward()

                self.assertEqual(ckpt_calls, 3)
                torch.testing.assert_close(loss_ref, loss_ckpt)
                assert x_ref.grad is not None and x_ckpt.grad is not None
                torch.testing.assert_close(x_ref.grad, x_ckpt.grad)
                for p_ref, p_ck in zip(ref_model.parameters(), ckpt_model.parameters()):
                    assert p_ref.grad is not None and p_ck.grad is not None
                    torch.testing.assert_close(p_ref.grad, p_ck.grad)

                # eval() must bypass activation checkpointing completely
                calls_before_eval = ckpt_calls
                ckpt_model.eval()
                with torch.no_grad():
                    _ = ckpt_model(torch.randn(2, 128))
                self.assertEqual(ckpt_calls, calls_before_eval)
            finally:
                uninstall_activation_checkpoint_hook(torch)
                mgr.reset()

    def test_activation_checkpoint_auto_triggers_above_82_pct_vram_and_handles_fallback(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.runtime.activation_checkpointing = "auto"
        mgr = HookManager()
        model = _StackedModel(dim=128, n_blocks=2).train()
        ratios = iter([0.45, 0.86, 0.90])

        with patch.dict(os.environ, {"PTOPT_ACT_CKPT_MIN_PARAMS": "60000"}):
            rt = install_activation_checkpoint_hook(cfg, mgr, torch, vram_probe_fn=lambda: next(ratios, 0.90))
            assert rt is not None
            try:
                # Step 0: 45% VRAM < 82% -> stays inactive
                model(torch.randn(2, 128)).sum().backward()
                self.assertFalse(getattr(model, "_ptopt_act_ckpt_active", False))
                self.assertEqual(rt.triggered_modules, 0)

                # Step 1: 86% VRAM >= 82% -> triggers checkpointing cleanly & marks compile ineligible for #166926 safety
                model(torch.randn(2, 128)).sum().backward()
                self.assertTrue(getattr(model, "_ptopt_act_ckpt_active", False))
                self.assertTrue(getattr(model, "_ptopt_compile_ineligible", False))
                self.assertEqual(rt.triggered_modules, 1)
                self.assertEqual(rt.patched_blocks, 2)
                self.assertTrue(
                    any(ev.pass_name == "runtime.activation_checkpointing" and ev.status == "APPLIED" for ev in mgr.events)
                )

                # Verify descendant submodules are marked _ptopt_compile_checked so backward recomputation
                # never triggers auto_compile mid-backward.
                for sub in model.modules():
                    self.assertTrue(getattr(sub, "_ptopt_compile_checked", False))
                    self.assertTrue(getattr(sub, "_ptopt_compile_ineligible", False))

                # Simulate checkpoint failure -> block falls back cleanly to original forward
                with patch.object(torch.utils.checkpoint, "checkpoint", side_effect=RuntimeError("unsupported return")):
                    out = model(torch.randn(2, 128))
                    self.assertEqual(out.shape, (2, 8))
                self.assertTrue(
                    any(ev.pass_name == "runtime.activation_checkpointing" and ev.status == "FALLBACK" for ev in mgr.events)
                )
            finally:
                uninstall_activation_checkpoint_hook(torch)
                mgr.reset()

    def test_activation_checkpoint_lazy_torch_import_invariant(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import src.passes.activation_checkpoint; assert 'torch' not in sys.modules",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_async_cuda_prefetch_iter_stages_ahead_and_preserves_custom_items(self) -> None:
        cuda_ns = _DummyCudaNamespace()
        fake_torch = SimpleNamespace(
            Tensor=_DummyCudaTensor,
            device=lambda kind, idx=0: f"{kind}:{idx}",
            cuda=cuda_ns,
        )
        tag0 = _CustomNonTensorTag("meta0")
        tag1 = _CustomNonTensorTag("meta1")
        raw_batches = [
            {"img": _DummyCudaTensor(1), "lbl": _DummyCudaTensor(2), "tag": tag0},
            (_DummyCudaTensor(3), tag1),
            [_DummyCudaTensor(4, device_type="cuda"), "keep_str"],
        ]

        it = _AsyncCudaPrefetchIter(iter(raw_batches), fake_torch)
        # Batch 0 was already pre-staged on copy stream at __init__!
        self.assertEqual(len(cuda_ns.created_streams), 1)
        copy_stream = cuda_ns.created_streams[0]

        out0 = next(it)
        self.assertIsInstance(out0, dict)
        self.assertTrue(out0["img"].is_cuda)
        self.assertTrue(out0["img"].moved_non_blocking)
        self.assertIs(out0["tag"], tag0)
        self.assertEqual(out0["img"].recorded_streams, [cuda_ns._main])
        self.assertEqual(cuda_ns._main.waited_on, [copy_stream])

        out1 = next(it)
        self.assertIsInstance(out1, tuple)
        self.assertTrue(out1[0].is_cuda)
        self.assertIs(out1[1], tag1)

        out2 = next(it)
        self.assertEqual(out2[1], "keep_str")
        self.assertTrue(out2[0].is_cuda)

        with self.assertRaises(StopIteration):
            next(it)

        # Verify DataLoader hook applies cuda_prefetch on CUDA and skips cleanly when CUDA is unavailable
        cfg = OptimizeConfig(profile="balanced")
        cfg.dataloader.auto_tune = True
        cfg.dataloader.cuda_prefetch = True
        ds = torch.utils.data.TensorDataset(torch.randn(8, 4))

        if torch.cuda.is_available():
            mgr_gpu = HookManager()
            install_dataloader_hook(cfg, mgr_gpu, torch)
            try:
                dl_gpu = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=0)
                batches_gpu = list(dl_gpu)
                self.assertEqual(len(batches_gpu), 2)
                self.assertTrue(batches_gpu[0][0].is_cuda)
                self.assertTrue(
                    any(ev.pass_name == "dataloader.cuda_prefetch" and ev.status == "APPLIED" for ev in mgr_gpu.events)
                )
            finally:
                uninstall_dataloader_hook(torch)
                mgr_gpu.reset()

        mgr_cpu = HookManager()
        with patch.object(torch.cuda, "is_available", return_value=False):
            install_dataloader_hook(cfg, mgr_cpu, torch)
            try:
                dl_cpu = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=0)
                batches_cpu = list(dl_cpu)
                self.assertEqual(len(batches_cpu), 2)
                self.assertFalse(batches_cpu[0][0].is_cuda)
                self.assertTrue(
                    any(ev.pass_name == "dataloader.cuda_prefetch" and ev.status == "SKIPPED" for ev in mgr_cpu.events)
                )
            finally:
                uninstall_dataloader_hook(torch)
                mgr_cpu.reset()

    def test_recompute_dispatch_guard_blocks_amp_reentry_during_backward(self) -> None:
        from src.hooks.module_dispatch import register_module_call_handler, unregister_module_call_handler

        cfg = OptimizeConfig(profile="balanced")
        cfg.runtime.activation_checkpointing = "always"
        mgr = HookManager()
        model = _StackedModel(dim=256, n_blocks=2).train()

        handler_calls: list[str] = []

        def _spy_amp_handler(mod_self: object, args: tuple[object, ...], kwargs: dict[str, object], next_fn: object) -> object:
            # Simulates amp.auto detecting whether an inner call enters top-level dispatch
            handler_calls.append(type(mod_self).__name__)
            return next_fn(mod_self, *args, **kwargs)  # type: ignore[operator]

        register_module_call_handler(torch, "amp.spy", 50, _spy_amp_handler)
        install_activation_checkpoint_hook(cfg, mgr, torch)
        try:
            x = torch.randn(2, 256, requires_grad=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = model(x).sum()
            # During forward, top-level _StackedModel triggers _spy_amp_handler once (depth 0 -> 1)
            self.assertEqual(handler_calls, ["_StackedModel"])
            loss.backward()
            # During backward recomputation inside checkpoint, _recompute_depth > 0 blocks re-entry!
            self.assertEqual(handler_calls, ["_StackedModel"])
        finally:
            uninstall_activation_checkpoint_hook(torch)
            unregister_module_call_handler(torch, "amp.spy")
            mgr.reset()

    def test_partial_strided_checkpointing_and_escalation(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.runtime.activation_checkpointing = "auto"
        mgr = HookManager()
        # dim=128 (~33k params per block) tests both len>=4 10k discovery & strided ratio=0.5
        model = _StackedModel(dim=128, n_blocks=4).train()

        probes = iter([0.90, 0.90, 0.40])
        rt = install_activation_checkpoint_hook(cfg, mgr, torch, vram_probe_fn=lambda: next(probes, 0.40))
        assert rt is not None
        try:
            x = torch.randn(2, 128, requires_grad=True)
            model(x).sum().backward()
            # Step 1 (ratio=0.5 on 4 blocks -> stride 2 -> 2 blocks patched)
            self.assertEqual(rt.patched_blocks, 2)
            self.assertTrue(getattr(model.blocks[0], "_ptopt_act_ckpt_patched", False))
            self.assertFalse(getattr(model.blocks[1], "_ptopt_act_ckpt_patched", False))
            self.assertTrue(getattr(model.blocks[2], "_ptopt_act_ckpt_patched", False))
            self.assertFalse(getattr(model.blocks[3], "_ptopt_act_ckpt_patched", False))

            # Step 2 (VRAM still >= 0.82 -> escalates to ratio=1.0 -> patches remaining 2 blocks)
            model(x).sum().backward()
            self.assertEqual(rt.patched_blocks, 4)
            self.assertTrue(all(getattr(b, "_ptopt_act_ckpt_patched", False) for b in model.blocks))
        finally:
            uninstall_activation_checkpoint_hook(torch)
            mgr.reset()

    def test_act_ckpt_enables_auto_triton_fallback_on_bypass(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.runtime.activation_checkpointing = "always"
        mgr = HookManager()
        model = _StackedModel(dim=256, n_blocks=2).train()

        install_activation_checkpoint_hook(cfg, mgr, torch)
        try:
            with patch("src.kernels.registry.enable_custom_kernels_on_module", return_value=2) as mocked_triton:
                x = torch.randn(2, 256, requires_grad=True)
                model(x).sum().backward()
                mocked_triton.assert_called_once()
        finally:
            uninstall_activation_checkpoint_hook(torch)
            mgr.reset()

    def test_step0_oom_catch_and_retry_guardrail2(self) -> None:
        import os

        # Case 1: mode="auto", non-strict -> catches Step-0 OOM, enables ratio=1.0, retries cleanly
        cfg = OptimizeConfig(profile="balanced")
        cfg.runtime.activation_checkpointing = "auto"
        mgr = HookManager()
        mgr.strict = False
        model = _StackedModel(dim=256, n_blocks=4).train()

        orig_fwd = model.forward
        call_count = 0

        def _oom_once(x: torch.Tensor) -> torch.Tensor:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")
            return orig_fwd(x)

        model.forward = _oom_once  # type: ignore[method-assign]
        # VRAM probe says 0.40 (below 0.82 threshold), so only OOM catch triggers checkpointing
        rt = install_activation_checkpoint_hook(cfg, mgr, torch, vram_probe_fn=lambda: 0.40)
        assert rt is not None
        try:
            x = torch.randn(2, 256, requires_grad=True)
            out = model(x)
            out.sum().backward()
            self.assertEqual(call_count, 2)
            self.assertEqual(rt.patched_blocks, 4)
            self.assertTrue(
                any(
                    ev.pass_name == "runtime.activation_checkpointing"
                    and ev.status == "FALLBACK"
                    and "OutOfMemoryError" in ev.reason
                    for ev in mgr.events
                )
            )
        finally:
            uninstall_activation_checkpoint_hook(torch)
            mgr.reset()

        # Case 2: strict=True -> Guardrail #2 disables OOM retry
        mgr_strict = HookManager()
        mgr_strict.strict = True
        model_strict = _StackedModel(dim=256, n_blocks=4).train()
        orig_strict_fwd = model_strict.forward

        def _always_oom(x: torch.Tensor) -> torch.Tensor:
            if not getattr(model_strict, "_ptopt_act_ckpt_active", False):
                raise RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")
            return orig_strict_fwd(x)

        model_strict.forward = _always_oom  # type: ignore[method-assign]
        with patch.dict(os.environ, {}, clear=False):
            install_activation_checkpoint_hook(cfg, mgr_strict, torch, vram_probe_fn=lambda: 0.40)
            try:
                x_s = torch.randn(2, 256, requires_grad=True)
                with self.assertRaises(RuntimeError):
                    model_strict(x_s)
            finally:
                uninstall_activation_checkpoint_hook(torch)
                mgr_strict.reset()

    def test_checkpoint_early_stop_control_flow_not_swallowed_by_kernel_wrappers(self) -> None:
        from src.kernels.registry import _is_checkpoint_control_flow

        stop_cls = getattr(torch.utils.checkpoint, "_StopRecomputationError", None)
        if stop_cls is not None:
            self.assertTrue(_is_checkpoint_control_flow(stop_cls()))
        self.assertFalse(_is_checkpoint_control_flow(RuntimeError("boom")))

        class _SavedTensorFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx: object, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)  # type: ignore[attr-defined]
                return x * 2.0

            @staticmethod
            def backward(ctx: object, grad_out: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors  # type: ignore[attr-defined]
                return grad_out * (2.0 + x * 0.0)

        lin = nn.Linear(16, 16)

        def _block_fwd(inp: torch.Tensor) -> torch.Tensor:
            h = torch.relu(lin(inp))
            try:
                # Last op inside checkpointed region triggers _StopRecomputationError during recompute
                return _SavedTensorFn.apply(h)  # type: ignore[return-value]
            except Exception as exc:  # noqa: BLE001
                if _is_checkpoint_control_flow(exc):
                    raise
                return h * 2.0

        x = torch.randn(4, 16, requires_grad=True)
        out = torch.utils.checkpoint.checkpoint(_block_fwd, x, use_reentrant=False)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(lin.weight.grad)


if __name__ == "__main__":
    unittest.main()
