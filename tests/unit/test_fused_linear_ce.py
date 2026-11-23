"""Unit parity tests for Phase 5.2: Fused Chunked Linear + CrossEntropy & LazyLinearLogits."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptopt.config import OptimizeConfig
from ptopt.hooks import HookManager
from ptopt.kernels import (
    LazyLinearLogits,
    install_custom_kernels_hooks,
    ref_linear_cross_entropy,
    triton_fused_linear_cross_entropy as fused_linear_cross_entropy,
    uninstall_custom_kernels_hooks,
)
from ptopt.kernels.fused_linear_cross_entropy import compute_optimal_chunk_size, create_lazy_linear_logits


class TestFusedLinearCrossEntropyParity(unittest.TestCase):
    """Exact forward & backward gradient parity across reductions, ignore_index, and chunk splits."""

    def test_optimal_chunk_size_caps_memory_budget(self) -> None:
        self.assertEqual(compute_optimal_chunk_size(4096, 32768, torch.bfloat16), 1024)
        self.assertEqual(compute_optimal_chunk_size(4096, 65536, torch.bfloat16), 512)
        self.assertEqual(compute_optimal_chunk_size(90, 32768, torch.float32), 90)

    def test_forward_backward_parity_mean_sum_none_with_ignore_index(self) -> None:
        torch.manual_seed(7)
        for device in (["cuda"] if torch.cuda.is_available() else ["cpu"]):
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            atol_loss = 1e-2 if dtype == torch.bfloat16 else 1e-5
            atol_grad = 5e-2 if dtype == torch.bfloat16 else 1e-5

            for reduction in ("mean", "sum", "none"):
                with self.subTest(device=device, dtype=dtype, reduction=reduction):
                    h_ref = torch.randn(37, 24, device=device, dtype=dtype, requires_grad=True)
                    w_ref = torch.randn(193, 24, device=device, dtype=dtype, requires_grad=True)
                    b_ref = torch.randn(193, device=device, dtype=dtype, requires_grad=True)
                    tgt = torch.randint(0, 193, (37,), device=device)
                    # Whole first chunk ignored + mixed tail
                    tgt[:16] = -100
                    tgt[20] = -100

                    h_fused = h_ref.detach().clone().requires_grad_(True)
                    w_fused = w_ref.detach().clone().requires_grad_(True)
                    b_fused = b_ref.detach().clone().requires_grad_(True)

                    loss_ref = ref_linear_cross_entropy(h_ref, w_ref, b_ref, tgt, ignore_index=-100, reduction=reduction)
                    loss_fused = fused_linear_cross_entropy(
                        h_fused,
                        w_fused,
                        tgt,
                        bias=b_fused,
                        ignore_index=-100,
                        reduction=reduction,
                        chunk_size=16,  # force 3 chunks: 16, 16, 5 (with first chunk all ignored)
                    )

                    self.assertTrue(torch.allclose(loss_ref.float(), loss_fused.float(), rtol=1e-2, atol=atol_loss))
                    if reduction == "none":
                        g = torch.randn_like(loss_ref)
                        loss_ref.backward(g)
                        loss_fused.backward(g)
                    else:
                        loss_ref.backward()
                        loss_fused.backward()

                    assert h_ref.grad is not None and h_fused.grad is not None
                    assert w_ref.grad is not None and w_fused.grad is not None
                    assert b_ref.grad is not None and b_fused.grad is not None
                    self.assertTrue(torch.allclose(h_ref.grad.float(), h_fused.grad.float(), rtol=1e-2, atol=atol_grad))
                    self.assertTrue(torch.allclose(w_ref.grad.float(), w_fused.grad.float(), rtol=1e-2, atol=atol_grad))
                    self.assertTrue(torch.allclose(b_ref.grad.float(), b_fused.grad.float(), rtol=1e-2, atol=atol_grad))

    def test_bfloat16_multi_chunk_addmm_parity(self) -> None:
        torch.manual_seed(11)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        h_ref = torch.randn(300, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        w_ref = torch.randn(128, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        b_ref = torch.randn(128, device=device, dtype=torch.bfloat16, requires_grad=True)
        tgt = torch.randint(0, 128, (300,), device=device)

        h_f = h_ref.detach().clone().requires_grad_(True)
        w_f = w_ref.detach().clone().requires_grad_(True)
        b_f = b_ref.detach().clone().requires_grad_(True)

        l_ref = ref_linear_cross_entropy(h_ref, w_ref, b_ref, tgt)
        l_ref.backward()
        l_f = fused_linear_cross_entropy(h_f, w_f, tgt, bias=b_f, chunk_size=100)
        l_f.backward()

        self.assertTrue(torch.allclose(l_ref.float(), l_f.float(), rtol=1e-2, atol=1e-2))
        assert w_ref.grad is not None and w_f.grad is not None
        assert b_ref.grad is not None and b_f.grad is not None
        self.assertTrue(torch.allclose(w_ref.grad.float(), w_f.grad.float(), rtol=1e-2, atol=5e-2))
        self.assertTrue(torch.allclose(b_ref.grad.float(), b_f.grad.float(), rtol=1e-2, atol=5e-2))

    def test_all_ignored_targets_safe_no_nan(self) -> None:
        h = torch.randn(12, 8, requires_grad=True)
        w = torch.randn(64, 8, requires_grad=True)
        tgt = torch.full((12,), -100, dtype=torch.long)
        loss = fused_linear_cross_entropy(h, w, tgt, chunk_size=5)
        self.assertEqual(float(loss.item()), 0.0)
        loss.backward()
        assert h.grad is not None and w.grad is not None
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertTrue(torch.isfinite(w.grad).all())

    def test_lazy_linear_logits_slice_view_and_materialize_fallback(self) -> None:
        torch.manual_seed(13)
        h = torch.randn(2, 5, 16, requires_grad=True)
        w = torch.randn(128, 16, requires_grad=True)
        b = torch.randn(128, requires_grad=True)

        lazy = create_lazy_linear_logits(h, w, b)
        self.assertIsInstance(lazy, LazyLinearLogits)
        self.assertIsInstance(lazy, torch.Tensor)
        self.assertEqual(lazy.shape, (2, 5, 128))
        self.assertEqual(lazy.device.type, "cpu")

        # Causal shift slice `[:, :-1, :].contiguous().view(-1, V)` preserves lazy status cleanly
        shifted = lazy[:, :-1, :].contiguous().view(-1, 128)
        self.assertIsInstance(shifted, LazyLinearLogits)
        self.assertEqual(shifted.shape, (8, 128))

        # Arbitrary op (argmax / topk / sum) materializes transparently
        preds = shifted.argmax(dim=-1)
        ref_preds = F.linear(h, w, b)[:, :-1, :].reshape(-1, 128).argmax(dim=-1)
        self.assertTrue(torch.equal(preds, ref_preds))

        # Backprop through materialized consumer works identically to F.linear
        loss_mat = (shifted * 0.5).sum()
        loss_mat.backward()
        assert h.grad is not None and w.grad is not None
        self.assertEqual(h.grad.shape, h.shape)
        self.assertEqual(w.grad.shape, w.shape)

    def test_registry_eager_mode_patches_large_head_and_fuses_with_cross_entropy(self) -> None:
        class _FakeHeadModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(16, 16)
                self.head = nn.Linear(16, 4096, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.head(torch.relu(self.proj(x)))

        cfg = OptimizeConfig(profile="aggressive")
        cfg.compile.enabled = False
        cfg.kernels.enabled = True
        cfg.kernels.allow_list = ["fused_linear_cross_entropy"]
        mgr = HookManager()

        reg = install_custom_kernels_hooks(cfg, mgr, torch, triton_available=True, sm_override=89)
        assert reg is not None
        self.assertEqual(reg.fallback_registry.entries["fused_linear_cross_entropy"].status, "APPLIED")

        model = _FakeHeadModel()
        # Mock CUDA checks cleanly on CPU so registry end-to-end hook dispatch runs in CPU CI too
        fake_w = model.head.weight
        try:
            with patch.dict(os.environ, {"PTOPT_FUSED_LCE_MIN_VOCAB": "4096"}):
                # Force head.weight.is_cuda check only via patched helper or run directly when CUDA available
                if torch.cuda.is_available():
                    model = model.cuda()
                    x = torch.randn(4, 8, 16, device="cuda")
                    y = torch.randint(0, 4096, (32,), device="cuda")
                    logits = model(x)
                    self.assertIsInstance(logits, LazyLinearLogits)
                    loss = F.cross_entropy(logits.view(-1, 4096), y)
                    loss.backward()
                    self.assertIsNotNone(model.head.weight.grad)
                else:
                    # Verify fused_linear_cross_entropy directly via LazyLinearLogits -> F.cross_entropy hook
                    h_in = torch.randn(4, 8, 16, requires_grad=True)
                    lazy = create_lazy_linear_logits(h_in, fake_w, None)
                    tgt = torch.randint(0, 4096, (32,))
                    loss_hooked = F.cross_entropy(lazy.view(-1, 4096), tgt)
                    loss_ref = ref_linear_cross_entropy(h_in.detach(), fake_w.detach(), None, tgt)
                    self.assertTrue(torch.allclose(loss_hooked.float(), loss_ref.float(), atol=1e-5))
                    loss_hooked.backward()
                    self.assertIsNotNone(fake_w.grad)
        finally:
            uninstall_custom_kernels_hooks(torch)
            mgr.reset()

    def test_lazy_linear_logits_dtype_cast_stays_lazy_and_matches_gradients(self) -> None:
        torch.manual_seed(7)
        cfg = OptimizeConfig(profile="balanced")
        cfg.compile.enabled = True
        mgr = HookManager()
        reg = install_custom_kernels_hooks(cfg, mgr, torch, triton_available=True, sm_override=89)
        assert reg is not None
        try:
            h = torch.randn(2, 4, 8, dtype=torch.bfloat16, requires_grad=True)
            w = torch.randn(32, 8, dtype=torch.bfloat16, requires_grad=True)
            b = torch.randn(32, dtype=torch.bfloat16, requires_grad=True)
            tgt = torch.randint(0, 32, (8,))

            lazy = create_lazy_linear_logits(h, w, b)
            casted = lazy.view(-1, 32).float()
            self.assertIsInstance(casted, LazyLinearLogits)
            self.assertEqual(casted.dtype, torch.float32)
            self.assertEqual(casted._ptopt_lazy_out_dtype, torch.float32)

            loss_fused = F.cross_entropy(casted, tgt)

            h_ref = h.detach().clone().requires_grad_(True)
            w_ref = w.detach().clone().requires_grad_(True)
            b_ref = b.detach().clone().requires_grad_(True)
            logits_ref = F.linear(h_ref, w_ref, b_ref).view(-1, 32).float()
            loss_ref = F.cross_entropy(logits_ref, tgt)

            self.assertTrue(torch.allclose(loss_fused.float(), loss_ref.float(), atol=1e-4, rtol=1e-4))
            loss_fused.backward()
            loss_ref.backward()
            self.assertIsNotNone(h.grad)
            self.assertIsNotNone(w.grad)
            self.assertTrue(torch.allclose(h.grad.float(), h_ref.grad.float(), atol=5e-3, rtol=5e-3))
            self.assertTrue(torch.allclose(w.grad.float(), w_ref.grad.float(), atol=5e-3, rtol=5e-3))
            self.assertTrue(torch.allclose(b.grad.float(), b_ref.grad.float(), atol=5e-3, rtol=5e-3))

            # Verify .to(dtype) and .half() also stay lazy without materializing
            lazy_half = lazy.to(torch.float16)
            self.assertIsInstance(lazy_half, LazyLinearLogits)
            self.assertEqual(lazy_half.dtype, torch.float16)
            self.assertEqual(lazy_half._ptopt_lazy_out_dtype, torch.float16)
        finally:
            uninstall_custom_kernels_hooks(torch)
            mgr.reset()

    def test_hybrid_compile_lce_balanced_patches_terminal_head_only(self) -> None:
        from types import SimpleNamespace
        from ptopt.kernels.registry import _patch_module_submodules

        cfg = OptimizeConfig(profile="balanced")
        cfg.compile.enabled = True
        mgr = HookManager()

        reg = install_custom_kernels_hooks(cfg, mgr, torch, triton_available=True, sm_override=89)
        assert reg is not None
        try:
            self.assertEqual(reg.fallback_registry.entries["fused_linear_cross_entropy"].status, "APPLIED")
            self.assertIn("hybrid", reg.fallback_registry.entries["fused_linear_cross_entropy"].reason.lower())

            # Verify _patch_module_submodules wraps terminal head with torch.compiler.disable in hybrid mode
            # while skipping standard 4x MLP up-projection below min_lce_vocab.
            class _FakeLinear:
                def __init__(self, in_f: int, out_f: int) -> None:
                    self.in_features = in_f
                    self.out_features = out_f
                    self.weight = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
                    self.bias = None

                def forward(self, x: object) -> object:
                    return x

            class _FakeModel:
                def __init__(self) -> None:
                    self.mlp_up = _FakeLinear(16, 128)  # 8x expansion, vocab=128 < 8192
                    self.head = _FakeLinear(128, 8192)  # terminal LM head >= 8192

                def modules(self) -> list[object]:
                    return [self, self.mlp_up, self.head]

            disable_calls: list[object] = []

            def _fake_compiler_disable(fn: object) -> object:
                disable_calls.append(fn)
                return fn

            fake_torch = SimpleNamespace(
                bfloat16=torch.bfloat16,
                float16=torch.float16,
                float32=torch.float32,
                nn=SimpleNamespace(Linear=_FakeLinear),
                compiler=SimpleNamespace(disable=_fake_compiler_disable),
            )
            fake_model = _FakeModel()
            patched = _patch_module_submodules(
                fake_model,
                fake_torch,
                {"fused_linear_cross_entropy": True},
                mgr,
                min_lce_vocab=8192,
                fallback_on_error=True,
                hybrid_compile_lce=True,
            )
            self.assertEqual(patched, 1)
            self.assertTrue(getattr(fake_model.head, "_ptopt_kern_patched", False))
            self.assertFalse(getattr(fake_model.mlp_up, "_ptopt_kern_patched", False))
            self.assertEqual(len(disable_calls), 1)
        finally:
            uninstall_custom_kernels_hooks(torch)
            mgr.reset()

    def test_chunk_ce_fwd_bwd_kernel_launch_signatures(self) -> None:
        from ptopt.kernels import fused_linear_cross_entropy as flce

        captured_fwd: list[tuple[tuple[object, ...], dict[str, object]]] = []
        captured_bwd: list[tuple[tuple[object, ...], dict[str, object]]] = []

        class _MockKernel:
            def __init__(self, sink: list[tuple[tuple[object, ...], dict[str, object]]]) -> None:
                self._sink = sink

            def __getitem__(self, _grid: object) -> "_MockKernel":
                return self

            def __call__(self, *args: object, **kwargs: object) -> None:
                self._sink.append((args, kwargs))

        class _CudaFakeTensor(torch.Tensor):
            @property
            def is_cuda(self) -> bool:  # type: ignore[override]
                return True

        logits_c = torch.randn(4, 16).as_subclass(_CudaFakeTensor)
        targets_c = torch.randint(0, 16, (4,)).as_subclass(_CudaFakeTensor)
        losses_out = torch.empty(4)
        lse_out = torch.empty(4)
        grad_in = torch.ones(())

        with patch.object(flce, "HAS_TRITON", True), patch.object(
            flce, "_cross_entropy_fwd_kernel", _MockKernel(captured_fwd)
        ), patch.object(flce, "_cross_entropy_bwd_kernel", _MockKernel(captured_bwd)):
            with patch.object(torch, "empty", return_value=losses_out):
                flce._chunk_ce_fwd(
                    logits_c, targets_c, losses_out, lse_out, ignore_index=-100, block_v=128, warps=4
                )
            flce._chunk_ce_bwd(
                logits_c, targets_c, lse_out, grad_in, is_scalar_grad=True, ignore_index=-100, block_v=128, warps=4
            )

        self.assertEqual(len(captured_fwd), 1)
        fwd_args, fwd_kw = captured_fwd[0]
        # ValidCount_ptr + stride_logits + stride_targets + V + ignore_index => 9 positional args
        self.assertEqual(len(fwd_args), 9)
        self.assertIn("COUNT_VALID", fwd_kw)
        self.assertFalse(fwd_kw["COUNT_VALID"])

        self.assertEqual(len(captured_bwd), 1)
        bwd_args, bwd_kw = captured_bwd[0]
        # DX, Logits, Targets, LSE, GradOut, stride_dx, stride_logits, stride_targets, V, ignore_index => 10 args
        self.assertEqual(len(bwd_args), 10)
        self.assertIn("IS_SCALAR_GRAD", bwd_kw)

    def test_adaptive_chunk_budget_scales_with_cuda_headroom_and_respects_tight_vram(self) -> None:
        from ptopt.kernels import fused_linear_cross_entropy as flce

        cuda_dev = torch.device("cuda", 0)
        with patch(
            "ptopt.passes.memory_allocator.probe_cuda_vram_headroom",
            return_value={"total_mb": 22500.0, "allocated_mb": 3300.0, "reserved_mb": 4000.0, "free_mb": 18500.0, "used_ratio": 0.1778},
        ):
            budget_hi, max_rows_hi = flce._select_adaptive_chunk_budget(cuda_dev)
            self.assertEqual(budget_hi, 448 * 1024 * 1024)
            self.assertEqual(max_rows_hi, 4096)
            self.assertEqual(
                compute_optimal_chunk_size(4096, 50304, torch.bfloat16, target_bytes=budget_hi, max_chunk=max_rows_hi),
                4096,
            )

        with patch(
            "ptopt.passes.memory_allocator.probe_cuda_vram_headroom",
            return_value={"total_mb": 8192.0, "allocated_mb": 6900.0, "reserved_mb": 7200.0, "free_mb": 992.0, "used_ratio": 0.8789},
        ):
            budget_lo, max_rows_lo = flce._select_adaptive_chunk_budget(cuda_dev)
            self.assertEqual(budget_lo, 64 * 1024 * 1024)
            self.assertEqual(max_rows_lo, 2048)
            self.assertEqual(
                compute_optimal_chunk_size(4096, 50304, torch.bfloat16, target_bytes=budget_lo, max_chunk=max_rows_lo),
                640,
            )

        with patch.dict(os.environ, {"PTOPT_FUSED_LCE_CHUNK_MB": "256"}):
            budget_env, max_rows_env = flce._select_adaptive_chunk_budget(torch.device("cpu"))
            self.assertEqual(budget_env, 256 * 1024 * 1024)
            self.assertEqual(max_rows_env, 4096)

    def test_hybrid_compile_lce_eval_zero_breaks_and_ce_dispatch_under_dynamo(self) -> None:
        from ptopt.kernels import fused_linear_cross_entropy as flce

        class _HybridLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(16, 16)
                self.head = nn.Linear(16, 128, bias=False)

            def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
                hid = torch.relu(self.proj(x))
                logits = self.head(hid)
                if y is not None:
                    return logits, F.cross_entropy(logits.view(-1, 128), y.view(-1))
                return logits, None

        cfg = OptimizeConfig(profile="balanced")
        cfg.compile.enabled = True
        mgr = HookManager()
        reg = install_custom_kernels_hooks(cfg, mgr, torch, triton_available=True, sm_override=89)
        assert reg is not None
        try:
            # _custom_cross_entropy must not spoof torch.nn.functional.__module__, otherwise Dynamo
            # dispatches LazyLinearLogits.__torch_function__ -> .materialize() instead of fusing.
            self.assertEqual(F.cross_entropy.__module__, "ptopt.kernels.registry")

            m = _HybridLM()
            orig_head = m.head.forward

            @torch.compiler.disable
            def _dis_head(x: torch.Tensor) -> torch.Tensor:
                return create_lazy_linear_logits(x, m.head.weight, m.head.bias)

            def _cond_head(x: torch.Tensor) -> torch.Tensor:
                if m.training and torch.is_grad_enabled() and x.dim() >= 2 and x.numel() > 0:
                    return _dis_head(x)
                return orig_head(x)

            m.head.forward = _cond_head  # type: ignore[method-assign]

            x = torch.randn(2, 4, 16)
            y = torch.randint(0, 128, (2, 4))

            torch._dynamo.reset()
            prev_suppress = torch._dynamo.config.suppress_errors
            torch._dynamo.config.suppress_errors = False
            try:
                m.eval()
                with torch.no_grad():
                    exp_eval = torch._dynamo.explain(m)(x, y)
                self.assertEqual(exp_eval.graph_break_count, 0)

                torch._dynamo.reset()
                m.train()
                # Direct compile run verifies LazyLinearLogits.view -> _custom_cross_entropy runs fused LCE cleanly
                materialize_calls = 0
                orig_mat = LazyLinearLogits.materialize

                def _spy_mat(self_l: LazyLinearLogits) -> torch.Tensor:
                    nonlocal materialize_calls
                    materialize_calls += 1
                    return orig_mat(self_l)

                compiled = torch.compile(m, backend="eager")
                with patch.object(LazyLinearLogits, "materialize", new=_spy_mat):
                    logits_out, loss_out = compiled(x, y)
                    assert loss_out is not None
                    loss_out.backward()
                self.assertIsInstance(logits_out, LazyLinearLogits)
                self.assertEqual(materialize_calls, 0)
                self.assertIsNotNone(m.head.weight.grad)
                self.assertIsNotNone(m.proj.weight.grad)
            finally:
                torch._dynamo.config.suppress_errors = prev_suppress
                torch._dynamo.reset()
        finally:
            uninstall_custom_kernels_hooks(torch)
            mgr.reset()

    def test_lazy_linear_logits_descriptor_setters_no_dynamo_error(self) -> None:
        import torch._dynamo.variables.functions as dyn_f

        h = torch.randn(2, 4, 16, requires_grad=True)
        w = torch.randn(32, 16, requires_grad=True)
        lazy = create_lazy_linear_logits(h, w, None)

        # Uninitialized subclass instance before _ptopt_lazy_linear is attached stays safe
        raw_base = torch.empty_strided((2, 4, 32), (0, 0, 0), dtype=torch.float32)
        with torch.no_grad():
            uninit = torch.Tensor._make_subclass(LazyLinearLogits, raw_base, False)
        self.assertEqual(uninit.device, torch.device("cpu"))
        self.assertFalse(uninit.is_cuda)

        # Eager grad descriptor set & delete parity
        g = torch.ones(2, 4, 32)
        lazy.grad = g
        self.assertIs(lazy.grad, g)
        del lazy.grad
        self.assertIsNone(lazy.grad)

        def _missing_tp_descr_set(*_args: object, **_kwargs: object) -> object:
            raise AttributeError("type object 'GetSetDescriptorVariable' has no attribute 'tp_descr_set_impl'")

        # Simulate PyTorch <= 2.14 where GetSetDescriptorVariable lacks tp_descr_set_impl
        cls_gs = getattr(dyn_f, "GetSetDescriptorVariable", None)
        patch_ctx = (
            patch.object(cls_gs, "tp_descr_set_impl", _missing_tp_descr_set)
            if cls_gs is not None and hasattr(cls_gs, "tp_descr_set_impl")
            else patch.dict(os.environ, {}, clear=False)
        )

        prev_suppress = torch._dynamo.config.suppress_errors
        torch._dynamo.config.suppress_errors = False
        try:
            with patch_ctx:
                torch._dynamo.reset()

                def _clear_grad(t: torch.Tensor) -> torch.Tensor:
                    t.grad = None
                    return t

                out_full = torch.compile(_clear_grad, backend="eager", fullgraph=True)(lazy)
                self.assertIsInstance(out_full, LazyLinearLogits)

                torch._dynamo.reset()

                def _del_grad(t: torch.Tensor) -> torch.Tensor:
                    del t.grad
                    return t

                out_del = torch.compile(_del_grad, backend="eager")(lazy)
                self.assertIsInstance(out_del, LazyLinearLogits)

                # Simulate hybrid train forward resuming into logits.view(...) + F.cross_entropy
                class _HybridTrainLM(nn.Module):
                    def __init__(self) -> None:
                        super().__init__()
                        self.proj = nn.Linear(16, 16)
                        self.head = nn.Linear(16, 64, bias=False)

                    def forward(self, inp: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                        hid = torch.relu(self.proj(inp))
                        logits = self.head(hid)
                        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))

                cfg = OptimizeConfig(profile="balanced")
                cfg.compile.enabled = True
                mgr = HookManager()
                install_custom_kernels_hooks(cfg, mgr, torch, triton_available=True, sm_override=89)
                try:
                    m = _HybridTrainLM()

                    @torch.compiler.disable
                    def _dis_head(inp: torch.Tensor) -> torch.Tensor:
                        return create_lazy_linear_logits(inp, m.head.weight, m.head.bias)

                    m.head.forward = _dis_head  # type: ignore[method-assign]
                    x = torch.randn(2, 4, 16)
                    y = torch.randint(0, 64, (2, 4))
                    torch._dynamo.reset()
                    compiled = torch.compile(m, backend="aot_eager")
                    logits_out, loss_out = compiled(x, y)
                    loss_out.backward()
                    self.assertIsInstance(logits_out, LazyLinearLogits)
                    self.assertIsNotNone(m.head.weight.grad)
                    self.assertIsNotNone(m.proj.weight.grad)
                finally:
                    uninstall_custom_kernels_hooks(torch)
                    mgr.reset()
        finally:
            torch._dynamo.config.suppress_errors = prev_suppress
            torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
