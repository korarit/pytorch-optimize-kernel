# Phase 7: Hardening, Calibration Integrity & Namespace Isolation

**Status**: `COMPLETED` (Local Fast Gate `14/14 PASS` + Colab L4 Phase 7 `8/8 PASS` [`061933_20261119-phase-7.txt`] & Full Gate `15/15 PASS` [`064444_20261119-phase-4.txt`])  
**Baseline Commit**: `b7ec00b` (`main`) → Verified at `c9e89a6`

---

## 1. Objectives & Audit Scope

Phase 7 resolves every structural, correctness, compatibility, and performance-regression bug uncovered in the post-Phase-6 codebase audit while preserving all core invariants:
- **Zero user-code modification** (`pipline-for-test/workloads/*.py` remain untouched).
- **Zero external runtime dependencies** (`dependencies = []` in `pyproject.toml`).
- **100% `state_dict()` & checkpoint cross-load parity** (safe save/load across `ptopt` and vanilla PyTorch).
- **Mandatory Phase 3/5/6-grade 7-Part Deep Debug Detailed Text Report** logging for GPU/CPU verification via `scripts/phase5_report_helper.py` with `PTOPT_DEBUG_PROFILE=1` and `PTOPT_TORCH_PROFILE=1`.

---

## 2. Workstreams & Exact Fixes

### 7.1 Telemetry & FP8 Static Calibration Integrity (`inference_opt.py`, `manager.py`, `telemetry_hook.py`)
1. **Fix `fp8_compile_static` Step-0 / Telemetry Desync (`s0..s3` Phantom Submodule Steps)**:
   - Previously, `TelemetryCollector` wrapped `nn.Module.__call__` *inside* `ModuleCallDispatcher`, so calling `mod_self._call_impl(*args, **kwargs)` inside `inference_opt` while `TelemetryCollector._module_depth == 0` caused each submodule (`in_proj`, `encoder`, `norm`, `head`) to be recorded as a fake top-level step (`s0..s3`), pushing real Step-0 `torch.compile` to `s4` inside the steady-state profiler window (`[4, 9]`).
   - Install `TelemetryCollector` *outer* to `ModuleCallDispatcher` (ordering hooks cleanly) **and** add an explicit internal calibration/recompute suppression context in `TelemetryCollector` so internal `ptopt` calibration forwards (`call_impl`) never emit bogus `step_times_ms` or pollute `torch_profiler` step counting.
2. **Respect `calib_steps` & Trim Calibration VRAM Overhead**:
   - Honor `PTOPT_FP8_CALIB_STEPS` (`calib_steps`, default `2`) instead of setting `_ptopt_fp8_static_ready = True` after a single forward pass.
   - Immediately drop calibration output references (`del _calib_out`) and avoid retaining unnecessary eager activation buffers after calibration completes (`empty_cache()` when safe on CUDA after initial multi-step calibration) so static FP8 peak VRAM tracks baseline instead of spiking (`~1019 MB` -> baseline range).

### 7.2 Package Namespace Isolation (`src/` -> `ptopt/`)
1. **Eliminate Top-Level `src/` Import Shadowing**:
   - Move package root from `<repo>/src/` to `<repo>/ptopt/` (`pyproject.toml`: `ptopt = "ptopt.cli.main:main"`, `include = ["ptopt*"]`, `[tool.setuptools.package-data] "ptopt.hooks.bootstrap" = ["sitecustomize.py"]`).
   - Update `ptopt/hooks/bootstrap/sitecustomize.py` and `ptopt/cli/main.py` (`build_child_env`) so injected child `PYTHONPATH` exposes `<repo>` (housing `ptopt/`) and `<repo>/ptopt/hooks/bootstrap`, never a top-level `src/` package.
   - Update all internal imports, entrypoints, scripts, tests, CI (`.github/workflows/ci.yml`), and docs (`README.md`, `ai-docs-dev/USER-GUIDE.md`) from `src.*` to `ptopt.*`.
2. **Verification Requirement**:
   - Add an end-to-end subprocess integration test launching a temporary user project that defines its own `src/model.py` (`from src.model import TinyNet`) under `ptopt` and verifying exit code `0`.

### 7.3 Hybrid LCE, Dynamo Boundary, Compile/OOM & Activation Checkpointing
1. **Tighten Hybrid LCE Head Guard (`ptopt/kernels/registry.py`)**:
   - Remove the overly broad `out_feat > in_feat * 4` fallback in `is_head_shape` that misclassified intermediate FFN projections (`gate_proj`, `up_proj`, `linear1`) as language-model heads when `out_features >= min_lce_vocab`, causing unexpected Dynamo graph breaks and unnecessary `LazyLinearLogits` materializations inside transformer blocks.
   - Restrict LCE wrapping to genuine terminal LM heads (`sub is last_linear` with `out_features >= min_lce_vocab`).
2. **`LazyLinearLogits` Cross-Boundary Dynamo Safety (`ptopt/kernels/fused_linear_cross_entropy.py`)**:
   - Replace `device="meta"` backing storage in `LazyLinearLogits` with a real-device 0-stride expanded scalar tensor (`torch.zeros((), dtype=eff_out_dtype, device=hidden.device).expand(out_shape)`), eliminating `InternalTorchDynamoError: Expected meta != nullptr to be true` when `F.cross_entropy(self.lm_head(h)..., target)` executes inside a compiled `forward()`.
   - Verify `.device`, `.dtype`, `.shape`, minimal memory footprint (1 scalar element), `materialize()`, backward gradients, and `torch.compile` compatibility on CPU and CUDA.
3. **Activation Checkpointing & Compile OOM Coordination (`ptopt/passes/auto_compile.py`, `ptopt/passes/activation_checkpoint.py`)**:
   - In `auto_compile.py`, do not swallow `OutOfMemoryError` into permanent eager `compile.auto` fallback before `activation_checkpointing` (`mode="auto"`, non-strict) has an opportunity to rescue and recover.
   - In `activation_checkpoint.py`, reset CUDA peak memory stats (`reset_peak_memory_stats()`) immediately after enabling partial checkpointing (`ratio < 1.0`, e.g., `0.5`) so the next step measures actual post-checkpoint peak memory rather than falsely escalating to `ratio=1.0` from pre-checkpoint peak stats.

### 7.4 Core Subsystem Hardening (`checkpoint_compat`, `channels_last`, `auto_amp`, `module_dispatch`, `rope`, `fingerprint`, `cli`, `docs`)
1. **`ptopt/hooks/checkpoint_compat.py`**:
   - Handle `injected_fused` alongside `injected_capturable` in `_ensure_contiguous_optimizer_state_dict` so CUDA `step` state tensors injected by fused optimizers are safely moved to CPU before `state_dict()`.
   - Extend `sync_optimizer_state_memory_format` to handle 5D `channels_last_3d` tensors in addition to 4D `channels_last`.
2. **`ptopt/passes/channels_last.py`**:
   - Convert `elif is_c3d` to independent `if is_c2d` + `if is_c3d` so models containing both `Conv2d` and `Conv3d` convert both parameter sets cleanly, and propagate `_ptopt_channels_last_done` markers to converted submodules for clean checkpoint sync.
3. **`ptopt/passes/auto_amp.py`**:
   - Add `"NAdam"` and `"RAdam"` to `_OPTIM_STEP_CLASSES`.
   - Scope `_last_forward_skipped_amp` cleanly (track training-forward context rather than letting unrelated helper/eval module calls silently suppress the AMP loss watchdog or `GradScaler` updates).
   - Prevent `enqueue_loss_stability_check` from dropping an unpolled `_watchdog_pending_slot` (queue pending slots or flush/check prior pending slot before reuse).
4. **`ptopt/hooks/module_dispatch.py`**:
   - Convert shared `_depth` and `_recompute_depth` counters to `threading.local()` state for multi-threaded `DataLoader` / inference safety.
5. **`ptopt/kernels/rope_or_swiglu.py`**:
   - Broadcast 3D `(B|1, S, D)` `cos`/`sin` tensors with 4D `(B, H, S, D)` `q`/`k` tensors in `ref_rope` (`cos.unsqueeze(1)`) for exact fallback parity with `triton_rope`.
6. **`ptopt/detect/fingerprint.py`**:
   - Unify `_fingerprint_cache_path` cache key regardless of whether `torch_mod` is passed (`importlib.metadata` or cached CUDA version lookup without forcing eager `torch` import), and cheaply include Linux NVIDIA driver version (`/proc/driver/nvidia/version`) when present without subprocess overhead.
7. **`ptopt/cli/main.py` & `ptopt/cli/explain.py`**:
   - Deduplicate `--set` override parsing (`parse_override_value` / `parse_cli_overrides` in a shared helper) and support comma-separated `list[str]` parsing for list-typed config fields such as `kernels.allow_list`.
8. **Docs & README Sync**:
   - Fix markdown table pipe typo in `ai-docs-dev/optimize-history/phase-6-dispatch-hybrid-and-llm-polish.md`.
   - Sync `README.md` and `ai-docs-dev/USER-GUIDE.md` with Phase 6 + Phase 7 (`ptopt` package layout, `ModuleCallDispatcher`, and opt-in environment flags).

---

## 3. Mandatory Deep Debug Verification & Report Specification

`scripts/verify_phase7.py` MUST reuse `scripts/phase5_report_helper.py` with `PTOPT_DEBUG_PROFILE=1` and `PTOPT_TORCH_PROFILE=1` to write a complete **7-part Phase 3/5/6-grade Deep Debug Detailed Text Report** at `report/HHMMSS_YYYYMMDD-phase-7.txt`:

1. **Part 1 — Report Header & Environment Snapshot**: Timestamp, `git_sha`, hardware/GPU/CPU probe, Python/PyTorch versions, `ptopt` namespace check, active env flags.
2. **Part 2 — Ablation Matrix + P95 / P99 / E2E Latency Summary**:
   - CUDA GPU (L4 on Colab): `fp8_compile_static` (confirming `P99` / `Steady` no longer inflated by `s4` compile stall), `hybrid_compile_lce`, `wide_ffn_lm` (no FFN head misclassification), `compiled_fwd_lce` (`F.cross_entropy` inside compiled `forward()`), `mixed_conv2d3d_channels_last`, `nadam_radam_amp`.
   - CPU Local Baseline: Targeted microbenchmarks + feature verification matrix + `[WAIT FOR USER]` markers for GPU-only cells when run locally without CUDA.
3. **Part 3 — Phase Breakdown (`Init`, `Warmup/Compile`, `Steady`, `Peak VRAM`, `Headroom`)**: Verify calibration overhead stays in Warmup, not Steady, and peak VRAM remains tight.
4. **Part 4 — CUDA Allocator & Memory Headroom**: `reserved`, `active`, `inactive_split`, `alloc_retries`, `oom_count`.
5. **Part 5 — Top PyTorch Profiler Ops (Steady Window)**: Must confirm profiler `[warmup, steady]` window matches true step indices cleanly.
6. **Part 6 — Warmup vs Steady Step-by-Step Timing Trace**: Assert `len(step_times_ms) == configured_steps` (`s0..sN-1`) with zero phantom submodule step entries during FP8 static calibration.
7. **Part 7 — Automated Diagnosis, Kernel/Fallback Audit & Final Status**: Per-fix verification checklist, `STATUS : PENDING_USER_REVIEW` (or `PASS` after Colab GPU verification).

---

## 4. Verification Plan & Exit Criteria

1. **Local CPU Gate**:
   - `python3 -m unittest discover -s tests/unit` & targeted compile tests (`tests/migrate/test_phase3_5_compile.py`, `tests/unit/test_phase7_features.py`).
   - `python3 scripts/verify_phase7.py` -> generates CPU-verified sections + clean 7-part report format in `report/HHMMSS_YYYYMMDD-phase-7.txt` with explicit `[WAIT FOR USER]` GPU ablation placeholders.
   - `python3 scripts/run_gate.py --mode fast` -> 100% PASS.
2. **Colab GPU Verification (`COMPLETED`)**:
   - `python3 scripts/verify_phase7.py` -> [`report/061933_20261119-phase-7.txt`](../report/061933_20261119-phase-7.txt) (`8/8 PASS`, `fp8_compile_static 7.42x`, `P99 90.08ms`, VRAM `1061.8 MB`, `Warm Hit 0.32ms`)
   - `python3 scripts/run_gate.py --mode full` -> [`report/064444_20261119-phase-4.txt`](../report/064444_20261119-phase-4.txt) (`15/15 Stages PASS`, `150 Unit Tests OK`, `12/12 Migrate Suites PASS`, `24/24 Smoke Cells PASS`)
