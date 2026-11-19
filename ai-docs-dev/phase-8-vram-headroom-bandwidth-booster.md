# Phase 8: VRAM-Headroom Bandwidth Booster & Detailed Telemetry

**Status**: `IN_PROGRESS` (Local CPU verification pending -> Colab GPU verification required)  
**Baseline Commit**: `c0b59b5` (`main`)

---

## 1. Objectives & Motivation

Up through Phase 7, `ptopt` reacted **defensively** when VRAM got tight (e.g. `activation_checkpoint.py` triggers above `82%` VRAM, `vram_lt_12gb` disables CUDA Graphs). However, when a GPU has **abundant free VRAM** (e.g. NVIDIA L4 `24 GB`, A100 `40/80 GB`, H100 `80 GB` using `< 50–65%` VRAM), `ptopt` left GPU compute & memory bandwidth under-fed:
- `dataloader.cuda_prefetch` stayed `False` by default, leaving PCIe H2D copies synchronous or single-buffered.
- `dataloader.prefetch_factor` stayed `2` and workers were capped at `4`.
- `balanced` kept `runtime.cudnn_benchmark=False` even on `20+ GB` GPUs.
- `torch._inductor.config.coordinate_descent_tuning` stayed `False` in `max-autotune`, and FP8 weight retention / prewarm didn't adapt to live VRAM headroom.

Phase 8 adds an **opt-in VRAM-Headroom Bandwidth Booster** (`--bandwidth-boost` / `PTOPT_BANDWIDTH_BOOST=1` / `runtime.bandwidth_boost=true`, **disabled by default**) across all 4 layers **without altering training math (`batch_size` / `lr`)**, paired with detailed runtime telemetry and audit logging.

---

## 2. Workstreams

### 8.1 Adaptive Multi-Batch GPU Prefetch Queue + Auto VRAM Back-Off (`ptopt/passes/dataloader_tune.py`)
1. **Live VRAM Headroom Probe (`probe_cuda_vram_headroom`)**:
   - Reads `memory_allocated`, `memory_reserved`, and cached per-device `total_memory` safely on CUDA (graceful `None` on CPU or mock `torch`).
2. **Adaptive Queue Depth (`compute_adaptive_prefetch_depth`)**:
   - When `runtime.bandwidth_boost=True`: allows up to `max_depth=3` (`used_ratio < 0.50`: depth `3`; `0.50 <= used_ratio < 0.68`: depth `2`; `0.68 <= used_ratio < 0.78`: depth `1`; `used_ratio >= 0.78`: depth `0` **Auto Back-Off** + `FALLBACK` log).
   - When `runtime.bandwidth_boost=False` (default): caps `cuda_prefetch` lookahead at `max_depth=1` (single-batch lookahead, unless `PTOPT_CUDA_PREFETCH_DEPTH` / `PTOPT_CUDA_PREFETCH_MAX_DEPTH` overrides).
   - Synchronizes each staged batch individually with per-batch `cuda.Event().record(copy_stream)` -> `compute_stream.wait_event(event)` (falling back to `wait_stream` on mocks without `Event`) so compute never stalls on later prefetched copies.
   - Overridable via `PTOPT_CUDA_PREFETCH_DEPTH` / `PTOPT_CUDA_PREFETCH_MAX_DEPTH` / `PTOPT_CUDA_PREFETCH_BACKOFF_RATIO`.
3. **CPU-Safety Guard for Auto-Enabled Prefetch**:
   - When `PTOPT_ACTIVE=1` and `PTOPT_CUDA_PREFETCH` is not forced, only wrap `DataLoader` iterators after `torch.cuda.is_initialized()` is `True`, preventing CPU-only scripts on GPU machines from receiving CUDA tensors unexpectedly.
4. **Detailed Prefetch Telemetry**:
   - Updates `torch._ptopt_prefetch_stats` (`loaders_wrapped`, `batches_prefetched`, `batches_served`, `max_queue_depth`, `current_depth`, `backoff_events`, `last_vram_used_ratio`, `last_vram_free_mb`) and emits detailed `HookManager` audit events.

### 8.2 Opt-In Bandwidth Boost Flag & High-VRAM Rule (`ptopt/cli/main.py`, `schema.py`, `resolver.py`, `rules_builtin.py`)
1. **Default-Disabled Flag (`runtime.bandwidth_boost: bool = False`)**:
   - Opt in via `--bandwidth-boost`, `PTOPT_BANDWIDTH_BOOST=1`, or `--override runtime.bandwidth_boost=true`.
2. **Post-Override Resolver Pass `vram_bandwidth_boost` (`ConfigResolver._apply_bandwidth_boost_if_enabled`)**:
   - Condition: `cfg.runtime.bandwidth_boost == True`, CUDA backend, `profile != "supersafe"`, and `vram_mb >= 12288` (`12+ GB`, or `0` mock).
   - Updates (preserving any explicit user overrides like `--no-dataloader-tune` or `--override dataloader.prefetch_factor=...`):
     - `runtime.cudnn_benchmark = True` (trades ~125 MB for optimal cuDNN Conv kernels; `auto_compile.py` still auto-disables if dynamic Conv shapes appear).
     - When `not fp.is_low_shm` and DataLoader is active:
       - `dataloader.cuda_prefetch = True`
       - `dataloader.prefetch_factor = 4`
       - `dataloader.num_workers = min(8, max(2, cores // 2))` if `total_ram_mb >= 16384` else `min(4, max(2, cores // 2))` (`recommended_workers_high_vram`)
   - `supersafe`, CPU/non-CUDA backends, and `< 12 GB` GPUs ignore `bandwidth_boost` safely and log a guard trace entry.

### 8.3 High-VRAM Inductor Bandwidth Autotuning (`ptopt/passes/auto_compile.py`)
1. **`configure_inductor_bandwidth_tuning` (Lazy on First Compile)**:
   - When `(cfg.runtime.bandwidth_boost and chosen_mode.startswith("max-autotune"))` (e.g. `aggressive --bandwidth-boost`) OR `PTOPT_INDUCTOR_BW_TUNE=1`, and CUDA VRAM `total_mb >= 16384` with `used_ratio < 0.75`:
     - Enables `torch._inductor.config.coordinate_descent_tuning = True` and `epilogue_fusion = True` so Inductor searches wider Triton block/warp/stage tiles to saturate HBM and Tensor Core bandwidth.
     - Kept off under `balanced --bandwidth-boost` (`chosen_mode="default"`) unless `PTOPT_INDUCTOR_BW_TUNE=1`, avoiding cold-compile latency explosions.
     - Records `torch._ptopt_inductor_bw_tune` and logs `compile.inductor_bw_tune` (`APPLIED` / `SKIPPED`) with exact `vram_free_mb`.

### 8.4 VRAM-Headroom-Aware Inference Resident Cache & Prewarm (`ptopt/passes/inference_opt.py`)
1. **Opt-In / Auto FP8 Dual-Resident vs Auto-Drop Policy (`decide_fp8_weight_retention`)**:
   - When `PTOPT_FP8_DROP_BF16_EVAL` is unset:
     - `bandwidth_boost=False` (default): uses `dual_resident_default` (keeps Phase 7 behavior unchanged).
     - `bandwidth_boost=True`: keeps `boost_dual_resident` below `85%` VRAM used, and switches to `boost_auto_drop` at `used_ratio >= 0.85` to preserve OOM headroom.
   - When `PTOPT_FP8_DROP_BF16_EVAL=auto`: keeps `auto_dual_resident` below `80%` and `auto_drop` at `used_ratio >= 0.80`.
   - Logs `inference.fp8_cache` event and records `torch._ptopt_fp8_cache_policy`.
2. **Proactive Medium-Batch FP8 Prewarm (`M >= 128`)**:
   - When `cfg.runtime.bandwidth_boost=True` and VRAM headroom is high (`total_mb >= 16384` and `used_ratio < 0.65`), prewarms FP8 weight caches ahead of `torch.compile` starting at `M >= 128` (instead of waiting for `512`).

---

## 3. Detailed Logging & Verification Plan

1. **Telemetry & Real-Bench Integration**:
   - [`TelemetryCollector`](../ptopt/hooks/telemetry_hook.py) surfaces `bandwidth_vram_booster` inside `telemetry.json` / `debug_profile` (`prefetch_stats`, `inductor_bw_tune`, `fp8_cache_policy`).
   - [`real_bench.py`](../pipline-for-test/real_bench.py) prints a `[VRAM HEADROOM & BANDWIDTH BOOSTER TELEMETRY]` table and live summary fields.
2. **Unit Tests (`tests/unit/test_phase8_bandwidth_booster.py`)**:
   - Adaptive queue depth `3 -> 2 -> 1 -> 0` back-off, per-batch `cuda.Event()` sync, cached device properties, default `max_depth=1` vs opt-in `max_depth=3`, CPU-safety guard, `--bandwidth-boost` / `vram_bandwidth_boost` rule matrix (and guards), Inductor BW tuning gates, FP8 default vs boost vs auto policies, and telemetry capture.
3. **7-Part Deep Debug Report (`scripts/verify_phase8.py`)**:
   - Writes `report/HHMMSS_YYYYMMDD-phase-8.txt` with phase breakdown, allocator headroom, top ops, and `bandwidth_vram_booster` logs (with clean `[WAIT FOR USER]` markers when run on CPU-only CI).
4. **Regression Gate**:
   - `python3 scripts/run_gate.py --mode fast` (Local CPU) + `scripts/verify_phase8.py` on Colab GPU (`L4`/`A100`).
