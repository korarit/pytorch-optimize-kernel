# kernel-pytouch-optimize (`ptopt`)

> **Zero-Code, Hardware-Aware PyTorch Optimization Wrapper** — Accelerates unmodified PyTorch **training** and **inference** scripts up to **4.37x faster** and cuts peak **VRAM by up to 45% (-4.3 GB)** without touching a single line of user model code, altering `nn.Parameter` values, or breaking `state_dict()` checkpoint compatibility.

---

## Why `ptopt` Outperforms Pure PyTorch

By default, **Pure PyTorch** prioritizes conservative cross-hardware defaults and leaves performance optimizations to manual boilerplate (enabling TF32, configuring `autocast` + `GradScaler`, tuning `DataLoader`, converting `channels_last`, enabling `fused=True` optimizers, wiring `torch.compile`, or capturing CUDA Graphs). Applying them ad-hoc is also error-prone—for example, enabling TF32/BF16 on T4 (`sm_75`), spawning DataLoader workers under small `/dev/shm`, hitting FP8 precision drift on small matrices, or breaking checkpoints with `_orig_mod.` prefixes.

**`ptopt` automates the entire stack before `import torch` runs:**
1. **Zero-Code Injection**: Wraps existing commands (`ptopt -- python3 train.py`) via `sitecustomize` and propagates automatically to `DataLoader` worker subprocesses.
2. **Hardware-Aware Auto-Tuning**: Probes GPU SM architecture (`T4 sm_75`, `A100 sm_80`, `L4 sm_89`, `H100 sm_90`, Apple Silicon MPS, or CPU), VRAM headroom, CPU cores, and `/dev/shm` size to apply only safe, profitable passes.
3. **Model & Checkpoint Transparency**: Never modifies `nn.Parameter` dtypes/shapes permanently, strips `_orig_mod.` prefixes, and restores contiguous `NCHW` weights on `state_dict()` so checkpoints load seamlessly between baseline PyTorch and `ptopt`.
4. **Fail-Safe Guards & Fallback**: Protects runs with an AMP `NaN` watchdog, `/dev/shm` & pickle checks, `requires_grad` guards on eval, dynamic-shape compilation fallback, and runtime error recovery.

---

## Architecture

```mermaid
flowchart TD
    UserCmd["User Command: ptopt --profile balanced -- python3 train.py"] --> CLI["CLI Launcher (src/cli/main.py)"]
    CLI --> Detect["Hardware Probe (src/detect: GPU SM, VRAM, CPU, /dev/shm)"]
    Detect --> Resolver["Config Resolver (src/config: safe / balanced / aggressive + overrides)"]
    Resolver --> PreEnv["Pre-Launch Env (src/passes/memory_allocator.py: ALLOC_CONF, OMP, Cache Dirs)"]
    PreEnv --> Spawn["Subprocess with PYTHONPATH Bootstrap"]

    Spawn --> Site["sitecustomize Hook (src/hooks/bootstrap/sitecustomize.py)"]
    Site --> LazyHook["Lazy sys.meta_path Hook on first 'import torch'"]
    LazyHook --> HookMgr["HookManager + Checkpoint Compat (src/hooks/manager.py, checkpoint_compat.py)"]

    HookMgr --> RuntimePasses["Runtime & Memory Passes (runtime.flags TF32/SDPA, memory_allocator)"]
    RuntimePasses --> DataPass["Data & Optim Passes (dataloader.auto_tune + async H2D, optimizer.fusion)"]
    DataPass --> ModuleWrappers["In-Place nn.Module Execution Stack (outer to inner)"]

    subgraph ModuleStack ["Transparent Forward Wrapping Stack"]
        direction TB
        InferOpt["1. inference.opt (inference_mode, CUDA Graphs, Fused Triton FP8)"]
        AmpWrap["2. amp.auto (BF16 / FP16 + GradScaler + Functional Loss FP32 Guard)"]
        LayoutWrap["3. runtime.channels_last (Conv2d spatial >= 128x128)"]
        CompileWrap["4. compile.auto (In-place mod._call_impl + dynamic shape guard)"]
        KernelWrap["5. kernels.auto (Custom Triton RMSNorm / CrossEntropy + Compile-Aware Gate)"]
        InferOpt --> AmpWrap --> LayoutWrap --> CompileWrap --> KernelWrap
    end

    ModuleWrappers --> ModuleStack
    KernelWrap --> UserScript["Unmodified User PyTorch Script (Exact state_dict & Parameters)"]
```

---

## Difference Matrix: Pure PyTorch vs. `ptopt`

| Optimization Area | What Pure PyTorch Does (Default / Ad-Hoc) | What `ptopt` Does Instead | Why `ptopt` Is Better (Colab L4 Results) |
| :--- | :--- | :--- | :--- |
| **1. Tensor Cores & Attention (`TF32` / `AMP` / `SDPA`)** | Keeps `allow_tf32=False`, runs `Linear`/`Conv` in **FP32 SIMT**, leaves functional losses unpromoted outside `autocast`, and suffers heavy dtype `cast churn` on deep LayerNorm MLPs. | Enables `TF32` (`sm >= 80`) + Flash/Mem-Efficient SDPA, wraps top-level `nn.Module` in `BF16` (`L4`/`A100`) or `FP16 + GradScaler` (`T4`), promotes functional losses to `fp32`, and skips eager AMP on cast-heavy MLPs. | **`3.08x`** eager speedup on `transformer_lm:large` (`777.6 -> 252.6 ms`) and nearly halves activation VRAM (`-2.85 GB`). |
| **2. Graph Fusion & `torch.compile`** | Executes **eager op-by-op** (`RMSNorm`, `SwiGLU`, `GELU` write intermediates to HBM separately); manual `torch.compile(model)` mutates `state_dict()` keys with `_orig_mod.`. | Compiles `mod._call_impl` **in-place** (no `_orig_mod.` keys exposed), auto-switches once to `dynamic=True` on shape shifts, and persists graph caches. | Flips `transformer_lm:small` from `0.96x` to **`2.08x`**; boosts `transformer_lm:large` to **`4.36x–4.37x`** and cuts VRAM **`-4.17 GB`**. |
| **3. Memory Layout (`Conv2d`)** | Stores 4D weights/activations in `NCHW`, forcing cuDNN to insert `nchwToNhwc` / `nhwcToNchw` transposes on every conv (~13.7% of GPU time). | Auto-converts weights and 4D inputs to `torch.channels_last` only for CUDA `Conv2d` with spatial size `>= 128x128`, restoring `contiguous` layout on `state_dict()`. | Eliminates `nchwToNhwc` kernels completely -> `cnn_vision:medium` **`3.04x`**, `large` **`2.90x`** (VRAM `-1.73 GB`). |
| **4. Optimizer (`Adam` / `AdamW` / `SGD`)** | Launches per-tensor / multi-tensor update kernels (`_foreach_mul_`, `_foreach_addcdiv_`, `_foreach_lerp_`) accounting for up to **66% of step time** in deep models. | Injects `fused=True` (for CUDA floating-point params) or `foreach=True` fallback directly into `Adam`/`AdamW`/`SGD.__init__`. | Collapses optimizer updates into one `aten::_fused_adamw_` kernel -> `deep_mlp:large` Optim time drops **35.02 ms -> 12.80 ms**. |
| **5. DataLoader & Host-to-Device Copy** | Defaults to `num_workers=0`, `pin_memory=False`, and synchronous blocking `.to(device)` / `.cuda()` host transfers. | Checks `/dev/shm` & fast picklability before tuning `num_workers`, enables `pin_memory=True` + `persistent_workers=True`, and auto-promotes pinned CPU->CUDA `.to()`/`.cuda()` calls to `non_blocking=True`. | `cnn_vision:large` Data/Other phase drops **16.77 ms -> 2.40 ms (~7x)** while preventing `/dev/shm` crashes. |
| **6. Custom Triton Kernels & Compile Cooperation** | Eager `RMSNorm` and `CrossEntropy` launch multiple HBM-bound kernels; naive `autograd.Function` custom ops inserted inside compiled graphs trigger graph breaks. | Ships fused Triton `RMSNorm`, online-softmax `CrossEntropy` (zero `.item()` CPU-GPU sync), `SwiGLU`, and `LayerNorm` for eager speedup (**3.80x–4.11x**), plus a **Compile-Aware Gate** that defers internal wrappers to Inductor when compiling. | Micro-benchmarks beat both Eager & `torch.compile`: `RMSNorm [large]` **`6.87x`** (`0.636 ms` vs `0.640 ms` compile), `CrossEntropy [large]` **`3.55x`** (`3.360 ms` vs `3.372 ms` compile). |
| **7. Inference Latency (`bs=1`) & FP8 (`sm >= 89`)** | Runs `.eval()` with autograd tracking and Python kernel dispatch overhead; naive eager FP8 drifts on small tensors and spends >50% GPU time quantizing. | Enters `torch.inference_mode()` (guarded against `requires_grad`/`enable_grad`), uses `reduce-overhead` CUDA Graphs for static shapes, and gates opt-in FP8 (`m_rows >= 512`) with a fused Triton 2-pass quantizer + lazy weight cache. | `bs=1` latency drops `1.53 ->` **`0.48 ms` (`3.19x`)**; `bs=64` hits **`4.14x`**; Fused FP8 `bs=64` improves `1.64x ->` **`2.49x`** (`-185 MB` vs naive FP8). |
| **8. Allocator & Cold-Start Cache** | Default CUDA allocator risks segment fragmentation; Inductor/Triton compile caches stay fragmented across runs. | Configures `expandable_segments:True`, physical-core `OMP/MKL_NUM_THREADS`, `TORCHINDUCTOR_CACHE_DIR`, `TORCHINDUCTOR_FX_GRAPH_CACHE=1`, `TORCHINDUCTOR_AUTOGRAD_CACHE=1`, and `TRITON_CACHE_DIR` pre-import. | Achieves `num_alloc_retries=0` and cuts Warm Step-0 compile time **24.9s -> 6.2s**. |

---

## Verified Benchmarks (`NVIDIA L4 sm_89` on Google Colab)

All metrics below are directly measured from the reports in [`report/`](report/):

### Training Benchmarks
| Workload & Scale | Baseline | `ptopt` Optimized | Speedup | Peak VRAM Delta | Report |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`transformer_lm:large`** | 795.53 ms | **182.03 ms** | **4.37x** | **-4,167.4 MB** (`-4,309.2 MB` aggressive) | [173112](report/173112_20261117-phase-3.7.txt) |
| **`transformer_lm:medium`** | 99.92 ms | **26.27 ms** | **3.80x** | **-742.0 MB** (`-777.5 MB` aggressive) | [173112](report/173112_20261117-phase-3.7.txt) |
| **`transformer_lm:small`** | 11.72 ms | **5.65 ms** | **2.08x** | **-121.9 MB** | [144108](report/144108_20261117-phase-3.5.txt) |
| **`cnn_vision:medium`** | 65.75 ms | **21.66 ms** | **3.04x** | **-385.6 MB** | [141819](report/141819_20261117-phase-3.4.txt) |
| **`cnn_vision:large`** | 280.96 ms | **96.84 ms** | **2.90x** | **-1,728.1 MB** | [144108](report/144108_20261117-phase-3.5.txt) |
| **`deep_mlp:large`** | 50.45 ms | **22.10 ms** | **2.28x** | **-382.6 MB** | [141819](report/141819_20261117-phase-3.4.txt) |

### Inference & Standalone Triton Kernel Micro-Benchmarks
| Benchmark Case | Baseline | `ptopt` Optimized | Gain | Report |
| :--- | :---: | :---: | :---: | :--- |
| **`infer_bs1_small` (Latency)** | 1.53 ms | **0.48 ms** (`CUDA Graphs`) | **3.19x** (`2,079 samp/s`) | [172306](report/172306_20261117-phase-3.6.txt) |
| **`infer_bs1_medium` (Latency)** | 3.54 ms | **1.25 ms** (`CUDA Graphs`) | **2.82x** (`798.7 samp/s`) | [172306](report/172306_20261117-phase-3.6.txt) |
| **`infer_bs64_medium` (Throughput)** | 44.31 ms | **12.10 ms** (`BF16 + Compile`) | **3.66x** (`-63.9 MB` VRAM) | [172306](report/172306_20261117-phase-3.6.txt) |
| **`infer_bs64_large` (Throughput)** | 373.93 ms | **90.22 ms** (`BF16 + Compile`) | **4.14x** (`-187.5 MB` VRAM) | [172306](report/172306_20261117-phase-3.6.txt) |
| **`RMSNorm [large]` (8192x1025)** | 4.371 ms (Eager) | **0.636 ms** (Triton) vs 0.640 ms (Compile) | **6.87x vs Eager** (`98.4 GB/s`) | [173112](report/173112_20261117-phase-3.7.txt) |
| **`CrossEntropy [large]` (8192x8192)** | 11.94 ms (Eager) | **3.360 ms** (Triton) vs 3.372 ms (Compile) | **3.55x vs Eager** (`74.4 GB/s`) | [173112](report/173112_20261117-phase-3.7.txt) |

---

## Installation & Quick Start

### 1. Install
```bash
pip install -e .
```
*(You can also invoke directly via `python3 -m src.cli.main ...` without installing.)*

### 2. Wrap Any Existing PyTorch Script (Zero Code Changes)
```bash
# Recommended balanced profile (safe + fast across training & inference)
ptopt --profile balanced -- python3 your_train_script.py --epochs 10

# Conservative profile (strict numerics, ideal for initial parity checks)
ptopt --profile safe -- python3 your_train_script.py

# Maximum optimization (enables CUDA Graphs on static eval + custom Triton kernels)
ptopt --profile aggressive -- python3 your_train_script.py
```

### 3. Inspect, Simulate Hardware & Override Passes
```bash
# Inspect detected hardware, rules triggered, and final resolved config
ptopt --explain

# Simulate config resolution for another target architecture
# Supported presets: t4, sm_75, low_shm_t4, a100, sm_80, sm_86, l4, sm_89, h100, sm_90, cpu, mps
ptopt --target-arch l4 --profile balanced --explain

# Disable a specific pass or override fields via dotted path
ptopt --profile balanced --no-compile -- python3 short_finetune.py
ptopt --profile aggressive --override compile.mode=reduce-overhead -- python3 eval.py

# Print applied/fallback audit report or capture PyTorch Profiler top ops
ptopt --fallback-report --verbose -- python3 your_script.py
PTOPT_DEBUG_PROFILE=1 PTOPT_TORCH_PROFILE=1 ptopt -- python3 your_script.py
```

### Opt-In Environment Flags
- `PTOPT_INFERENCE_FP8=1`: Enables FP8 `_scaled_mm` inference with fused Triton quantization for `nn.Linear` (`m_rows >= 512`) on `sm >= 89` (L4/H100).
- `PTOPT_INFERENCE_QUANT=int8`: Enables CPU INT8 dynamic quantization during `.eval()`.
- `PTOPT_FORCE_ALL_KERNELS=1`: Forces all custom Triton kernels even when `torch.compile` is enabled.
- `PTOPT_CHANNELS_LAST=1`: Forces `channels_last` conversion even for small spatial CNN inputs.

---

## Repository Structure

- [`src/detect/`](src/detect/): Hardware fingerprinting (GPU SM, VRAM, CPU ISA/cores, `/dev/shm`, PyTorch/Triton/CUDA).
- [`src/config/`](src/config/): Schema v1 dataclasses, built-in hardware rules, and `--override` resolver.
- [`src/hooks/`](src/hooks/): `sitecustomize` pre-torch bootstrap, `HookManager`, checkpoint `_orig_mod.`/`channels_last` compatibility, and Deep Debug Profiler telemetry.
- [`src/passes/`](src/passes/): Optimization passes (`runtime_flags`, `memory_allocator`, `dataloader_tune`, `optimizer_fusion`, `auto_amp`, `channels_last`, `auto_compile`, `inference_opt`).
- [`src/kernels/`](src/kernels/): Custom Triton kernels (`rmsnorm`, `layernorm`, `cross_entropy`, `swiglu`) with `KernelSpec` & `FallbackRegistry`.
- [`ai-docs-dev/optimize-history/`](ai-docs-dev/optimize-history/): Root-cause & profiler writeups ([Phase 3.1–3.5](ai-docs-dev/optimize-history/phase-3.1-to-3.5-root-cause-and-gains.md), [Phase 3.6–3.7](ai-docs-dev/optimize-history/phase-3.6-to-3.7-inference-kernels-and-tuning.md)).
- [`scripts/`](scripts/) & [`tests/`](tests/): Multi-scale A/B experiment runners (`exp_phase*.py`), unit tests, and migration suites.
