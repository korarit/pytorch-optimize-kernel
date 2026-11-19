# `ptopt` User & Developer Guide

Zero-code hardware-aware runtime optimizer for PyTorch training and inference.

---

## 1. CLI Quickstart & Common Flags

Run any existing PyTorch training or inference script without editing source code:

```bash
# Default balanced profile (recommended for most CUDA workloads)
ptopt -- python train.py --epochs 10

# Or via module invocation without installing console script
python3 -m ptopt.cli.main --profile balanced -- python train.py
```

### Diagnostic & Inspection Flags

| Flag | Purpose |
| :--- | :--- |
| `--doctor` | Inspect GPU, CUDA/PyTorch/Triton versions, `/dev/shm`, host allocators, cache/report write permissions, and print the recommended profile. |
| `--explain` | Show detected hardware fingerprint, matched config rules across all 5 layers, and resolved Schema v1 settings without executing user code (when no command follows). |
| `--show-config` | Print only the resolved Schema v1 JSON config. |
| `--dump-config PATH` | Save full hardware fingerprint + resolved config + rule trace to JSON. |
| `--fallback-report` | Print `[APPLIED]`, `[SKIPPED (by arch/config)]`, and `[FALLBACK (runtime error)]` summary to `stderr` when the wrapped command exits. |
| `--save-report DIR` | Write structured plain-text run diagnostics into `DIR/HHMMSS_YYYYMMDD-run-<name>.txt` (use `--run-name NAME` to customize). |
| `--target-arch NAME` | Simulate target hardware (`t4`, `low_shm_t4`, `a100`, `l4`, `h100`, `cpu`, `mps`) with `--doctor` or `--explain`. |

### Execution & Override Flags

| Flag | Purpose |
| :--- | :--- |
| `--profile {supersafe,safe,balanced,aggressive}` | Select safety vs performance trade-off (default: `balanced`). |
| `--override section.field=value` | Fine-tune any Schema v1 field (e.g. `--override compile.cache_dir=/tmp/ptopt_cache`). |
| `--no-compile`, `--no-amp`, `--no-dataloader-tune`, `--no-kernels` | Disable individual subsystems cleanly without editing code. |
| `--compile-mode {default,reduce-overhead,max-autotune}` | Override `torch.compile` mode. |
| `--amp-dtype {fp32,fp16,bf16}` | Override AMP tensor dtype. |
| `--strict` | Re-raise hook/pass setup exceptions instead of falling back gracefully. |
| `--verbose` | Emit runtime hook warnings and fallback summary to `stderr`. |

---

## 2. Profile Matrix

| Profile | Math Precision | Enabled CUDA Passes (`sm >= 80` vs `sm_75`) | When to Use |
| :--- | :--- | :--- | :--- |
| **`supersafe`** | Strict **FP32 SIMT** (`tf32=False`, `0.0` drift) | Allocator (`expandable_segments`) + `Fused Optimizer` + `inference_mode` | Reproducibility audits, numerical debugging, financial/scientific tasks requiring exact FP32 parity. |
| **`safe`** | **FP32 weights + TF32 Tensor Cores** (`sm >= 80`, off on T4) | `supersafe` + `TF32` + Flash/Mem-Efficient SDPA + `DataLoader` auto-tune | Faster FP32 training/inference without AMP casting or `torch.compile` warmup. |
| **`balanced`** | **BF16 AMP** (`sm >= 80`) / **FP16+Scaler** (`sm < 80`) | `safe` + `channels_last` (Conv2d >=128x128) + **Auto-AMP** + **Auto-Compile (`default`)** | Recommended default for general training and batch inference (`3.1x–3.8x` on L4). |
| **`aggressive`** | **BF16/FP16 AMP** + TF32 *(optional opt-in FP8 inference)* | `balanced` + `cudnn_benchmark` + **Custom Triton Kernels** + **CUDA Graphs (`reduce-overhead`)** / `max-autotune` | Maximum large-scale training throughput (`4.41x` on L4) and minimum `bs=1` latency (`3.18x+`). |

---

## 3. Step-by-Step Google Colab Usage

1. **Select GPU Runtime**: `Runtime -> Change runtime type -> T4 / L4 / A100 GPU`.
2. **Clone & Install Without Touching Colab's CUDA PyTorch**:
   ```bash
   git clone <your-repo-url> pytorch-optimize-kernel
   cd pytorch-optimize-kernel
   pip install --no-deps -e .
   ```
   *(Using `--no-deps` preserves Colab's preinstalled CUDA PyTorch & Triton builds.)*
3. **Run Environment Diagnostics First**:
   ```bash
   ptopt --doctor
   ```
   Check `/dev/shm` and cache writability. Optional CPU allocator boost on Colab Ubuntu:
   ```bash
   apt-get update -qq && apt-get install -y -qq libjemalloc2
   ```
4. **Run Your Training Script Zero-Code + Audit Passes**:
   ```bash
   ptopt --profile balanced --fallback-report --save-report report/ -- python train.py
   ```
5. **Persist Compile Cache Across Colab Sessions (Optional)**:
   ```bash
   ptopt --override compile.cache_dir=/content/drive/MyDrive/ptopt_cache -- python train.py
   ```
   Cache directories are automatically scoped by `(gpu_arch, torch_version, triton_version)` in `.ptopt_cache_meta.json`, so switching from T4 to L4 never reuses stale Inductor/Triton artifacts.

---

## 4. Developer Guide: Extending `ptopt`

### 4.1 Adding a New GPU Rule

1. Open [`ptopt/config/rules_builtin.py`](../ptopt/config/rules_builtin.py).
2. Add a `ConfigRule` into `get_builtin_rules()` under the appropriate layer (`gpu_arch`, `vram_tier`, or `user_profile`):
   ```python
   ConfigRule(
       name="gpu_blackwell_sm100",
       layer="gpu_arch",
       reason="Blackwell sm_100: enable TF32 and Flash SDP",
       condition=lambda fp, _p: _is_cuda(fp) and fp.sm_int >= 100,
       updates={"runtime.tf32": True, "attention.flash_sdp": True},
   )
   ```
3. Add a mock preset in `_MOCK_PRESETS` inside [`ptopt/cli/explain.py`](../ptopt/cli/explain.py) if needed, and add unit coverage in `tests/unit/test_config_resolver.py`.

### 4.2 Adding a New Optimization Pass

1. Create `ptopt/passes/my_pass.py` exposing `apply_my_pass(cfg: OptimizeConfig, manager: HookManager) -> str`:
   - **Never import `torch` at module top level** (keep `sitecustomize` lazy-import safe). Fetch `sys.modules.get("torch")` or pass `torch_mod`.
   - Return `"SKIPPED: <reason>"` when disabled by config or unsupported on hardware; `HookManager` automatically records `[SKIPPED]` vs `[APPLIED]` vs `[FALLBACK]`.
   - Yield to explicit user code (`torch.is_autocast_enabled()`, pre-existing `torch.compile`, non-picklable datasets).
2. Register the pass cleanly in `register_builtin_passes(manager)` inside [`ptopt/passes/__init__.py`](../ptopt/passes/__init__.py).
3. Add unit tests under `tests/unit/` and subprocess end-to-end parity tests under `tests/migrate/`.

### 4.3 Adding a Custom Triton Kernel

1. Place your Triton kernel and PyTorch eager/autograd wrapper under `ptopt/kernels/my_kernel.py`:
   - Guard with `TRITON_AVAILABLE` and fall back gracefully to pure PyTorch (`fallback_on_error=True`).
   - Avoid `.item()` CPU-GPU synchronizations inside the kernel hot path.
2. Wire dispatch through [`ptopt/kernels/registry.py`](../ptopt/kernels/registry.py):
   - Use **Compile-Aware Gate**: skip custom autograd ops when `torch.compile` (`Inductor`) is already active so Inductor can fuse surrounding operations without graph breaks (unless your Triton kernel strictly beats Inductor, like `CrossEntropy`).
3. Verify numerical parity (`MaxRelDiff` / `MaxAbsDiff`), speed vs Eager & `torch.compile`, and checkpoint compatibility in `tests/unit/test_kernels_parity.py` and `tests/migrate/test_phase3_7_kernels.py`.
