# Real-World Benchmark Analysis: `pytorch-cifar-resnet18` (`kuangliu/pytorch-cifar`)

**Target**: `kuangliu/pytorch-cifar` (`ResNet18`, CIFAR-10, 10 epochs, `batch_size=128` train / `100` eval, wrapped in `nn.DataParallel`, `15,142,970` params)  
**Hardware / Stack**: Google Colab · **NVIDIA L4 (`sm_89`, 22,563.1 MB VRAM)** · PyTorch `2.14.1+cu130` · Triton `3.8.0` · CUDA `13.0`  
**Latest Verified Bundle**: [`174503_20261119-real-pytorch-cifar-resnet18`](../../benchmark_results/174503_20261119-real-pytorch-cifar-resnet18/summary.txt) (`main @ 3dfd990`, **Overall Status: `PASS` 5/5**)

---

## 1. Benchmark Progression (Before vs After Tuning)

| Run Bundle | Commit | Overall Status | `balanced/cold` Pure Wall | `balanced/warm` Pure Wall | `aggressive/cold` Pure Wall | `aggressive/warm` Pure Wall | Key Observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [`pytorch-cifar-resnet18-benchmark`](../../benchmark_results/pytorch-cifar-resnet18-benchmark/summary.txt) | `db40061` | `FAIL` (2/5) | `359.03s` (`0.88x`) | `233.37s` (`1.35x`) | `557.97s` (`0.56x`) | `228.65s` (`1.38x`) | Step 390 dynamic recompile (`82–95s`), `aggressive` wasted `271s` on `max-autotune`, non-deterministic `test_acc` false-fail |
| [`155258_20261119`](../../benchmark_results/155258_20261119-real-pytorch-cifar-resnet18/summary.txt) | `db40061` | `FAIL` (2/5) | `362.60s` (`0.87x`) | `236.46s` (`1.34x`) | `554.86s` (`0.57x`) | `228.39s` (`1.39x`) | Re-confirmed `db40061` baseline; even FP32 `safe` false-failed (`66.87%` vs `79.67%`) |
| [`174503_20261119`](../../benchmark_results/174503_20261119-real-pytorch-cifar-resnet18/summary.txt) | `3dfd990` *(incl. `1ebbea8`, `4451df2`)* | **`PASS` (5/5)** | **`313.56s` (`1.01x`)** | **`226.31s` (`1.40x`)** | **`297.82s` (`1.06x`)** | **`215.30s` (`1.47x`)** | `dl_remainder` pre-emption (`step390` `95s -> 0.41s`), `aggressive/cold` `-257s`, `tel_e2e` vs `pure_wall` gap `94s -> 6s` |

---

## 2. Detailed Results (`174503_20261119-real-pytorch-cifar-resnet18`)

Baseline (`--telemetry-only`): Train step median **`74.03 ms`**, Eval median **`13.04 ms`**, Pure Wall E2E **`316.48s`** (`tel_e2e=318.04s`), Peak Alloc VRAM **`2926.2 MB`** (`nvsmi=2160.0 MB`), Avg GPU Util **`89.4%`**.

| Arm | Cache | Train Step (ms) | Step Spd | Eval (ms / Spd) | Pure Wall E2E | Tel E2E | Alloc VRAM (Delta) | NVSMI Peak | Avg GPU Util | `test_acc` (Tel / Wall) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **safe** | none | `70.09` | `1.06x` | `13.26` (`0.98x`) | `302.34s` (`1.05x`) | `303.49s` | `2925.7` (`-0.5 MB`) | `3202 MB` | `91.9%` | `70.28%` / `76.16%` | **PASS** |
| **balanced** | cold | `41.75` | `1.77x` | `8.29` (`1.57x`) | `313.56s` (`1.01x`) | `318.77s` | `760.6` (`-2165.6 MB`) | `1068 MB` | `51.3%` | `70.37%` / `76.67%` | **PASS** |
| **balanced** | warm | `42.03` | `1.76x` | `8.41` (`1.55x`) | **`226.31s` (`1.40x`)** | `233.31s` | `760.6` (`-2165.6 MB`) | `1068 MB` | `66.6%` | `76.76%` / `73.59%` | **PASS** |
| **aggressive** | cold | `41.63` | `1.78x` | `7.20` (`1.81x`) | `297.82s` (`1.06x`) | `301.55s` | `760.6` (`-2165.6 MB`) | `1210 MB` | `53.1%` | `67.72%` / `74.92%` | **PASS** |
| **aggressive** | warm | **`41.03`** | **`1.80x`** | **`6.85` (`1.90x`)** | **`215.30s` (`1.47x`)** | `221.13s` | `760.6` (`-2165.6 MB`) | `1170 MB` | `68.3%` | `73.15%` / `75.60%` | **PASS** |

### Phase Breakdown & Dynamo Diagnostics (`174503_20261119`)
- **Steady Phase Breakdown** (`baseline` `79.96 ms` -> `aggressive/warm` `49.62 ms`):
  - `Forward`: `23.91 ms (30%)` -> **`18.12 ms (36%)`**
  - `Backward`: `47.06 ms (59%)` -> **`22.92 ms (46%)`**
  - `Optim`: `4.02 ms (5%)` -> **`1.89 ms (4%)`** (via `optimizer.SGD: fused=True`)
  - `Data/Other`: `3.39 ms (4%)` -> `3.16 ms (6%)`
- **Dynamo / Cache**:
  - `Frames(OK/Tot)`: `27/29`, `Unique Graphs`: `2`, `Graph Breaks`: `4`, `Recompiles`: `0`
  - `Inductor Cache`: `3/0` hit/miss (warm), `AOT Cache`: `2/2` (`balanced/warm`), `2/1` (`aggressive/warm`)
  - Step 390 (batch เศษขนาด `80`): ลดจาก `82–95s` เหลือ **`411–422 ms`** (DataLoader epoch boundary เท่านั้น ไม่มีการ recompile)

---

## 3. สิ่งที่ปรับแล้วและได้ผลจริงในรอบ `174503_20261119`

1. **DataLoader Remainder Pre-emption (`1ebbea8`)**:
   - [`dataloader_tune.py`](../../ptopt/passes/dataloader_tune.py) ตรวจจับ `drop_last=False` ที่ `50000 % 128 = 80 != 0` แล้วแจ้ง [`auto_compile.py`](../../ptopt/passes/auto_compile.py) ให้ compile ครั้งเดียวตั้งแต่ Step 0 ด้วย `dynamic=True, mode="default"` (`Compiled(dynamic,dl_remainder:dataset=50000, batch_size=128)`)
   - ตัดปัญหา `aggressive/cold` compile `max-autotune` ทิ้งเปล่า (`266s -> 95s`) และตัด dynamic-shape recompile ที่ step 390 (`82–95s -> 0.41s`) ทำให้ **Cold E2E เร็วกว่า Baseline ตั้งแต่รันครั้งแรก (`1.01x–1.06x`)**
2. **Telemetry Dynamo Short-Circuit (`1ebbea8`)**:
   - [`telemetry_hook.py`](../../ptopt/hooks/telemetry_hook.py) และ [`module_dispatch.py`](../../ptopt/hooks/module_dispatch.py) เช็ก `torch.compiler.is_compiling()` ทำให้รอบ `telemetry` กับ `pure_wall` ได้ FX/AOT graph ตรงกัน ลดช่องว่าง `tel_e2e` vs `pure_wall` บน warm cache จาก `~76–94s` เหลือ **`~5.8–7.0s`**
3. **Multi-Pass Variance Parity (`4451df2`)**:
   - [`real_bench.py`](../../pipline-for-test/real_bench.py) ดึง `pure_wall_metrics` มาเช็ก variance ของ baseline (`67.33%..78.33%`) ร่วมกับรอบ telemetry ทำให้เคส non-deterministic 10 epochs ไม่ติด false-fail

---

## 4. จุดที่ยังปรับต่อได้เพื่อรีดความเร็วเพิ่ม ("ปรับแล้วจะดีขึ้นอีก")

ข้อมูลเชิงลึกจาก [`summary.txt`](../../benchmark_results/174503_20261119-real-pytorch-cifar-resnet18/summary.txt) และ `diag_stderr.log` / `debug_profile.json` ชี้คอขวดที่เหลืออยู่ 5 จุดเรียงตาม impact:

### 4.1 Unwrap `nn.DataParallel` บน Single-GPU (Impact: **สูงมาก**)
- **อาการที่พบ**:
  - แม้ Colab มี L4 เพียง 1 ใบ โค้ด `pytorch-cifar/main.py` ห่อ `ResNet18` ด้วย `nn.DataParallel(net)` ทำให้เกิด **Graph Break ทั้งหมด 4 จุด** (`_get_device_index` ใน `torch/nn/parallel/_functions.py:129` จำนวน 2 ครั้ง + `Attempt to trace generator` `unimplemented` จำนวน 2 ครั้ง)
  - กราฟถูกหั่นเป็น `2` unique graphs, Baseline เองเสียเวลา CUDA ไปกับ `DataParallel.forward` ถึง **`12.9%`** (`58.3 ms` ใน 3 profiled steps) และ CPU **`7.9%`**, รวมถึงทำให้ `AOTAutograd` cache miss (`2/2` บน `balanced/warm`, `2/1` บน `aggressive/warm`)
- **แนวทางแก้ใน `ptopt`**:
  - เพิ่ม guard/pass สำหรับ `nn.DataParallel` เมื่อ `torch.cuda.device_count() == 1` (หรือ `len(device_ids) <= 1` และไม่มี hook/attribute เฉพาะ) ให้ unwrap/bypass `scatter`/`replicate`/`parallel_apply`/`gather` ไปเรียก `mod_self.module` บน device 0 โดยตรง (พร้อมรักษา `.state_dict()` / `module.` prefix compat ผ่าน `checkpoint_compat.py`)
- **ผลที่คาดหวัง**:
  - Graph breaks: `4 -> 0` (รวมเป็น `1` graph เต็มตั้งแต่ต้นจนจบ)
  - ตัด `DataParallel.forward` + Python dispatch overhead
  - ลด parameter count ที่เห็นตรงจุด compile จาก `15.14M` (หรือเปิดทางให้ `channels_last` / `train_cuda_graphs` ทำงานบนโมเดลจริงได้ง่ายขึ้น)

### 4.2 เปิด `channels_last` สำหรับ Small-Spatial High-Channel `bf16` Conv2d (Impact: **กลาง-สูง**)
- **อาการที่พบ**:
  - `runtime.channels_last` ถูกข้ามด้วยเหตุผล `Conv spatial/volume input below auto threshold (16384 px)` เพราะภาพ CIFAR-10 ขนาด `32x32` (`1,024 px`)
  - แต่ใน `debug_profile.json` พบ kernel **`cudnn::engines_precompiled::nchwToNhwcKernel` ถูกรันถึง 525 ครั้ง** (`3.8%` ของ CUDA time หรือ ~`12.1 ms` ต่อ 3 steps) เพราะ Tensor Core `bf16` บน `sm_89` (L4) บังคับแปลง layout NCHW ↔ NHWC ทุกรอบ Conv2d (ResNet18 มี 64–512 channels แม้ spatial เล็กลงเหลือ 4×4)
  - นอกจากนี้ใน `diag_stderr.log` ยังพบ `Using FallbackKernel: torch.ops.aten.convolution_backward.default` ซึ่งกิน CUDA time สูงถึง **`13.7%`** (`43.1 ms` ต่อ 3 steps)
- **แนวทางแก้ใน `ptopt`**:
  - ทดสอบ A/B เพิ่มเกณฑ์ heuristic ของ `channels_last` เช่น อนุญาตเมื่อ `bf16`/`fp16` Tensor Core active และโมเดลมี Conv2d `out_channels >= 64` ลึก >= 8 layers (หรือตั้ง opt-in env/flag เช่น `PTOPT_CHANNELS_LAST_MIN_PX`) และตรวจสอบว่า `nchwToNhwcKernel` + `convolution_backward` ลดลงจริงหรือไม่

### 4.3 อัด GPU Utilization จาก `66.6%–68.3%` ขึ้นใกล้ `90%` (Impact: **กลาง-สูง**)
- **อาการที่พบ**:
  - Baseline ใช้ GPU เฉลี่ย **`89.4%`** (เพราะ compute นานกว่า CPU), แต่หลัง `ptopt` เร่ง compute ลง ~`1.8x` แล้ว GPU Utilization ลดลงเหลือ **`66.6%` (`balanced/warm`)** และ **`68.3%` (`aggressive/warm`)** (ใน cold เหลือ `51.3%–53.1%`)
  - ชี้ว่า workload เริ่มเข้าสู่ **CPU / DataLoader / Kernel-Launch bound** (`cuLaunchKernel` + `cudaLaunchKernel` รวม `~10%` CPU time, `Data/Other` `~6–7%`, และทุกสิ้น epoch มี spike `~400–422 ms` จาก `DataLoader` worker restart)
- **แนวทางแก้ที่ทดสอบได้ทันที**:
  1. รัน `--ptopt-arg=--bandwidth-boost` (Phase 8): เปิด `dataloader.cuda_prefetch=True` (VRAM ว่างอยู่ถึง **`21.7 GB` / 96% free**), ดัน `prefetch_factor=4`, และขยับ `num_workers` จาก `2` (Colab มี `12` logical cores / `6` physical cores)
  2. หากแก้ single-GPU `DataParallel` + ใช้คู่กับ workload ที่มี static batch (`drop_last=True` หรือ bucketed padding) จะเปิด `compile.train_cuda_graphs` ได้ ลด CPU launch overhead ที่ทำให้ GPU util ต่ำ

### 4.4 แยก `eval` Compiled Handle & ลด `grad_mode` Recompile (Impact: **กลาง**)
- **อาการที่พบ**:
  - ใน `balanced` (`inference.cuda_graphs=False`), `auto_compile.py` (`can_reuse_other`) นำ `_ptopt_train_compiled_impl` มาใช้ซ้ำกับ `eval`, ทำให้พอเข้า `with torch.no_grad():` เกิด log `[0/1] Recompiling function inner ... GLOBAL_STATE changed: grad_mode` และเป็นสาเหตุหนึ่งของ `Step0/eval0` overhead (`~26s` ใน warm telemetry) กับ `aot_cache_miss: 2`
- **แนวทางแก้ใน `ptopt`**:
  - ทดสอบแยก `torch.compile` handle / guard สำหรับ `eval` ออกจาก `train` หรือใช้ `torch._dynamo` inference guard isolation และหลีกเลี่ยง `Auto-AMP` wrapping ที่ทำให้เกิด `GLOBAL_STATE` ต่างกันโดยไม่จำเป็น

### 4.5 ใช้คำสั่ง Benchmark แบบล็อก Seed ตามคู่มือล่าสุด
- **อาการที่พบ**:
  - ใน `174503_20261119` ผล `test_acc` ยังแกว่งตาม random seed (`67.33%..78.33%` บน baseline) เพราะยังใช้ `--cmd` เก่าที่ไม่ได้ใส่ seed
- **แนวทาง**:
  - ใช้คำสั่งที่ล็อก `random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)` ตามที่อัปเดตไว้ใน [`REAL-WORLD-BENCHMARK.md`](../REAL-WORLD-BENCHMARK.md#L106-L117) เพื่อให้เทียบ `test_acc` และ `loss` ระหว่าง `baseline` กับทุก profile ได้แบบ deterministic 100%
