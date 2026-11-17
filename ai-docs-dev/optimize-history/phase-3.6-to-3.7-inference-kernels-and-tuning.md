# Optimization Analysis (Phase 3.6 – 3.7): Inference, Custom Triton Kernels และการจูนร่วมกับ TorchInductor

เอกสารนี้วิเคราะห์เชิงลึกจากโค้ดใน `src/passes/inference_opt.py`, `src/kernels/`, `src/passes/dataloader_tune.py`, `src/passes/auto_compile.py`, `src/passes/memory_allocator.py` และผลวัดจริงบน **Google Colab NVIDIA L4 (`sm_89`)** จากรายงาน:
- **Phase 3.6 (Inference)**: [152328 (รอบแรก)](../../report/152328_20261117-phase-3.6.txt) → [155721 (Multi-scale + M≥512 FP8 guard)](../../report/155721_20261117-phase-3.6.txt) → [172306 (Fused Triton FP8 Quantizer)](../../report/172306_20261117-phase-3.6.txt)
- **Phase 3.7 (Custom Triton Kernels)**: [160510 (รอบแรก)](../../report/160510_20261117-phase-3.7.txt) → [173112 (ตัด .item() sync + num_warps + Compile-Aware Gate)](../../report/173112_20261117-phase-3.7.txt)
- **ล่าสุด (`commit aa4d1c6`, รอผลรัน Colab)**: จุด A (`registry.py` หลีกทาง `RMSNorm` ใต้ `torch.compile`) + จุด B (`inference_opt.py` lazy FP8 weight cache เฉพาะ `m_rows >= 512`)

---

## ภาพรวม: ปัญหาใน Vanilla PyTorch และผลลัพธ์จาก Phase 3.6 – 3.7

| # | จุดช้า / คอขวดเดิม | Pass / Kernel ที่แก้ | กลไกที่เปลี่ยน | ผลวัดจริงบน NVIDIA L4 (`sm_89`) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `infer` `bs=1` ติด Python + CUDA kernel launch overhead หนัก | [`inference_opt.py`](../../src/passes/inference_opt.py) | Auto `torch.inference_mode()` + `mode="reduce-overhead"` (CUDA Graphs) + dynamic-shape fallback | `infer_bs1_small`: `1.53 ->` **`0.48 ms` (`3.19x`)**<br>`infer_bs1_medium`: **`2.82x`**, `bs1_large`: **`2.36x`** |
| 2 | Batch inference (`bs=64`) คำนวณเป็น FP32 SIMT และไม่ fuse ops | [`auto_amp.py`](../../src/passes/auto_amp.py), [`auto_compile.py`](../../src/passes/auto_compile.py), [`inference_opt.py`](../../src/passes/inference_opt.py) | `inference_mode()` + `BF16` AMP + `torch.compile` (Inductor + FlashAttention) | `infer_bs64_large`: `373.93 ->` **`90.22 ms` (`4.14x`)**<br>VRAM ลด **`-187.5 MB` (`766.9 -> 579.4 MB`)** |
| 3 | Eager Dynamic FP8 ช้า+เพี้ยนบน matrix เล็ก และกินเวลา quantize >50% | [`inference_opt.py`](../../src/passes/inference_opt.py) | (1) Guard `m_rows >= 512` & `m_rows % 16 == 0`<br>(2) Fused Triton 2-pass FP8 quant (`_fp8_partial_amax_kernel` + `_fp8_quant_scale_kernel`) + lazy weight cache | แก้ parity `0.035899 FAIL` → `0.0026–0.0069 PASS`<br>`bs64_large`: **`227.88 -> 149.98 ms` (`1.64x -> 2.49x`)**, VRAM ลด **`-185 MB`** |
| 4 | PyTorch Eager `RMSNorm` / `CrossEntropy` / `SwiGLU` ยิงหลาย kernel แยก & เปลือง HBM | [`src/kernels/`](../../src/kernels/) (`rmsnorm.py`, `cross_entropy.py`, `layernorm.py`, `rope_or_swiglu.py`) | Custom Triton Fused Kernels (Fused `RMSNorm`, Online-Softmax `CrossEntropy`, Fused `SwiGLU`) + Zero `.item()` GPU sync + `num_warps` tuning | **`RMSNorm [large]`**: `4.371 ->` **`0.636 ms` (`6.87x`)**, ชนะ Compile (`0.640 ms`)<br>**`CrossEntropy [large]`**: `11.94 ->` **`3.360 ms` (`3.55x`)**, ชนะ Compile (`3.372 ms`) |
| 5 | `autograd.Function` ของ Triton คั่นกลางกราฟ `torch.compile` | [`src/kernels/registry.py`](../../src/kernels/registry.py) | Compile-Aware Gate: ใช้ Triton ครบชุดใน Eager (`3.80x–4.11x`) และหลีกทางให้ Inductor fuse เมื่อ `compile=True` (คง `cross_entropy` ไว้) | `3.7a_kernels_eager` `medium`: **`3.80x` (`26.27 ms`)**, `large`: **`4.11x`**, `3.7c` ลด VRAM **`-4,309.2 MB`** *(จูนไม่ให้เกิด 25 RMSNorm graph breaks เพิ่มใน `aa4d1c6`)* |
| 6 | Host copy & Cold-start compile | [`dataloader_tune.py`](../../src/passes/dataloader_tune.py), [`memory_allocator.py`](../../src/passes/memory_allocator.py), [`auto_compile.py`](../../src/passes/auto_compile.py) | Auto-promote pinned `.to(cuda)` / `.cuda()` เป็น `non_blocking=True` + Persistent `TRITON_CACHE_DIR` & `TORCHINDUCTOR_AUTOGRAD_CACHE=1` | ลด blocking H2D ใน DataLoader pipeline + แชร์ cache ข้าม run โดยไม่แตะ `state_dict()` / parameters |

---

## เจาะลึก 1: Inference Optimization (Phase 3.6)

### 1.1 Latency (`bs=1`) vs Throughput (`bs=64`): `LAUNCH_BOUND` กับ `COMPUTE_BOUND`
จาก Deep Debug Profiler ใน [172306](../../report/172306_20261117-phase-3.6.txt):
- **เคส `bs=1` (`infer_bs1_small | baseline_dbg`)**:
  - เวลาที่ GPU คำนวณจริงใน Profiler window `[4, 9]` (`self_cuda`) มีแค่ **`2.446 ms`** แต่ CPU เสียเวลาจัดคิวและยิง kernel (`self_cpu`) ถึง **`14.851 ms`** (`Host/Data = 69.4%`)
  - การเปิดแค่ `3.6b_infer_amp_compile` (mode=`default`) บน `bs1_small` ได้เพียง **`0.97x` (`1.58 ms`)** เพราะ CPU ยังต้องยิง Triton/CUTLASS kernel ทีละตัว
  - แต่เมื่อเปิด **`3.6c_cuda_graphs` (`mode="reduce-overhead"`)**: GPU บันทึกชุดคำสั่งทั้งหมดเป็น **CUDA Graph** แล้วให้ CPU สั่ง replay ครั้งเดียวต่อ step → `self_cpu` ลดจาก **`14.851 -> 6.696 ms`**, `Med` เหลือ **`0.48 ms` (`3.19x`)**, Throughput พุ่ง `652.7 -> 2,079.0 samp/s`
- **เคส `bs=64` (`infer_bs64_large`)**:
  - งานย้ายไปเป็น Compute-bound เต็มตัว (`Model Fwd = 95.5%`, Baseline GPU time `3,692.2 ms` ด้วย FP32 `ampere_sgemm` / `aten::addmm`)
  - **`3.6b_infer_amp_compile`** (`BF16 + Inductor`) ยุบลงเหลือ **`90.22 ms` (`4.14x`, `709.3 samp/s`)** และลด VRAM **`-187.5 MB` (`766.9 -> 579.4 MB`)**

### 1.2 วิวัฒนาการของ `3.6d_fp8_sm89`: จากช้า+เพี้ยน (`0.35x` FAIL) สู่ Fused Triton FP8 (`2.49x` PASS)
1. **ปัญหาแรกที่พบบน L4 (`152328`)**:
   - บน `infer_bs1_small` (`M = 1*64 = 64`), Eager FP8 `_scaled_mm` ทำได้ **`0.35x` (`4.26 ms` vs `1.51 ms`)** และ `MaxRelDiff = 0.035899` (**FAIL** เกณฑ์ `0.03`) เพราะ tensor เล็กมากทำให้ overhead การ quantize แพงกว่า BF16 และ E4M3 สูญเสียความแม่นยำกับ activation ช่วงแคบ
   - **วิธีแก้**: เพิ่ม guard `m_rows < 512 or m_rows % 16 != 0` ให้ fallback กลับไปใช้ Linear ปกติ → รอบ [155721](../../report/155721_20261117-phase-3.6.txt) **PASS ครบทุกขนาด (`MaxRelDiff = 0.0026–0.0068`)**
2. **คอขวดที่เจอจาก Profiler รอบ `155721` ของ `infer_bs64_large | fp8_sm89_dbg`**:
   - เวลา GPU ครึ่งหนึ่งหายไปกับ **dynamic activation quantization ใน PyTorch Eager** (5–6 kernels + สร้าง `[M, K]` FP32 buffer ใหม่ทุก `nn.Linear`):
     - `aten::copy_` `269.78 ms (11.4%)`
     - `aten::abs` `131.96 ms (5.6%, total 263.92 ms)`
     - `aten::clamp` `132.64 ms (5.6%)`
     - `aten::div` `118.19 ms (5.0%)`
   - และจอง temporary allocations ส่งผลให้ VRAM พุ่ง **`+390.6 MB` (`766.9 -> 1,157.5 MB`)**
3. **วิธีแก้ใน [`inference_opt.py`](../../src/passes/inference_opt.py) (`commit 38810b4` + `aa4d1c6`)**:
   - เขียน Fused Triton 2-pass quantizer (`_get_triton_fp8_quantizer`): `_fp8_partial_amax_kernel` หา block max แทบไม่ใช้ HBM + `_fp8_quant_scale_kernel` คูณ `inv_scale`, clamp `[-448, 448]`, และ cast เป็น `float8_e4m3fn` ใน kernel เดียว (zero `.item()` CPU sync!)
   - Cache weight (`w_t`, `w_scale`, `bias_bf16`/`bias_fp16`) ใน `lin._ptopt_fp8_w_cache` ผูกกับ `lin.weight._version` **โดยไม่แตะ `lin.weight` / `state_dict()`** และปรับเป็นสร้างแบบ **lazy เฉพาะตอนผ่าน `m_rows >= 512` (`aa4d1c6`)** เพื่อไม่ให้ `bs=1` เสีย VRAM เปล่า (`+63.8 MB` → คืนเป็น `+0 MB`)
   - **ผลลัพธ์ใน [172306](../../report/172306_20261117-phase-3.6.txt)**:
     - GPU time รวม (`self_cuda`) ลดจาก **`2,358.19 -> 1,439.07 ms` (-39%)**
     - `abs`/`clamp`/`div` หายไปจาก Top Ops เกลี้ยง เหลือ `_fp8_quant_scale_kernel 69.86 ms (4.9%)` และ `aten::_scaled_mm 82.41 ms`
     - `infer_bs64_large` เร็วขึ้น **`227.88 -> 149.98 ms` (`1.64x -> 2.49x`)**, VRAM ลดจาก `1,157.5 -> 972.5 MB` (**`-185.0 MB`**); `infer_bs64_medium` เร็วขึ้น **`30.41 -> 20.14 ms` (`1.47x -> 2.20x`)**

---

## เจาะลึก 2: Custom Triton Kernels & Compile Cooperation (Phase 3.7)

### 2.1 เหตุใด Micro-benchmark ของ `CrossEntropy` และ `RMSNorm` ถึงชนะ `torch.compile`
เทียบผลระหว่าง [160510](../../report/160510_20261117-phase-3.7.txt) → [173112](../../report/173112_20261117-phase-3.7.txt):
1. **`triton_cross_entropy` ([`cross_entropy.py`](../../src/kernels/cross_entropy.py))**:
   - **คอขวดเดิม (`160510`)**: มี `valid_count = int(valid_mask.sum().item())` ซึ่งบังคับ **CUDA Stream Synchronization (`CPU wait GPU`)** ทุกครั้งที่คำนวณ loss! ทำให้ `CrossEntropy [large] (8192x8192)` อยู่ที่ `3.751 ms` (แพ้ `torch.compile` `3.374 ms`)
   - **การแก้ (`38810b4`)**: ย้าย `valid_count_t = valid_mask.sum().to(torch.float32).clamp(min=1.0)` เป็น on-device scalar tensor ล้วนทั้ง forward/backward + เพิ่ม `num_warps = 8` → **`CrossEntropy [large]` ลงมาที่ `3.360 ms (74.4 GB/s)` แซง `torch.compile` (`3.372 ms`) และเร็วกว่า Eager (`11.94 ms`) ถึง `3.55x`!**
2. **`triton_rmsnorm` ([`rmsnorm.py`](../../src/kernels/rmsnorm.py))**:
   - Eager PyTorch ทำ `pow -> mean -> rsqrt -> mul -> mul` หลาย kernel อ่าน-เขียน HBM ซ้ำหลายรอบ (`4.371 ms` บน `8192x1025`)
   - Triton kernel อ่านแถวเดียวเข้า register/SRAM คำนวณ `mean_sq`, `rsqrt`, คูณ `w` รวดเดียว + จูน `_select_num_warps` (`4/8/16`) และขยาย backward workers `min(M, 256)` → ได้ **`0.577 ms` (`medium`)** และ **`0.636 ms / 98.4 GB/s` (`large`, ชนะ Eager `6.87x` และชนะ `torch.compile` `0.640 ms`)**

### 2.2 ปัญหาสำคัญที่เจอเมื่อใช้ `autograd.Function` คู่กับ `torch.compile` (`4.07x` vs `4.37x`)
- ใน [160510](../../report/160510_20261117-phase-3.7.txt) และ [173112](../../report/173112_20261117-phase-3.7.txt) แม้ `CrossEntropy` และ `RMSNorm` เดี่ยวๆ จะเร็วกว่า `torch.compile` แล้ว แต่ E2E `transformer:large` ของ `3.7c_aggressive_full` (`195.68 ms / 4.07x`) ยังช้ากว่า `3.7b_compile_only` (`182.03 ms / 4.37x`) ~7.5%
- **หลักฐานจาก Hook Logs (`173112` บรรทัด 251–258)**:
  - `3.7a_kernels_eager` (`compile=False`) ได้ `193.65 ms (4.11x)`
  - `3.7c_aggressive_full` patch `RMSNorm` ไว้ **25 จุด** ด้วย custom `autograd.Function` แล้วค่อยสั่ง `torch.compile` → Dynamo/Inductor ไม่สามารถ trace ทะลุ `autograd.Function` เพื่อ fuse `Residual Add + RMSNorm` ได้ จึงเกิด **Graph Break 25 จุด** (1 จุดต่อ norm layer) ทำให้ได้ความเร็วพอกับโหมด eager แถมได้ประโยชน์จาก compile ไม่เต็มที่
- **การแก้ใน [`src/kernels/registry.py`](../../src/kernels/registry.py) (`commit aa4d1c6`)**:
  - **โหมด Eager (`compile.enabled=False`)**: เปิด Triton kernels ทั้ง 4 ตัว (`rmsnorm`, `layernorm`, `cross_entropy`, `swiglu`) เพื่อได้ความเร็ว **`3.80x–4.11x`** โดย **ไม่ต้องรอ compile warmup `~8.8s`**
  - **โหมด Compiled (`compile.enabled=True`)**: หลีกทางให้ Inductor จัดการ internal module ops (`rmsnorm`, `layernorm`, `swiglu`) เพื่อ **0 graph breaks** และคง `cross_entropy` (ซึ่งอยู่นอก `model.forward` และมี shape gate `numel >= 8_388_608 or V >= 4096`) ไว้ทำงานร่วมกันอย่างลงตัว

---

## เจาะลึก 3: Host, Transfer & Persistent Cache (`dataloader_tune.py`, `memory_allocator.py`, `auto_compile.py`)

1. **Auto Async H2D สำหรับ Pinned Batches ([`dataloader_tune.py`](../../src/passes/dataloader_tune.py))**:
   - โค้ดเทรนทั่วไปมักเขียน `images = images.to(device)` หรือ `.cuda()` โดยไม่ได้ส่ง `non_blocking=True` ทำให้ host ต้องรอ DMA Copy เสร็จทั้งที่ `DataLoader` เปิด `pin_memory=True` ไว้แล้ว
   - เพิ่ม wrapper บน `Tensor.to` / `Tensor.cuda` (เฉพาะเมื่อ CUDA พร้อม, `pin_memory=True`, `not torch.compiler.is_compiling()`, เป็น `cpu` tensor ที่ `is_pinned() == True`, และผู้ใช้ไม่ได้ส่ง `non_blocking=...` เอง) → promote เป็น `non_blocking=True` อัตโนมัติ โดยไม่กระทบพารามิเตอร์หรือ internal CUDA ops ของโมเดล
2. **Persistent Cross-Run Cache ([`memory_allocator.py`](../../src/passes/memory_allocator.py), [`auto_compile.py`](../../src/passes/auto_compile.py))**:
   - ตั้ง `TRITON_CACHE_DIR` (`<cache_dir>/triton`) และ `TORCHINDUCTOR_AUTOGRAD_CACHE=1` คู่กับ `TORCHINDUCTOR_CACHE_DIR` และ `TORCHINDUCTOR_FX_GRAPH_CACHE=1` ตั้งแต่ pre-launch (`setdefault` ไม่ทับค่า user) เพื่อให้ทั้ง Inductor graph, AOTAutograd และ Triton custom kernels แชร์ cache ข้ามการรันได้

---

## บทเรียนเชิงสถาปัตยกรรม (Key Takeaways)
1. **Fusion Boundaries สำคัญกว่า Micro-benchmark เดี่ยวๆ**: Custom Triton op ที่เร็วกว่าในระดับ standalone ถ้าห่อด้วย `autograd.Function` แล้วไปอยู่กลาง `torch.compile` จะหักกราฟ Inductor และช้าลง E2E (~7.5%) — ต้องแยกใช้ชัดเจนระหว่าง Eager (`3.7a`) กับ Full-Graph Compile (`3.7b/c`) หรือห่อด้วย `torch.library.custom_op`
2. **อย่าปล่อย `.item()` หรือ intermediate allocations ไว้ใน Fast Path**: แค่ตัด `valid_mask.sum().item()` ใน `CrossEntropy` และยุบ 4 Eager elementwise kernels (`abs`/`div`/`clamp`/`copy_`) ใน `FP8` เข้า Triton ช่วยลดเวลา GPU ได้ **`39%`** และแซง `torch.compile` บน matrix ใหญ่ได้จริง
3. **Zero-Code Parameter Safety**: ทุก optimization ใน 3.6–3.7 (CUDA Graphs, FP8 pre-quant cache, Triton forward patch, `non_blocking` pinned transfer) ทำงานที่ **runtime execution / forward hook layer เท่านั้น** ไม่เคยเปลี่ยน `nn.Parameter` dtype, ชื่อคีย์ใน `state_dict()`, หรือความสามารถในการ save/load ข้ามระหว่าง baseline กับ `ptopt`
