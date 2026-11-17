# Optimization Analysis (Phase 3.1 – 3.5): ทำไม PyTorch เร็วขึ้นและประหยัด VRAM ลง?

เอกสารนี้วิเคราะห์จากโค้ดปัจจุบันใน `src/passes/`, `src/hooks/` และผลวัดจริงบน **Google Colab NVIDIA L4 (`sm_89`)** จาก [104602 (3.1)](../../report/104602_20261117-phase-3.1.txt), [105422 (3.2)](../../report/105422_20261117-phase-3.2.txt), [112720 (3.3)](../../report/112720_20261117-phase-3.3.txt), [132418 (3.4 before)](../../report/132418_20261117-phase-3.4.txt), [141819 (3.4 tuned)](../../report/141819_20261117-phase-3.4.txt) และ [144108 (3.5 compile)](../../report/144108_20261117-phase-3.5.txt)

---

## ภาพรวม: Vanilla PyTorch ช้าที่จุดไหน และ `ptopt` แก้แต่ละจุดอย่างไร

| # | จุดช้า / คอขวดของ PyTorch เดิม | Pass & ไฟล์ที่เข้าไปแก้ | กลไกที่เปลี่ยน | ผลวัดจริงบน NVIDIA L4 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | ใช้ FP32 SIMT Cores ไม่ใช้ Tensor Cores และไม่เปิด Flash Attention SDPA | [`runtime_flags.py`](../../src/passes/runtime_flags.py), [`auto_amp.py`](../../src/passes/auto_amp.py) | เปิด `TF32` (`sm>=80`) + Flash/Mem-Efficient SDPA + `BF16` Auto-AMP (`FP16+GradScaler` บน `sm<80`) | `transformer_lm:large` `777.6 -> 428.9 ms` (`1.81x` TF32) -> **`252.6 ms` (`3.08x` +BF16)**, VRAM **`-2,848.9 MB`** |
| 2 | Optimizer รันแยกลูป/หลาย kernel ต่อ tensor (`foreach` off/ไม่ fused) | [`optimizer_fusion.py`](../../src/passes/optimizer_fusion.py) | ฉีด `fused=True` (fallback `foreach=True`) เข้า `Adam`/`AdamW`/`SGD.__init__` | `deep_mlp:large` เวลาช่วง `Optim` ลดจาก **`35.02 ms (66.1%) -> 12.80 ms (47.4%)`** (`2.28x` total) |
| 3 | `DataLoader(num_workers=0, pin_memory=False)` บล็อก GPU thread | [`dataloader_tune.py`](../../src/passes/dataloader_tune.py) | ปรับ `num_workers=2..4`, `pin_memory=True`, `persistent_workers=True` พร้อม `/dev/shm` & pickle guard | `cnn_vision:large` ช่วง `Data/Other` ลดจาก **`16.77 ms -> 2.40 ms`** (~7x เร็วขึ้น) |
| 4 | Layout `NCHW` บังคับให้ cuDNN เสียเวลา Transpose `NCHW ↔ NHWC` ทุก Conv | [`channels_last.py`](../../src/passes/channels_last.py), [`checkpoint_compat.py`](../../src/hooks/checkpoint_compat.py) | แปลงน้ำหนัก + input 4D เป็น `channels_last` อัตโนมัติเมื่อ `Conv2d` ขนาด `>=128x128` บน CUDA | ล้าง `nchwToNhwc`/`nhwcToNchw` (`112.6 ms` / `13.7%`) ออกหมด -> `cnn_vision:medium` **`3.04x`**, `large` `2.00x ->` **`2.59x`** |
| 5 | Python dispatch, cast-churn และ elementwise kernels ที่ไม่ถูก fuse | [`auto_amp.py`](../../src/passes/auto_amp.py), [`auto_compile.py`](../../src/passes/auto_compile.py) | (ก) ข้าม eager AMP ใน MLP ที่มี `LayerNorm>=6` (ข) ใช้ `torch.compile` in-place + cache | `transformer_lm:small` พลิกจาก `0.96x ->` **`2.08x`**; `transformer_lm:large` พุ่งเป็น **`4.36x`**, VRAM **`-4,167.4 MB` (-45.4%)** |
| 6 | Memory segment fragmentation และ overhead thread CPU | [`memory_allocator.py`](../../src/passes/memory_allocator.py) | `expandable_segments:True`, `omp_threads`, `TORCHINDUCTOR_CACHE_DIR` + `FX_GRAPH_CACHE=1` | `num_alloc_retries=0`, Warm Step0 compile startup ลด **3.4x–4.0x** (`24.9s -> 6.2s`) |

---

## เจาะลึกทีละกลไก: ทำไมโค้ดถึงทำให้เร็วขึ้นและลด VRAM

### 1. ย้ายงานคำนวณจาก FP32 SIMT ไป Tensor Cores (`runtime_flags.py` + `auto_amp.py`)
- **ปัญหาใน PyTorch ปกติ**:
  - ค่า default ของ PyTorch ปิด `allow_tf32` และรัน `Linear`/`Conv2d` เป็น `FP32` (`SIMT`) บน L4 (`sm_89`) แม้การ์ดจะมี Tensor Cores ที่แรงกว่ามากสำหรับ `TF32` และ `BF16`/`FP16`
  - หลักฐานจาก `torch.profiler` ใน [132418](../../report/132418_20261117-phase-3.4.txt): Baseline `transformer_lm:large` รัน `ampere_sgemm_128x64_tn`, `cutlass_80_simt_sgemm` และ `fmha_cutlassB_f32` กินเวลา CUDA รวม `4,812.5 ms` ใน 3 steps
- **สิ่งที่โค้ดเราแก้**:
  - `apply_runtime_flags_pass` เปิด `allow_tf32=True`, `set_float32_matmul_precision("high")` เฉพาะ `sm >= 80` (ป้องกัน T4 `sm_75`) และเปิด `flash_sdp` / `mem_efficient_sdp`
  - `install_auto_amp_hook` ครอบเฉพาะ top-level `nn.Module.__call__` ด้วย `torch.amp.autocast(device_type="cuda", dtype=bf16|fp16)` พร้อม **Functional Loss FP32 Guard** (`_install_functional_loss_fp32_guards`) ที่ promote `logits` ของ `F.cross_entropy`, `nll_loss`, `binary_cross_entropy_with_logits`, `kl_div` กลับเป็น `fp32` เมื่อถูกเรียกนอก `autocast` (แก้ `NaN` ของ `fp16 + GradScaler` ใน `cnn_vision` ได้เด็ดขาด)
  - หลักฐานจาก profiler ใน [141819](../../report/141819_20261117-phase-3.4.txt): Kernel เปลี่ยนเป็น `ampere_bf16_s1688gemm` / `s16816gemm` (Tensor Core) เวลารวม 3 steps ลดจาก `4,812.5 ms -> 1,533.9 ms` และขนาด Activation Tensor ลดครึ่งหนึ่ง (`4 bytes -> 2 bytes`) ช่วยลด Peak VRAM ได้ **31%–48.6%** ทันที

### 2. ลดเวลา Kernel Launch ของ Optimizer จาก 66% เหลือ Kernel เดียว (`optimizer_fusion.py`)
- **ปัญหาใน PyTorch ปกติ**:
  - โมเดลที่มีหลายชั้น (เช่น `deep_mlp` 24 ชั้น = ~100 tensors) เมื่อใช้ `AdamW` ปกติจะยิง multi-tensor/per-tensor ops หลายรอบต่อ 1 step (`_foreach_mul_`, `_foreach_addcdiv_`, `_foreach_lerp_` กิน `102.4 ms` ใน 3 steps บน `deep_mlp:large`) คิดเป็น **`66.1%` ของเวลาทั้ง step** (`35.02 / 53.00 ms`)
- **สิ่งที่โค้ดเราแก้**:
  - `install_optimizer_fusion_hook` ดัก `__init__` ของ `Adam`/`AdamW`/`SGD`, แปลง generator params อย่างปลอดภัยครั้งเดียว (`_materialize_param_groups`), ตรวจว่าทุก tensor เป็น CUDA floating-point ผ่าน `check_fused_eligibility`, แล้วฉีด `fused=True` (หรือ fallback `foreach=True`)
  - ใน [141819](../../report/141819_20261117-phase-3.4.txt): `AdamW` ถูกรวมเป็น single kernel `aten::_fused_adamw_` เวลาช่วง `Optim` ของ `deep_mlp:large` ลดจาก **`35.02 ms -> 12.80 ms`** (และ `deep_mlp:small` จาก `2.48 ms -> 0.63 ms`)

### 3. ปลดบล็อก CPU Augmentation & Pageable Copy ใน DataLoader (`dataloader_tune.py`)
- **ปัญหาใน PyTorch ปกติ**:
  - ผู้ใช้มักไม่ตั้ง `num_workers` (`0`) และ `pin_memory` (`False`) ทำให้ GPU ต้องหยุดรอ CPU ทำ crop/normalize (`SyntheticVisionDataset`) และคัดลอกผ่าน pageable memory
- **สิ่งที่โค้ดเราแก้**:
  - `decide_dataloader_params` ตรวจ `has_cuda_tensors`, เช็ก `/dev/shm` (`<1024 MB` ปิด workers, `<2048 MB` cap ที่ `2`), และเช็ก `is_picklable_fast` โดยไม่ serialize tensor buffer ก่อนปรับเป็น `num_workers=3, pin_memory=True, persistent_workers=True`
  - ใน [141819](../../report/141819_20261117-phase-3.4.txt): ช่วง `Data/Other` ของ `cnn_vision:large` ลดจาก **`16.77 ms -> 2.40 ms`** และ `cnn_vision:small` ลดจาก **`3.73 ms (25.5%) -> 0.74 ms (7.2%)`** พร้อม fallback `num_workers=0` อัตโนมัติสำหรับ unpicklable dataset (`_LambdaDataset`)

### 4. กำจัด Layout Transpose ฟรี ๆ `NCHW ↔ NHWC` ด้วย Auto `channels_last` (`channels_last.py`)
- **ปัญหาที่ตรวจพบจาก Profiler (`132418`)**:
  - PyTorch เก็บ weight และ input 4D แบบ `NCHW` แต่ cuDNN Tensor Core Conv2d ต้องการ `NHWC` ทำให้ cuDNN ต้องแทรก `nchwToNhwcKernel` (`74.78 ms`) + `nhwcToNchwKernel` (`37.83 ms`) รวม **`112.6 ms` (`13.7%` ของ CUDA time ทั้งหมด)** ใน `cnn_vision:large`!
- **สิ่งที่โค้ดเราแก้**:
  - เพิ่ม auto spatial heuristic ใน `apply_channels_last_pass` + `install_channels_last_hook`: เมื่อเป็น `Conv2d` บน CUDA และ 4D input มีพื้นที่ `H*W >= 128*128` (`16,384` px) จะแปลง model weights และ input pre-hook เป็น `torch.channels_last` อัตโนมัติ (ข้าม CNN เล็ก `64x64` เพื่อไม่ให้เพิ่ม overhead เกินจำเป็น) และ [checkpoint_compat.py](../../src/hooks/checkpoint_compat.py) คืน layout `contiguous()` ตอน `state_dict()`
  - ผลเทียบ `132418 -> 141819`: `nchwToNhwc`/`nhwcToNchw` หายจาก Top 8 CUDA Ops ทั้งหมด -> `cnn_vision:large` เร็วขึ้นต่อจาก `140.50 ms (2.00x)` เป็น **`107.92 ms (2.59x)`** และ `medium` จาก `28.40 ms (2.31x)` เป็น **`21.66 ms (3.04x)`**

### 5. แก้จุดตายของ Eager AMP (`cast churn`) และรวม Kernel ด้วย `torch.compile` (`auto_amp.py` + `auto_compile.py`)
- **ทำไม Eager AMP อย่างเดียวถึงแพ้ในบางโมเดล (หลักฐานจาก `132418`)**:
  1. **Cast-churn ใน Deep LayerNorm MLP**: `Linear` รัน `bf16` แต่ PyTorch autocast บังคับ `LayerNorm` รัน `fp32` ทำให้ `DeepNarrowMLP` 24 ชั้นเกิด `aten::copy_` ถึง **609 ครั้งต่อ 3 steps (`16.6%` ของ small, `7.9%` ของ large)** -> Eager BF16 จึงทั้งช้ากว่า TF32 (`11.68 vs 7.53 ms` small, `25.01 vs 21.97 ms` large) และ drift สูง (`0.286452`)
     - *วิธีแก้ชั่วคราวใน Eager*: `_is_cast_heavy_eager_mlp` ข้าม eager AMP อัตโนมัติถ้าโมเดลมี `LayerNorm >= 6` โดยไม่มี `Conv2d`/`Attention` และไม่ได้เปิด compile -> `deep_mlp` กลับมาเร็วขึ้นและ drift ลด **13x** ทันที
  2. **Kernel Launch & Memory-Bandwidth ของ Elementwise Ops**: ใน `transformer_lm:small` เวลา compute น้อยกว่า overhead ของ eager wrapper (`0.96x`), และใน `transformer_lm:large` ยังเหลือ `aten::mul` (`180.3 ms / 11.8%`) + `vectorized_elementwise` (`64.1 ms`) + `aten::copy_` (`59.7 ms`) ที่อ่าน-เขียน VRAM ซ้ำหลายรอบ
- **วิธีแก้ด้วย `auto_compile.py` (Phase 3.5)**:
  - ใช้ `mod_self._compiled_call_impl = torch_mod.compile(mod_self._call_impl, ...)` **in-place** (ไม่ต้องแก้ตัวแปร `model` ในโค้ดผู้ใช้ และ `state_dict()` ไม่มี prefix `_orig_mod.`) โดยซ้อนให้อยู่ **ด้านใน** `auto_amp` และ `channels_last` เพื่อให้ TorchInductor มองเห็นกราฟที่เป็น `bf16` + `channels_last` ครบถ้วน และเปิด `auto_amp` ให้ `deep_mlp` เมื่อ compile เปิดอยู่เพราะ Inductor fuse cast ops ได้
  - ผลใน [144108](../../report/144108_20261117-phase-3.5.txt):
    - TorchInductor ยุบ `RMSNorm`, `SwiGLU`, `LayerNorm`, `GELU` และ cast ops เป็น Triton fused kernels เดียว ไม่ต้องเขียน activation ลง VRAM ระหว่าง op ->
      - `transformer_lm:small` พลิกจาก `0.96x` เป็น **`2.08x`** (`11.72 -> 5.65 ms`)
      - `transformer_lm:large` ทะลุ **`4.36x`** (`780.07 -> 179.02 ms`) และ **Peak VRAM ลดลง `-4,167.4 MB` (`9,174.4 -> 5,006.9 MB`, ลดลง 45.4%)** โดยคุ้มทุน Cold Compile ใน **`42 steps`**
      - `deep_mlp:medium` ใช้ BF16+Compile ได้อย่างแม่นยำ (`0.002193` PASS) และเร็วขึ้นเป็น **`1.78x`** (`16.40 -> 9.20 ms`)
      - `TORCHINDUCTOR_CACHE_DIR` + `TORCHINDUCTOR_FX_GRAPH_CACHE=1` ลด Cold Step0 -> Warm Step0 จาก `24.91s -> 6.20s` (เกือบ 4x) และมี 1-shot `dynamic=True` guard สำหรับ `edge_cases:dynamic_shape`

### 6. Allocator & Host Prelaunch (`memory_allocator.py`)
- ตั้ง `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` เพื่อลด CUDA virtual memory fragmentation (วัดได้ `num_alloc_retries=0`, `num_ooms=0` บน L4), ตั้ง `OMP_NUM_THREADS`/`MKL_NUM_THREADS` ตาม physical cores, และตั้ง cache env ก่อน `import torch`

---

## จุดจำกัดที่เหลือและเป้าหมายถัดไป (Phase 3.6 – 3.7)
1. **Inference Latency / Batch=1 (`Phase 3.6`)**: งาน inference สั้น ๆ ยังติด CPU launch overhead และ autograd tracking -> ใช้ `torch.inference_mode()`, `reduce-overhead` / CUDA Graphs และทดลอง FP8 บน L4 (`sm_89`)
2. **Custom Triton Kernels (`Phase 3.7`)**: สำหรับโมเดลหรือ ops ที่แม้ผ่าน `torch.compile` หรือรันในโหมดที่ไม่ compile แล้วยัง memory/bandwidth-bound (เช่น `RMSNorm`, `CrossEntropy` เพื่อลด peak VRAM ของ logits ขนาดใหญ่, `SwiGLU` ลด `aten::mul` 180 ms)
3. **Break-Even Heuristic สำหรับงานสั้นมาก (`< 50 steps`)**: บนโมเดลเล็ก (`transformer_lm:small`, `deep_mlp:small`) หรือ `dynamic_shape` การ compile ครั้งแรกใช้เวลา `~13–19s` (BreakEven `2,172–8,313 steps` เมื่อ cold) จึงเหมาะกับงานเทรนระยะยาวหรือเมื่อมี warm cache บนดิสก์ถาวร
4. **ความไวเชิงตัวเลขของ Deep Sequential Unresidualized MLP (`deep_mlp:large w2048 d24 lr=1e-3`)**: แม้ข้าม AMP แล้ว `TF32` ยังมี `MaxRelDiff = 0.021711` (เกินเกณฑ์ `0.02` เล็กน้อย) เพราะไม่มียืดหยุ่นจาก skip connection เมื่อคูณเมทริกซ์ 24 ชั้นติดกัน
