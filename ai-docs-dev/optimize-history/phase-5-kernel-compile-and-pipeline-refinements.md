# Optimization Analysis (Phase 5.1 – 5.5): Compile Guardrails, Fused Linear+CE, FP8 Fusion, Training CUDA Graphs & Memory/Async Pipeline

เอกสารนี้สรุปการวิเคราะห์คอขวด กลไกการแก้ และผลวัดจริงบน **Google Colab NVIDIA L4 (`sm_89`, PyTorch `2.14.1+cu130`, Triton `3.8.0`)** ของ **Phase 5 (Sub-Phase 5.1 – 5.5)** โดยอ้างอิงจากโค้ดใน [`src/`](../../src/) และรายงาน Deep Debug + Torch Profiler ใน [`report/`](../../report/):
- **Phase 5.1**: [044642 (รอบแรก)](../../report/044642_20261118-phase-5.1.txt) → [051102 (PASS + Deep Debug)](../../report/051102_20261118-phase-5.1.txt)
- **Phase 5.2**: [052905 (รอบแรก)](../../report/052905_20261118-phase-5.2.txt) → [053832 (64MB Chunk + `addmm_` PASS)](../../report/053832_20261118-phase-5.2.txt)
- **Phase 5.3**: [060633 (drop_w FAIL)](../../report/060633_20261118-phase-5.3.txt) → [061553 (MHA Guard + PASS)](../../report/061553_20261118-phase-5.3.txt)
- **Phase 5.4**: [065346 (Training CUDA Graphs PASS)](../../report/065346_20261118-phase-5.4.txt)
- **Phase 5.5**: [072807 (act_ckpt FAIL)](../../report/072807_20261118-phase-5.5.txt) → [074734 (Bypass Child Compile PASS)](../../report/074734_20261118-phase-5.5.txt)

---

## ภาพรวม: คอขวดหลัง Phase 4 และผลลัพธ์จาก Phase 5

| Sub-Phase | คอขวดเดิมหลัง Phase 4 | โมดูลที่แก้ | กลไกหลักที่เพิ่ม | ผลวัดจริงบน NVIDIA L4 (`sm_89`) |
| :--- | :--- | :--- | :--- | :--- |
| **5.1** | โมเดลเล็ก + dynamic shapes (`edge_cases`) ช้าลงเป็น **`0.81x`** และกิน VRAM เพิ่ม **`+47.5 MB`** | [`auto_compile.py`](../../src/passes/auto_compile.py), [`auto_amp.py`](../../src/passes/auto_amp.py) | Step-0 Static Tiny Compile Gate (`<50k params` & `<32k elems`) + Eager Tiny AMP Skip | `edge_dynamic_only`: `0.81x (+47.5 MB)` → **`1.00x (+0.0 MB)`**<br>`edge_all`: `0.87x (+47.7 MB)` → **`1.00x (-0.1 MB)`** |
| **5.2** | `Linear + CrossEntropy` บน vocab ใหญ่สร้าง `[N, V]` logits เต็มก้อนใน VRAM (`3.36 GB` ที่ `V=65536`) | [`fused_linear_cross_entropy.py`](../../src/kernels/fused_linear_cross_entropy.py), [`registry.py`](../../src/kernels/registry.py) | `LazyLinearLogits` tensor subclass + Chunked `FusedLinearCrossEntropyFn` (64 MB chunk + `addmm_` in-place `d_weight`) | Micro `V=65536`: **`1.24x`**, VRAM **`-81.4%` (`3,361.2 → 625.2 MB`)**<br>E2E `vocab32k`: **`3.34x`** (`36.78 ms` vs `47.56 ms` compile), VRAM **`-65.3%` (`-2,287.0 MB`)** |
| **5.3** | Eager FP8 (`150.5 ms`, `2.48x`) แพ้ `BF16+Compile` (`89.9 ms`, `4.16x`) เพราะ Triton quantizer ขวางกราฟ Dynamo | [`inference_opt.py`](../../src/passes/inference_opt.py) | Pure-ATen Traceable FP8 Quantizer (`0 graph breaks`) + Prewarm Weight Cache + `PTOPT_FP8_DROP_BF16_EVAL=1` + `MHA.out_proj` guard | `infer_bs64_large` (`fp8_compile`): **`77.41 ms` (`4.78x`, `826.8 samp/s`)** แซง `bf16_compile` (`88.73 ms / 4.17x`)<br>`fp8_compile_drop_w`: **`78.08 ms` (`4.74x`)**, VRAM **`411.4 MB` (`-355.5 MB`)** |
| **5.4** | Training โมเดลขนาดเล็ก/launch-bound ติด CPU dispatch overhead แม้ใช้ `compile=default` | [`auto_compile.py`](../../src/passes/auto_compile.py), [`schema.py`](../../src/config/schema.py), [`rules_builtin.py`](../../src/config/rules_builtin.py) | 2-Step Shape Stability Gate + `mode="reduce-overhead"` + `cudagraph_mark_step_begin()` + Dropout/GradScaler/`>15M` guards | `deep_mlp`: `7.77 →` **`4.25 ms` (`2.12x` base / `1.83x` vs compile)**<br>`cnn_vision_small`: **`4.71 ms` (`2.01x` / `1.53x`)**<br>`transformer_lm_small`: **`4.75 ms` (`2.58x` / `1.46x`)** |
| **5.5** | (ก) โมเดล activation สูงเจอ OOM (ข) DataLoader H2D ไม่ overlap กับ compute stream เต็มที่ | [`activation_checkpoint.py`](../../src/passes/activation_checkpoint.py), [`dataloader_tune.py`](../../src/passes/dataloader_tune.py) | VRAM-aware Auto Activation Checkpointing (`>82%` / `always`, `use_reentrant=False`) + `_AsyncCudaPrefetchIter` (`copy_stream` + `record_stream`) | `lm_act_med`: **`1,171.8 MB` (`45.6%` saved vs `no_ckpt`)**<br>`lm_act_large`: **`1,684.1 MB` (`53.5%` saved vs `no_ckpt`, `-5,547.2 MB` vs base)**<br>`cnn_vision_224`: `Data/Other` `4.11 → 3.35 ms`, E2E `30.35s → 16.74s` |

---

## เจาะลึกทีละ Sub-Phase: Root Cause, การจูน และบทเรียนจาก Profiler

### 1. Phase 5.1: Tiny Dynamic-Shape Compile & Eager AMP Bypass
- **สาเหตุที่พบในรอบแรก ([044642](../../report/044642_20261118-phase-5.1.txt))**:
  1. `Sequential` ใน `edge_dynamic_only` (`9,352` parameters, input `3,072–5,376` elements) ถูกสั่ง `torch.compile()` ตั้งแต่ **Step 0** ก่อนที่จะพบว่า input shape เปลี่ยนใน Step 1 แม้จะ revert กลับ eager แล้วก็ยังทิ้ง Inductor scratch memory **`+47.5 MB`** (`16.9 → 64.3 MB`) และทำ `edge_dynamic_only` ตกไป **`0.81x` (`1.86 ms`)**
  2. หลัง revert จาก compile, `auto_amp` ยังครอบ `torch.amp.autocast(cuda, bf16)` ให้โมเดลจิ๋วที่รันแบบ eager ซึ่งค่า dispatch + dtype cast แพงกว่าเวลาคำนวณ GEMM เล็กๆ
- **สิ่งที่แก้ใน [`auto_compile.py`](../../src/passes/auto_compile.py) & [`auto_amp.py`](../../src/passes/auto_amp.py)**:
  - เพิ่ม `_is_worth_static_compile` (Step 0 gate: ข้าม compile หาก `total_params < 50_000` และ `input_elems < 32_768`) คู่กับ `_is_worth_dynamic_compile` (Step 1 gate: `< 250_000` params & `< 65_536` elems → revert eager)
  - เพิ่ม `_is_tiny_uncompiled_mlp` ใน `auto_amp.py` เพื่อข้าม eager `autocast` สำหรับโมเดลที่ไม่มี Conv/Attention และไม่ผ่านเกณฑ์ compile
- **ผลลัพธ์ ([051102](../../report/051102_20261118-phase-5.1.txt))**: `edge_dynamic_only` และ `edge_all` กลับมาที่ **`1.00x`** (`1.48 / 1.45 ms`) และ VRAM **`+0.0 / -0.1 MB`** ในทุกโปรไฟล์ โดยไม่กระทบ `transformer_lm:medium` (**`3.75x–3.82x`**, `-779.5 MB`)

### 2. Phase 5.2: Fused Chunked Linear + CrossEntropy (Zero Logits Materialization)
- **สาเหตุที่ต้องจูนรอบสอง (`052905 → 053832`)**:
  - ในรอบแรก ([052905](../../report/052905_20261118-phase-5.2.txt)) ใช้ chunk budget เล็กเกินไปและสะสม `d_weight` ด้วย `torch.mm(...).float()` แยก chunk ทำให้ micro-benchmark `4096x1024 -> V=65536` ช้ากว่า Eager (`256.99 ms` = `0.25x`) และ `vocab32k` ช้ากว่า `balanced_compile` (`55.14 ms` vs `48.08 ms`)
- **การแก้ (`commit 25d613a`) ใน [`fused_linear_cross_entropy.py`](../../src/kernels/fused_linear_cross_entropy.py)**:
  1. ขยาย `_TARGET_CHUNK_BYTES` เป็น **`64 MB`** (`compute_optimal_chunk_size`, align เป็นพหุคูณของ 64) เพื่อเพิ่ม occupancy ของ cuBLAS Tensor Core GEMM
  2. ใช้ in-place cuBLAS **`d_weight.addmm_(dl_mm.t(), h_c)`** โดยตรงใน `mm_dtype` เมื่อปลอดภัย (`fp32` หรือ `bf16` ที่ `n_chunks <= 16`) ตัด temporary `[V, H]` allocation และ separate `add_` + `.float()` cast ทิ้ง
  3. ใช้ `LazyLinearLogits` (`torch.Tensor` subclass บน `meta` device) ดักจับเฉพาะคู่ `nn.Linear(out_features >= 4096) -> F.cross_entropy`, รองรับ `view`/`reshape`/`contiguous`/slice ที่ไม่แตะ vocab dim และ materialize อัตโนมัติหาก consumer อื่นแตะ logits
- **ผลลัพธ์ ([053832](../../report/053832_20261118-phase-5.2.txt))**:
  - Micro `4096x1024 -> V=65536`: **`64.60 → 52.27 ms` (`1.24x`)**, Peak VRAM **`3,361.2 → 625.2 MB` (`-81.4%`)**, `LossAbsDiff = 1.53e-05`
  - E2E `vocab32k`: **`123.01 → 36.78 ms` (`3.34x`)** (เร็วกว่า `balanced_compile` `47.56 ms` ถึง **22.7%**), Peak VRAM **`3,501.5 → 1,214.5 MB` (`-65.3%`)**, E2E `9.24s` vs `24.13s`

### 3. Phase 5.3: Compilable FP8 Inference Fusion (`sm >= 89`)
- **วิวัฒนาการ (`060633 → 061553`)**:
  - **ปัญหาที่เจอในรอบ `060633`**:
    1. เมื่อ `torch.compile` ทำงาน, Triton quantizer เดิมทำให้เกิด graph break ทำให้ Inductor รวม quantize เข้ากับ `LayerNorm`/`GELU` ก่อนหน้าไม่ได้ → แก้ด้วยการแยก Path 1 (`_is_dynamo_compiling` ใช้ pure-ATen `_quantize_fp8_traceable` **0 graph breaks**) และ Path 2 (Eager ใช้ 2-pass Triton quantizer) พร้อม `prewarm_fp8_weight_caches` ก่อน compile
    2. ในโหมด `PTOPT_FP8_DROP_BF16_EVAL=1`, [`060633`](../../report/060633_20261118-phase-5.3.txt#L32) เกิด `FAIL` เพราะ `nn.MultiheadAttention` ดึง `out_proj.weight` ไปใช้ตรงๆ ใน `F.multi_head_attention_forward` โดยไม่ผ่าน `out_proj.forward()` → แก้ใน `commit 57943cc` ให้ข้าม `MHA.out_proj` และ `NonDynamicallyQuantizableLinear`
- **ผลลัพธ์ ([061553](../../report/061553_20261118-phase-5.3.txt))**:
  - Torch Profiler ยืนยันว่า `gelu` + dynamic quantize ถูก fuse เป็น kernel เดียว (`triton_poi_fused__scaled_mm__to_copy_clamp_div_g`) และ CUTLASS/cuBLASLt ใช้ `sm89_xmma_gemm_e4m3bf16`
  - `infer_bs64_large`: **`370.01 → 77.41 ms` (`4.78x`, `826.8 samp/s`)** แซง `bf16_compile` (`88.73 ms`, `4.17x`), และเมื่อเปิด `PTOPT_FP8_DROP_BF16_EVAL=1` ลด VRAM ลงเหลือ **`411.4 MB` (`-355.5 MB` vs baseline)**

### 4. Phase 5.4: Static-Shape Training CUDA Graphs
- **กลไกใน [`auto_compile.py`](../../src/passes/auto_compile.py)**:
  - รอให้ shape signature คงที่ติดต่อกัน 2 steps (`stable >= 2`) ก่อนเลื่อนจาก `mode="default"` เป็น `mode="reduce-overhead"` พร้อมยิง `torch.compiler.cudagraph_mark_step_begin()` หน้าทุก training forward
  - Guardrails ป้องกันความเพี้ยน/แครช: ปิดอัตโนมัติเมื่อ `sm < 80`, `GradScaler` active (`fp16` on T4), มี active `Dropout` (`p > 0` เว้นตั้ง `PTOPT_TRAIN_CG_ALLOW_DROPOUT=1`), `params > 15M` (โมเดลใหญ่ที่ compute-bound อยู่แล้ว), หรือ shape เปลี่ยน
- **ผลลัพธ์ ([065346](../../report/065346_20261118-phase-5.4.txt))**:
  - `self_cpu` (window 3 steps) ลดฮวบ: `deep_mlp` (`53.6 → 21.8 ms`), `cnn_vision_small` (`48.0 → 19.2 ms`), `transformer_lm_small` (`45.4 → 22.1 ms`)
  - Steady Step Time: `deep_mlp` **`4.25 ms` (`2.12x` base / `1.83x` vs compile)**, `cnn_vision_small` **`4.71 ms` (`2.01x` / `1.53x`)**, `transformer_lm_small` **`4.75 ms` (`2.58x` / `1.46x`)**

### 5. Phase 5.5: Auto Activation Checkpointing & Dedicated CUDA Copy Stream Prefetch
- **ปัญหาที่เจอระหว่างพัฒนา (`072807 → 074734`)**:
  - ในรอบแรก ([072807](../../report/072807_20261118-phase-5.5.txt#L21-L23)) `act_ckpt` พังกลาง backward เพราะ `checkpoint(..., use_reentrant=False)` รัน recompute ใน backward แล้วไปชนกับ submodule compile หรือ PyTorch `#166926` tensor count mismatch → แก้ใน `commits 78209e3, 5b7f9aa` โดย mark ทั้ง root และ descendant submodules เป็น `_ptopt_compile_ineligible = True` และเคลียร์ `_compiled_call_impl` เมื่อเปิด activation checkpointing
- **ผลลัพธ์ ([074734](../../report/074734_20261118-phase-5.5.txt))**:
  - **Part A (Activation Checkpointing)**: `lm_act_med` VRAM **`2,153.1 → 1,171.8 MB` (`-45.6%` vs `no_ckpt`)**, `lm_act_large` VRAM **`3,623.3 → 1,684.1 MB` (`-53.5%` vs `no_ckpt`, `-5,547.2 MB` vs baseline)** โดย `MaxRelDiff = 1e-6–2e-6`
  - **Part B (Dedicated CUDA Copy Stream Prefetch)**: `cnn_vision_224` (`DataLoader` + `pin_memory`) ลดเวลา `Data/Other` จาก `4.11 → 3.35 ms` และลด E2E จาก `30.35s → 16.74s` (`3.37x` steady step)

---

## ข้อจำกัดและช่องว่างที่ตรวจพบจาก Phase 5 (ส่งต่อสู่ Phase 6)

1. **Per-Step Sync แฝงใน `auto_amp.py`**: [`check_loss_stability`](../../src/passes/auto_amp.py#L228-L245) ยังเรียก `.detach().cpu().item()` บน loss ทุก backward step (`balanced`/`aggressive`) ทำให้เกิด CPU-GPU synchronization ระหว่าง Forward และ Backward
2. **Double-Compile Overhead ใน `train_cuda_graphs`**: Step 0 compile `mode="default"` (`~6.2–6.9s`) แล้ว Step 1 recompile `mode="reduce-overhead"` (`~8.5–14.6s`) รวม `15–21s` ทำให้ E2E ของ `deep_mlp` ยาวกว่า `compile_default` (`28.45s` vs `27.81s`), และ `mode="max-autotune"` ใน `aggressive` ยังเปิด CUDA Graphs โดยไม่เคารพ Dropout/>15M guard
3. **`Module.__call__` ซ้อน 6 ชั้น + Conflict ข้ามโหมด**: 6 passes wrap `nn.Module.__call__` แยกกัน และแชร์ `_compiled_call_impl` เดียวกัน ทำให้ถ้าสลับ `.train() ↔ .eval()` บนโมเดลเดิม `inference.cuda_graphs` หรือ `compile.auto` ข้าม/ค้างโหมดผิด
4. **ความเร็วตกเมื่อเปิด Activation Checkpointing (`4.17x → 1.35x`)**: การปิด `torch.compile` ทั้งโมเดลโดยไม่ auto-enable custom Triton kernels (`RMSNorm`, `SwiGLU`, `FusedLinearCE`) และไม่มี partial-block/SAC หรือ Step-0 OOM catch ทำให้ความเร็วตกแรงและยังไม่กัน OOM ภายใน Step 0
5. **ไม่สามารถใช้ `torch.compile` + `FusedLinearCrossEntropy` พร้อมกัน**: `registry.py` ปิด `fused_linear_cross_entropy` เมื่อ `compile=True` ทำให้โมเดล LLM ทั่วไปยังต้องเสีย VRAM จอง `[N, V]` logits ก้อนใหญ่หากต้องการใช้ `torch.compile` บน backbone
6. **ช่องว่างใน Real-World LLMs & FP8**: ยังขาด **Fused RoPE**, การจับ HF `gate_proj`/`up_proj`/`down_proj`, `nested dict/list` input signature, zero-copy row-strided Triton kernels, และแก้ bug `has_large_m` บน 2D token IDs `(B, S)` ใน FP8 inference
