# Phase 6: Zero-Sync Runtime Dispatch, Hybrid Compile+FusedLCE & LLM-Scale Kernels (Overview)

## เป้าหมาย (Goal)
ต่อยอดจากผลการตรวจสอบโค้ดและผล Profiler จริงใน Phase 5 ([Phase 5 Optimization History](./optimize-history/phase-5-kernel-compile-and-pipeline-refinements.md)) โดยแบ่งแผนออกเป็น **3 Sub-Plans** เพื่อปิดจุดเสีย latency แฝง, ปลดล็อกการใช้ optimization ร่วมกันแบบไม่ต้องเลือกอย่างใดอย่างหนึ่ง, และขยายความเข้ากันได้กับโมเดล LLM/Vision จริง:
1. **กำจัด Hidden Sync, Double-Compile & Hook Dispatch Overhead** — ตัด `.cpu().item()` sync ทุก step ใน `auto_amp`, ตัด Step-0 double-compile ใน `train_cuda_graphs`, และยุบ `Module.__call__` 6 ชั้นเป็น Unified Dispatcher พร้อมแยก train/eval compile state
2. **ปลดล็อก Hybrid `torch.compile` + `FusedLinearCrossEntropy` และฟื้นความเร็ว Activation Checkpointing** — ให้ Backbone คงความเร็วจาก Inductor (`~4x`) พร้อมประหยัด head VRAM (`-65% to -81%`), Auto-enable Triton kernels เมื่อ bypass compile, เพิ่ม Partial/Selective Checkpointing และ Step-0 OOM Retry Shield
3. **LLM-Scale Kernels (`Fused RoPE`, Strided Zero-Copy, HF Matching), FP8 Static Scale & Pipeline Polish** — เติม Fused RoPE, จับ `gate_proj`/`up_proj`/`down_proj`, ลบ `.contiguous()` copy บน causal/chunked tensors, แก้ `has_large_m` + static scale ใน FP8, และปรับ heuristic ของ `channels_last` / `optimizer` / `CLI` startup

---

## Sub-Plans (แบ่งเป็น 3 แผนย่อย)

| Sub-Plan | ชื่อ | จุดเน้นหลัก | ไฟล์รายละเอียด | Report เป้าหมาย |
| :--- | :--- | :--- | :--- | :--- |
| **6.1** | Zero-Sync AMP Watchdog, Unified Module Dispatcher & Single-Pass Training CUDA Graphs | ตัด sync & Python dispatch overhead + แก้ double-compile 6–7s & train/eval conflict | [6.1-zero-sync-amp-unified-dispatch-and-cg-warmup.md](./phase-6/6.1-zero-sync-amp-unified-dispatch-and-cg-warmup.md) | `report/HHMMSS_YYYYMMDD-phase-6.1.txt` |
| **6.2** | Hybrid Compile+FusedLCE, Triton-Backed Activation Checkpointing & OOM Shield | ใช้ Compile + FusedLCE พร้อมกัน + ฟื้นความเร็ว act-ckpt + กัน OOM ตั้งแต่ Step 0 | [6.2-compile-fused-lce-and-selective-act-ckpt.md](./phase-6/6.2-compile-fused-lce-and-selective-act-ckpt.md) | `report/HHMMSS_YYYYMMDD-phase-6.2.txt` |
| **6.3** | LLM Triton Kernels (Fused RoPE & Zero-Copy Strides), FP8 Calibration & Pipeline Polish | Fused RoPE, HF SwiGLU, row-strided kernels, FP8 2D-token/static-scale, CLI cache | [6.3-llm-kernels-fp8-scale-and-startup.md](./phase-6/6.3-llm-kernels-fp8-scale-and-startup.md) | `report/HHMMSS_YYYYMMDD-phase-6.3.txt` |

---

## Zero-Code Preservation Guardrails (ทั้ง 5 ข้อที่ต้องรักษาตลอด Phase 6)
เพื่อให้หลักการ **Zero-Code Injection** (`ptopt -- python3 script.py` โดยไม่แก้โค้ดผู้ใช้, ไม่แก้ `state_dict()`, ไม่ทำผลลัพธ์เพี้ยน) ยังคงสมบูรณ์ 100% ทุก sub-plan ต้องปฏิบัติตาม guardrails ทั้ง 5 ข้อนี้:
1. **Strict Semantic Guard สำหรับ `RoPE` และ `SwiGLU` ([Plan 6.3](./phase-6/6.3-llm-kernels-fp8-scale-and-startup.md))**: ห้ามเดาจากชื่อ attribute (`gate_proj`/`up_proj`/`down_proj`) เพียงอย่างเดียวเพราะโมเดลแบบ **GeGLU (`GELU`)** จะพัง; ต้องตรวจ `act_fn == SiLU` + ไม่มี `Dropout`/submodules แฝง และ `RoPE` ต้องตรวจ layout (half-rotate vs interleaved) ชัดเจนก่อน patch เสมอ
2. **Step-0 `OutOfMemoryError` Catch-and-Retry Safety ([Plan 6.2](./phase-6/6.2-compile-fused-lce-and-selective-act-ckpt.md))**: ใช้เป็น last-resort rescue เฉพาะ `mode="auto"` เท่านั้น (ปิดเมื่อ `--strict` หรือ `PTOPT_ACT_CKPT_OOM_RETRY=0`), log `[FALLBACK]` ชัดเจนเพื่อแจ้งผู้ใช้เรื่อง side-effect ที่อาจเกิดก่อน OOM (เช่น `BatchNorm` stats / RNG)
3. **Zero-Sync AMP Watchdog Without NaN Weight Leak ([Plan 6.1](./phase-6/6.1-zero-sync-amp-unified-dispatch-and-cg-warmup.md))**: สั่ง D2H copy แบบ `non_blocking=True` ใน `backward()` แล้วไปตรวจผลใน **`optimizer.step()` pre-hook** (ไม่ปล่อย NaN ไป step ถัดไปจนแก้ weights ใน `bf16`, และคง sync ทันทีใน `--strict`)
4. **`LazyLinearLogits` Cross-Boundary Safety ([Plan 6.2](./phase-6/6.2-compile-fused-lce-and-selective-act-ckpt.md))**: จำกัด `LazyLinearLogits` เฉพาะคู่ `lm_head` ใน `training=True` + `is_grad_enabled()`, ป้องกัน `meta` tensor leakage นอก `__torch_function__` และทดสอบ boundary ร่วมกับ `AOTAutograd` / `torch.compile`
5. **`capturable=True` & Fingerprint Cache Freshness ([Plan 6.3](./phase-6/6.3-llm-kernels-fp8-scale-and-startup.md))**: เปิด `capturable=True` เฉพาะ opt-in/CUDA Graph + normalize `state['step']` ใน `checkpoint_compat.py` ไม่ให้พังข้าม CPU/CUDA, และ **ห้าม cache `/dev/shm` size แบบตายตัว** ใน Fingerprint disk cache (ต้อง probe `/dev/shm` สดและผูก key กับ `CUDA_VISIBLE_DEVICES` + driver/torch/triton versions)

---

## มาตรฐาน Report Log ฉบับละเอียด (Phase 3/5-Grade Deep Debug Standard)
ทุก Sub-Plan (`6.1 – 6.3`) **ต้องออก Detailed Text Report ครบทั้ง 7 ส่วน** ผ่าน `scripts/phase5_report_helper.py` (หรือ shared helper ที่อัปเกรดเพิ่ม) ด้วย `PTOPT_DEBUG_PROFILE=1` และ `PTOPT_TORCH_PROFILE=1` เพื่อพิสูจน์ผลด้วยตัวเลขระดับ Op และ Step:
1. **Full Header & Hardware Stack**: `Timestamp`, `Repo Root`, `Git Branch/Commit`, `Google Colab`, `GPU / SM`, `Python / PyTorch`, `CUDA / Triton`, `Runtime Cache Key`, ผล `Unit` & `Migrate` tests
2. **Ablation & Regression Matrix**: `Step0 ms`, `Base ms`, `Wrap ms (Med)`, `Wrap P95`, `Wrap P99`, `E2E s (B/W)`, `Speedup`, `VRAM MB (B/W/d)`, `MaxRelDiff`, `Status`
3. **`[DEEP DEBUG: STEP PHASE BREAKDOWN]`**: แยก `Step0 ms`, `Steady ms`, `Data/Other (%)`, `Forward (%)`, `Loss/Clip (%)`, `Backward (%)`, `Optim (%)` (เทียบ `baseline_dbg` vs ก่อนแก้ vs หลังแก้)
4. **`[DEEP DEBUG: CUDA ALLOCATOR & VRAM HEADROOM]`**: `Alloc MB`, `Resv MB`, `Ovhead MB`, `Headroom MB`, `Util %`, `Retries` (`num_alloc_retries`)
5. **`[TOP PYTORCH PROFILER OPS]`**: แสดง `self_cuda_ms`, `self_cpu_ms` และ Top 8–12 CUDA/CPU Ops (`calls`, `self_ms`, `%`, `total_ms`) เพื่อพิสูจน์ระดับ kernel ว่าไม่มี hidden sync / redundant copy / graph break
6. **`[WARMUP VS STEADY STEP TRACE]`**: แสดงเวลา step รายตัว (`step0`, `steady_med`, `s0..s7+`) เพื่อจับ compile/warmup spike
7. **`[AUTOMATED DIAGNOSIS]` + `[HOOK / KERNEL / FALLBACK AUDIT]` + `[VERIFICATION STATUS]`**: สรุป decision, skipped, fallback, guardrail trigger ของทุก pass พร้อม `STATUS : PENDING_USER_REVIEW`

---

## กฎกลางและ Exit Criteria ของ Phase 6
- ใช้ Shared Deep Debug Telemetry Helper (`scripts/phase5_report_helper.py`) ออก Detailed Text Report (`report/HHMMSS_YYYYMMDD-phase-6.*.txt`) ครบทั้ง 7 ส่วนข้างต้นทุก sub-phase
- คง `dependencies = []` ใน [`pyproject.toml`](../pyproject.toml), ไม่แก้ไฟล์ workload ใน [`pipline-for-test/workloads/`](../pipline-for-test/workloads/), และรักษาความเข้ากันได้ของ `state_dict()` / Checkpoint Cross-Load 100%
- `python3 scripts/run_gate.py --mode fast` (CPU/CI) และ `--mode full` (Colab GPU) ต้องผ่าน **100%**
- **[WAIT FOR USER]**: หลังทำเสร็จแต่ละ Sub-Plan (`6.1 → 6.2 → 6.3`) ต้องรอให้ผู้ใช้รัน verify script และตรวจ Text Report บน Google Colab GPU ก่อนไปขั้นถัดไป
