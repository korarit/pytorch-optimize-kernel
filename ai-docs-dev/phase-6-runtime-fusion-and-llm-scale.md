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

## กฎกลางและ Exit Criteria ของ Phase 6
- ใช้ Shared Deep Debug Telemetry Helper (`scripts/phase5_report_helper.py` หรือ helper กลางที่อัปเกรดเพิ่ม) ออกรายงานครบ 7 ส่วนเทียบเท่าระดับ Phase 3/5
- คง `dependencies = []` ใน [`pyproject.toml`](../pyproject.toml), ไม่แก้ไฟล์ workload ใน [`pipline-for-test/workloads/`](../pipline-for-test/workloads/), และรักษาความเข้ากันได้ของ `state_dict()` / Checkpoint Cross-Load 100%
- `python3 scripts/run_gate.py --mode fast` (CPU/CI) และ `--mode full` (Colab GPU) ต้องผ่าน **100%**
- **[WAIT FOR USER]**: หลังทำเสร็จแต่ละ Sub-Plan (`6.1 → 6.2 → 6.3`) ต้องรอให้ผู้ใช้รัน verify script และตรวจ Text Report บน Google Colab GPU ก่อนไปขั้นถัดไป
