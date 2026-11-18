# Phase 5: Next-Wave Kernel, Compile & Pipeline Optimizations (Overview)

## เป้าหมาย (Goal)
ต่อยอดจาก Phase 1–4 เพื่อปิดช่องว่างประสิทธิภาพทั้ง **กลุ่ม 1 (Kernel & Compile Refinements: 5.1–5.3)** และ **กลุ่ม 2 (Training Graphs, VRAM Scaling & Async Pipeline: 5.4–5.5)** โดยไม่กระทบ accuracy, checkpoint compatibility และโครงสร้าง zero-code injection:
1. หยุดการช้าลงของโมเดลขนาดเล็กที่มี dynamic shapes (`0.79x -> >= 1.00x`) พร้อมพิสูจน์สาเหตุที่ระดับ Op ด้วย Profiler
2. ตัดการจอง/อ่าน-เขียน Tensor `logits` ขนาดใหญ่ `[N, V]` ด้วย **Fused Chunked Linear + CrossEntropy**
3. ทำให้โหมด Opt-in **FP8 (`sm >= 89`)** ทำงานร่วมกับ `torch.compile` ได้โดยไม่เกิด graph breaks เพื่อแข่งกับ `BF16 + Compile` (`4.16x`)
4. นำ **CUDA Graphs (`reduce-overhead`)** มาเร่ง Static-Shape Training Loops สำหรับโมเดล launch-bound (`deep_mlp`, small Transformers)
5. เพิ่ม **Auto Activation Checkpointing** (VRAM-aware OOM shield) และ **Dedicated CUDA Copy Stream H2D Prefetcher**

---

## Sub-Phases

| กลุ่ม | Sub-Phase | ชื่อ | ไฟล์รายละเอียด | Report เป้าหมาย |
| :--- | :--- | :--- | :--- | :--- |
| กลุ่ม 1 | **5.1** | Tiny Dynamic-Shape Compile & AMP Bypass | [5.1-tiny-dynamic-compile-guard.md](./phase-5/5.1-tiny-dynamic-compile-guard.md) | `report/HHMMSS_YYYYMMDD-phase-5.1.txt` |
| กลุ่ม 1 | **5.2** | Fused Chunked Linear + CrossEntropy Kernel | [5.2-fused-linear-cross-entropy.md](./phase-5/5.2-fused-linear-cross-entropy.md) | `report/HHMMSS_YYYYMMDD-phase-5.2.txt` |
| กลุ่ม 1 | **5.3** | Compilable FP8 Inference Fusion (`sm >= 89`) | [5.3-compilable-fp8-inference.md](./phase-5/5.3-compilable-fp8-inference.md) | `report/HHMMSS_YYYYMMDD-phase-5.3.txt` |
| กลุ่ม 2 | **5.4** | CUDA Graphs for Static-Shape Training Loops | [5.4-training-cuda-graphs.md](./phase-5/5.4-training-cuda-graphs.md) | `report/HHMMSS_YYYYMMDD-phase-5.4.txt` |
| กลุ่ม 2 | **5.5** | Auto Activation Checkpointing & Async H2D Prefetch | [5.5-auto-activation-ckpt-and-async-prefetch.md](./phase-5/5.5-auto-activation-ckpt-and-async-prefetch.md) | `report/HHMMSS_YYYYMMDD-phase-5.5.txt` |

---

## มาตรฐานรายงานและ Deep Debug Logging ของ Phase 5 (Phase 3-Grade Telemetry Standard)

เนื่องจากรายงาน Phase 5.1 รอบแรก ([044642_20261118-phase-5.1.txt](../report/044642_20261118-phase-5.1.txt)) มีเพียงตารางสรุปเวลาเฉลี่ย ทำให้มองไม่เห็นว่าหลัง revert แล้วเวลาส่วนที่เหลือ (`1.86 ms vs 1.50 ms`) และ VRAM (`+47.5 MB`) หายไปกับขั้นตอนใด, **ทุก Sub-Phase (`5.1 – 5.5`) ต้องใช้ Shared Telemetry Helper (`scripts/phase5_report_helper.py`) และออกรายงานครบทั้ง 7 ส่วนเทียบเท่ารายงาน Phase 3.5–3.7**:

1. **Full Header & Hardware Stack**: `Timestamp`, `Git Commit`, `Google Colab`, `GPU / SM`, `Python / PyTorch`, `CUDA / Triton`, `Runtime Cache Key`, `Unit/Migrate Outcome`
2. **Ablation & Regression Matrix (ครบ Timing Percentiles)**:
   - `Case`, `Variant / Profile`, `Step0 ms`, `Base ms`, `Wrap ms (Med)`, `Wrap P95`, `Wrap P99`, `E2E s (B/W)`, `Speedup`, `VRAM MB (B/W/d)`, `MaxRelDiff`, `Status`
3. **`[DEEP DEBUG: STEP PHASE BREAKDOWN]`** (`PTOPT_DEBUG_PROFILE=1`):
   - แยกเวลา `Step0 ms`, `Steady ms`, `Data/Other (%)`, `Forward (%)`, `Loss/Clip (%)`, `Backward (%)`, `Optim (%)` (เทียบ `baseline_dbg` vs `safe_dbg` vs `balanced_dbg` / variants ของ sub-phase นั้น)
4. **`[DEEP DEBUG: CUDA ALLOCATOR & VRAM HEADROOM]`**:
   - แสดง `Alloc MB`, `Resv MB`, `Ovhead MB`, `Headroom MB`, `Util %`, `Retries` (`num_alloc_retries`)
5. **`[TOP PYTORCH PROFILER OPS]`** (`PTOPT_TORCH_PROFILE=1`, profiler window):
   - แสดง `self_cuda_ms`, `self_cpu_ms` และ Top CUDA/CPU Kernels พร้อม `calls`, `self_ms`, `%`, `total_ms`
6. **`[WARMUP VS STEADY STEP TRACE]`**:
   - แสดง Step 0..N latency breakdown (`step 0, 1, 2, ..., steady`) เพื่อพิสูจน์ว่า Step ใดเกิด compile/guard/fallback
7. **`[AUTOMATED DIAGNOSIS]` + `[HOOK / FALLBACK AUDIT]`**:
   - สรุปการตัดสินใจของแต่ละ Pass + สาเหตุอัตโนมัติ (เช่น CPU launch-bound vs compute-bound vs AMP cast churn) และ `STATUS : PENDING_USER_REVIEW`

---

## Exit Criteria รวมของ Phase 5
- [ ] สร้าง `scripts/phase5_report_helper.py` ให้ทุก `scripts/verify_phase5_*.py` ใช้ Deep Debug & Torch Profiler มาตรฐานเดียวกับ Phase 3
- [ ] Sub-Phase **5.1 – 5.5** ผ่าน Unit Tests, Migrate Parity, Deep Debug Profiler Audit และ Benchmark Gate ครบทุกข้อ
- [ ] `python3 scripts/run_gate.py --mode fast` (CPU) และ `--mode full` (Colab GPU) ยังคง **PASS 100%**
- [ ] ไม่แก้โค้ดใน `pipline-for-test/workloads/*.py` และคง `dependencies = []` ใน `pyproject.toml`
- [ ] **[WAIT FOR USER]** ตรวจผลรายงานบน Google Colab GPU ของแต่ละ sub-phase (5.1 → 5.5) ก่อน commit / ปิดรอบ
