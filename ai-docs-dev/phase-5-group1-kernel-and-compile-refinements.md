# Phase 5: Next-Wave Kernel, Compile & Pipeline Optimizations (Overview)

## เป้าหมาย (Goal)
ต่อยอดจาก Phase 1–4 เพื่อปิดช่องว่างประสิทธิภาพทั้ง **กลุ่ม 1 (Kernel & Compile Refinements: 5.1–5.3)** และ **กลุ่ม 2 (Training Graphs, VRAM Scaling & Async Pipeline: 5.4–5.5)** โดยไม่กระทบ accuracy, checkpoint compatibility และโครงสร้าง zero-code injection:
1. หยุดการช้าลงของโมเดลขนาดเล็กที่มี dynamic shapes (`0.79x -> >= 1.05x`)
2. ตัดการจอง/อ่าน-เขียน Tensor `logits` ขนาดใหญ่ `[N, V]` ด้วย **Fused Chunked Linear + CrossEntropy**
3. ทำให้โหมด Opt-in **FP8 (`sm >= 89`)** ทำงานร่วมกับ `torch.compile` ได้โดยไม่เกิด graph breaks เพื่อแข่งกับ `BF16 + Compile` (`4.16x`)
4. นำ **CUDA Graphs (`reduce-overhead`)** มาเร่ง Static-Shape Training Loops สำหรับโมเดล launch-bound (`deep_mlp`, small Transformers)
5. เพิ่ม **Auto Activation Checkpointing** (VRAM-aware OOM shield) และ **Dedicated CUDA Copy Stream H2D Prefetcher**

---

## Sub-Phases

| กลุ่ม | Sub-Phase | ชื่อ | ไฟล์รายละเอียด | Report เป้าหมาย |
| :--- | :--- | :--- | :--- | :--- |
| กลุ่ม 1 | **5.1** | Tiny Dynamic-Shape Compile Bypass | [5.1-tiny-dynamic-compile-guard.md](./phase-5/5.1-tiny-dynamic-compile-guard.md) | `report/HHMMSS_YYYYMMDD-phase-5.1.txt` |
| กลุ่ม 1 | **5.2** | Fused Chunked Linear + CrossEntropy Kernel | [5.2-fused-linear-cross-entropy.md](./phase-5/5.2-fused-linear-cross-entropy.md) | `report/HHMMSS_YYYYMMDD-phase-5.2.txt` |
| กลุ่ม 1 | **5.3** | Compilable FP8 Inference Fusion (`sm >= 89`) | [5.3-compilable-fp8-inference.md](./phase-5/5.3-compilable-fp8-inference.md) | `report/HHMMSS_YYYYMMDD-phase-5.3.txt` |
| กลุ่ม 2 | **5.4** | CUDA Graphs for Static-Shape Training Loops | [5.4-training-cuda-graphs.md](./phase-5/5.4-training-cuda-graphs.md) | `report/HHMMSS_YYYYMMDD-phase-5.4.txt` |
| กลุ่ม 2 | **5.5** | Auto Activation Checkpointing & Async H2D Prefetch | [5.5-auto-activation-ckpt-and-async-prefetch.md](./phase-5/5.5-auto-activation-ckpt-and-async-prefetch.md) | `report/HHMMSS_YYYYMMDD-phase-5.5.txt` |

---

## Exit Criteria รวมของ Phase 5
- [ ] Sub-Phase **5.1 – 5.5** ผ่าน Unit Tests, Migrate Parity, และ Benchmark Gate ครบทุกข้อ
- [ ] `python3 scripts/run_gate.py --mode fast` (CPU) และ `--mode full` (Colab GPU) ยังคง **PASS 100%**
- [ ] ไม่แก้โค้ดใน `pipline-for-test/workloads/*.py` และคง `dependencies = []` ใน `pyproject.toml`
- [ ] **[WAIT FOR USER]** ตรวจผลรายงานบน Google Colab GPU ของแต่ละ sub-phase (5.1 → 5.5) ก่อน commit / ปิดรอบ
