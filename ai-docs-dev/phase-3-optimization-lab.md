# Phase 3: [Dedicated] Optimization Lab — Experimentation & Tuning (Overview)

## เป้าหมาย (Goal)
Phase นี้แยกออกมาเฉพาะสำหรับการ **ทดลอง, วัดผล, และจูนเทคนิค Optimization ทีละ Sub-Phase** อย่างเป็นระบบ ทุกเทคนิคต้องสร้างเป็น Pass อิสระที่เปิด-ปิดได้ มี Unit Test + Migrate Test คุ้มครอง และต้องผ่าน **Promotion Gate** บน Google Colab ก่อนรวมเข้า Profile หลัก (`safe` / `balanced` / `aggressive`)

---

## Sub-Phases

| Sub-Phase | Track | ชื่อ | ไฟล์รายละเอียด | Report เป้าหมาย |
| :--- | :--- | :--- | :--- | :--- |
| **3.1** | 3A.1 | Runtime, CUDA Flags, Memory Allocator & `channels_last` | [3.1-runtime-and-memory.md](./phase-3/3.1-runtime-and-memory.md) | `report/HHMMSS_YYYYMMDD-phase-3.1.txt` |
| **3.2** | 3A.2 | DataLoader Auto-Tuning & Colab `/dev/shm` + Pickle Guard | [3.2-dataloader-autotune.md](./phase-3/3.2-dataloader-autotune.md) | `report/HHMMSS_YYYYMMDD-phase-3.2.txt` |
| **3.3** | 3B.1 | Optimizer Fusion (`fused`/`foreach`) & Safe Auto-AMP | [3.3-fused-optimizer-and-amp.md](./phase-3/3.3-fused-optimizer-and-amp.md) | `report/HHMMSS_YYYYMMDD-phase-3.3.txt` |
| **3.4** | 3B.0 | Scale-Up Bottleneck Profiling & Deep Debug Diagnostics | [3.4-scale-bottleneck-profiling.md](./phase-3/3.4-scale-bottleneck-profiling.md) | `report/HHMMSS_YYYYMMDD-phase-3.4.txt` |
| **3.5** | 3B.2 | Auto `torch.compile`, Persistent Cache & Checkpoint Transparency | [3.5-torch-compile.md](./phase-3/3.5-torch-compile.md) | `report/HHMMSS_YYYYMMDD-phase-3.5.txt` |
| **3.6** | 3C | Inference / Running Optimization (`inference_mode`, CUDA Graphs, Quant/FP8) | [3.6-inference-running.md](./phase-3/3.6-inference-running.md) | `report/HHMMSS_YYYYMMDD-phase-3.6.txt` |
| **3.7** | 3D | Custom Kernels (Triton / CUDA), Capability Gate & Final Consolidation | [3.7-custom-kernels.md](./phase-3/3.7-custom-kernels.md) | `report/HHMMSS_YYYYMMDD-phase-3.7.txt` & `phase-3.txt` |

---

## Promotion Gate มาตรฐานสำหรับทุก Sub-Phase
ทุก Experiment ใน Phase 3 ต้องผ่านครบ 4 ข้อ ก่อน Promote เข้า Profile หลัก:
1. **Unit & Kernel Tests Pass**: Unit tests + Capability gate mock tests (และ Kernel numerical parity สำหรับ 3.7) ผ่าน 100%
2. **Migrate Tests Pass**: ทุก unmodified script ใน `tests/migrate/` ผ่าน และ Checkpoint Baseline ↔ Wrapped โหลดข้ามกันได้
3. **Verified Benchmark Gain (จาก Colab Text Report)**: เร็วขึ้นหรือลด VRAM บน workload เป้าหมายจริง และไม่ regression บน workload อื่นเกินเกณฑ์
4. **Numerical Parity Pass**: Loss trajectory, grad norm, และ eval metric อยู่ในเกณฑ์ (ไม่มี `NaN`/`Inf`)
5. **[WAIT FOR USER]** ผู้ใช้ตรวจ **Detailed Text Report** ใน `report/` บน Google Colab และอนุมัติผลด้วยตนเองก่อนเริ่ม Sub-Phase ถัดไป
