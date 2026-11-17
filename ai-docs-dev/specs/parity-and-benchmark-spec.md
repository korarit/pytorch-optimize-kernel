# Parity, Benchmark & Text Report Standard v1

เอกสารนี้กำหนดมาตรฐานกลางสำหรับ:
1. เกณฑ์ **Numerical Parity**
2. กฎการวัด **Benchmark** และ **Telemetry JSON Schema**
3. รูปแบบ **Detailed Text Report (`report/HHMMSS_YYYYMMDD-<name>.txt`)**

---

## 1. เกณฑ์ Numerical Parity (Correctness Thresholds)

การเปรียบเทียบ Baseline vs Wrapped ต้องใช้ `--seed` เดียวกันและข้อมูลลำดับเดียวกันบน device เดียวกัน:

| Profile | Non-finite (`NaN`/`Inf`) | Training Loss Relative Diff (Mean / Max) | Eval / Inference Output Max Abs Diff |
| :--- | :--- | :--- | :--- |
| **`safe`** | ห้ามเกิด (0 tolerance) | `<= 1e-2` (เผื่อ CUDA Conv2d/BatchNorm backward atomicAdd noise ข้าม 20+ steps) | `<= 1e-2` |
| **`balanced`** | ห้ามเกิด (0 tolerance) | `<= 2e-2` (เผื่อ `bf16` / `fp16` mixed precision) | `<= 2e-2` |
| **`aggressive`** | ห้ามเกิด (0 tolerance) | `<= 3e-2` (รวม Custom Kernels / FP8 / `max-autotune`) | `<= 3e-2` |

> หมายเหตุ: สำหรับ `fp16` / `bf16` หากเทียบ **Loss Trajectory Slope** (ทิศทางการลดลงของ loss) แล้วลู่เข้าใกล้เคียงกันโดยไม่ diverge และไม่มี `NaN`/`Inf` ให้บันทึกทั้ง `max_abs_diff` และ `max_rel_diff` ในรายงานเพื่อประกอบการตรวจของมนุษย์

---

## 2. กฎการวัด Benchmark (GPU & Subprocess Rules)

1. **Subprocess Isolation**: Baseline และ Wrapped แต่ละโหมดต้องรันใน Python subprocess แยกกัน เพื่อไม่ให้ CUDA caching allocator, cuDNN autotuner, หรือ Dynamo/Inductor state ปนกัน
2. **GPU Synchronization**: การจับเวลาบน CUDA ต้องใช้ `torch.cuda.Event(enable_timing=True)` หรือเรียก `torch.cuda.synchronize()` ก่อนเริ่มและก่อนจบช่วงจับเวลาทุกครั้ง
3. **Warmup vs Steady-State**:
   - ตัดช่วง warmup (เช่น `warmup_steps = max(3, int(0.2 * total_steps))`) ออกจาก steady-state statistics
   - รายงานแยกทั้ง `Steady Step Time (Median, P95, Min, Std)` และ `End-to-End Wall Time`
4. **Cold Cache vs Warm Cache**:
   - สำหรับโหมดที่เปิด `torch.compile` ต้องแยกวัด **Cold Run** (ล้าง `TORCHINDUCTOR_CACHE_DIR`) และ **Warm Run** (ใช้ cache จากรอบก่อน)

---

## 3. Telemetry JSON Schema (`telemetry_schema_version: 1`)

```json
{
  "telemetry_schema_version": 1,
  "run_id": "20261117_154500_transformer_lm_balanced",
  "mode": "balanced",
  "cache_state": "warm",
  "exit_code": 0,
  "end_to_end_sec": 12.48,
  "warmup_steps": 5,
  "steady_steps": 25,
  "step_time_ms": {
    "median": 42.15,
    "p95": 45.80,
    "min": 40.90,
    "std": 1.62
  },
  "throughput": {
    "value": 1518.4,
    "unit": "samples/sec"
  },
  "memory_mb": {
    "peak_allocated": 1420.5,
    "peak_reserved": 1664.0
  },
  "losses": [2.41, 2.35, 2.28],
  "fallback_events": [
    {
      "pass": "kernels.rmsnorm",
      "status": "SKIPPED",
      "reason": "sm_75 < min_sm 80"
    }
  ]
}
```

---

## 4. มาตรฐาน Detailed Text Report (`report/HHMMSS_YYYYMMDD-<name>.txt`)

ทุก Phase และ Sub-Phase ต้องสร้างรายงาน Text File โดยยึดมาตรฐานนี้:

- **โฟลเดอร์เก็บ**: `report/` เท่านั้น (สร้างอัตโนมัติหากยังไม่มี)
- **รูปแบบชื่อไฟล์**: `HHMMSS_YYYYMMDD-<name>.txt` (เวลา-วันที่ท้องถิ่นที่รัน เช่น `report/154512_20261117-phase-0-env.txt`)
- **Output**: เขียนลงไฟล์ `.txt` พร้อมพิมพ์เนื้อหาหรือสรุปออก `stdout` (เพื่อให้ผู้ใช้ตรวจได้ทันทีใน cell ของ Google Colab)
- **Sections บังคับในทุก Report**:
  1. `[HEADER]` — Phase/Sub-Phase, Timestamp, Git Commit/Branch, Host/Colab Tag
  2. `[HARDWARE & ENVIRONMENT]` — GPU, `sm_XX`, VRAM, CPU Arch/Cores, RAM, `/dev/shm`, Python/PyTorch/CUDA/Triton
  3. `[CONFIG & RULE TRACE]` — *(ตั้งแต่ Phase 1 เป็นต้นไป)* Profile, Matched Rules, Final Resolved Config
  4. `[TEST RESULTS]` — *(ตั้งแต่ Phase 1 เป็นต้นไป)* Unit Tests, Migrate Tests, Kernel Parity (PASS/FAIL/SKIP + failure logs)
  5. `[BENCHMARK COMPARISON]` — *(ตั้งแต่ Phase 2 เป็นต้นไป)* ตารางเทียบ Baseline vs Wrapped (Cold/Warm, Median/P95 ms, Speedup, VRAM, Loss Diff, Status)
  6. `[FALLBACK & WARNINGS]` — รายการ pass ที่ถูก skip/fallback พร้อมเหตุผล
  7. `[VERIFICATION STATUS]` — แสดง `STATUS: PENDING_USER_REVIEW (Google Colab Manual Check Required)` เสมอจนกว่าผู้ใช้จะตรวจผ่าน
