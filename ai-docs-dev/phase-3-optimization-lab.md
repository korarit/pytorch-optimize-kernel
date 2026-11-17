# Phase 3: [Dedicated] Optimization Lab — Experimentation & Tuning

## เป้าหมาย (Goal)
Phase นี้แยกออกมาเฉพาะสำหรับการ **ทดลอง, วัดผล, และจูนเทคนิค Optimization ทีละเรื่อง** อย่างเป็นระบบ ทุกเทคนิคต้องสร้างเป็น Pass อิสระที่เปิด-ปิดได้ และต้องผ่าน **Promotion Gate** ก่อนถูกนำไปเปิดเป็นค่าเริ่มต้นใน Profile (`safe` / `balanced` / `aggressive`)

---

## โครงสร้างการทดลอง (4 Experiment Tracks)

### Track 3A: Runtime, Memory Allocator & Data Pipeline
- [ ] **CUDA / Math Flags**: `allow_tf32`, `float32_matmul_precision`, `cudnn.benchmark`
- [ ] **Memory Tuning**: `PYTORCH_CUDA_ALLOC_CONF` (`expandable_segments:True`), `channels_last` memory format (สำหรับ 4D vision tensors)
- [ ] **Host / CPU Tuning**: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `LD_PRELOAD` (`jemalloc`/`tcmalloc` ถ้ามี)
- [ ] **DataLoader Auto-Tuning**:
  - ทดลองปรับ `pin_memory=True`, `persistent_workers=True`, `prefetch_factor`, adaptive `num_workers`
  - **Safety Guard**: ตรวจ `/dev/shm` ก่อนเพิ่ม `num_workers`, เช็กว่า dataset pickle ได้ และ fallback กลับค่าเดิมถ้า DataLoader worker crash

### Track 3B: Training Graph, Optimizer Fusion & Mixed Precision
- [ ] **Optimizer Fusion**: Auto-inject `fused=True` (หรือ `foreach=True`) ใน `AdamW`, `Adam`, `SGD` (พร้อมเช็ก device & dtype support)
- [ ] **Safe Auto-AMP**:
  - เลือก `bf16` บน GPU `sm >= 80` และ `fp16 + GradScaler` บน `sm < 80`
  - ข้ามอัตโนมัติถ้า user โค้ดใช้ `autocast` หรือ `GradScaler` อยู่แล้ว
  - ทดลอง loss-scaling / numerical edge cases (เช่น CrossEntropy / LayerNorm precision)
- [ ] **Auto `torch.compile`**:
  - ทดลอง mode `default`, `reduce-overhead`, `max-autotune`
  - ตั้ง persistent cache (`TORCHINDUCTOR_CACHE_DIR`) และวัดเทียบทั้ง Cold Run vs Warm Run
  - จัดการ `state_dict` unwrap/proxy เพื่อไม่ให้ `_orig_mod.` ทำ Checkpoint Migrate Test พัง
  - Dynamic shape / graph break guard: ถ้า compile fail หรือ recompile บ่อยเกิน threshold ให้ fallback gracefully

### Track 3C: Inference / Running Optimization
- [ ] Auto-wrap `torch.inference_mode()` เมื่อตรวจพบ eval/inference path (ไม่ได้ขอ gradients)
- [ ] ทดลอง `torch.compile(mode="reduce-overhead")` / **CUDA Graphs** สำหรับ static-shape inference
- [ ] ทดลอง dynamic quantization / FP8 hooks (เฉพาะบน GPU architecture ที่รองรับ)

### Track 3D: Custom Kernels (Triton / CUDA)
- [ ] พัฒนา custom kernels ทดแทน ops ที่เป็นคอขวด (เช่น `RMSNorm`, `LayerNorm`, `Fused CrossEntropy`, `RoPE`, `SwiGLU`)
- [ ] สร้าง **Kernel Registry & Capability Gate**:
  - ระบุเงื่อนไขขั้นต่ำต่อ kernel (`min_sm`, `supported_dtypes`, `min_torch_version`, `triton_required`)
  - ระบบ monkey-patch อัตโนมัติเฉพาะเมื่อ capability ผ่าน; ถ้าไม่ผ่าน fallback ไปใช้ PyTorch op เดิม
- [ ] **Kernel Unit Tests (`tests/unit/test_kernels_*.py`)**:
  - ทดสอบ Forward + Backward gradient parity เทียบกับ PyTorch reference ในหลาย shape (รวม non-power-of-2, edge lengths) และหลาย dtype (`fp32`, `fp16`, `bf16`)
  - Micro-benchmark ราย kernel (`ms`, `GB/s`, `TFLOPS`) เทียบกับ PyTorch native op

---

## Promotion Gate & Experiment Reporting (เกณฑ์ก่อนนำผลทดลองเข้า Profile หลัก)

ทุก Experiment ใน Phase 3 ต้องรันผ่าน script ที่สร้าง **Detailed Text Report** เก็บลง `report/HHMMSS_YYYYMMDD-phase-3-track-<X>.txt` เพื่อส่งให้ผู้ใช้ตรวจบน Google Colab และต้องผ่านครบ 4 ข้อนี้ก่อน Promote:

1. **Unit & Kernel Tests Pass**: Unit tests + Kernel forward/backward parity + Mock fallback tests ผ่าน 100%
2. **Migrate Tests Pass**: ทุก script ใน `tests/migrate/` รันผ่าน และ Checkpoint Baseline ↔ Wrapped โหลดข้ามกันได้
3. **Verified Benchmark Gain (จาก Colab Text Report)**:
   - มี speedup หรือ VRAM reduction ชัดเจนบน workload เป้าหมายอย่างน้อย 1 กลุ่มใน `pipline-for-test/`
   - ไม่ทำให้ workload อื่นช้าลงเกิน threshold โดยไม่ได้ตั้งใจ (ตรวจสอบ Cold vs Warm run ด้วย)
4. **Numerical Parity Pass**: Loss trajectory, grad norm, และ eval metric อยู่ในเกณฑ์ Parity ที่กำหนดใน Phase 0 (ไม่มี `NaN`/`Inf`)

---

## Exit Criteria & Human Verification Gate (เกณฑ์ผ่าน Phase 3)
- [ ] มีชุด optimization passes ที่ผ่าน Promotion Gate ครบและจัดเข้า `safe` / `balanced` / `aggressive` profiles แล้ว
- [ ] มี script สร้าง **Detailed Text Report (`report/HHMMSS_YYYYMMDD-phase-3.txt`)** เปรียบเทียบ Baseline vs `safe` vs `balanced` vs `aggressive` ครบทุก workload (รวม Micro-benchmark ของ Custom Kernels และ Fallback Log)
- [ ] **[WAIT FOR USER]** รอผู้ใช้รันแต่ละ Track และ Final Comparison บน Google Colab, ตรวจ **Detailed Text Report** ใน `report/`, และยืนยันผลการ Promote ด้วยตนเองก่อนเริ่ม Phase 4
