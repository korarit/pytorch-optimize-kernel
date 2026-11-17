# Config Schema v1 Specification (`schema_version: 1`)

เอกสารนี้กำหนดโครงสร้างมาตรฐานของ Resolved Configuration สำหรับ `ptopt` (`kernel-pytouch-optimize`) เพื่อใช้ใน Phase 1–4

---

## 1. โครงสร้าง Config (`OptimizeConfig`)

```json
{
  "schema_version": 1,
  "profile": "balanced",
  "runtime": {
    "tf32": false,
    "cudnn_benchmark": false,
    "alloc_conf": "expandable_segments:True",
    "omp_threads": 0,
    "jemalloc": false
  },
  "amp": {
    "enabled": false,
    "dtype": "fp32",
    "grad_scaler": false
  },
  "compile": {
    "enabled": false,
    "mode": "default",
    "dynamic": false,
    "cache_dir": ".torch_inductor_cache"
  },
  "optimizer": {
    "fused": false,
    "foreach": true
  },
  "attention": {
    "flash_sdp": false,
    "mem_efficient_sdp": true,
    "math_sdp": true
  },
  "dataloader": {
    "auto_tune": false,
    "pin_memory": false,
    "persistent_workers": false,
    "num_workers": null,
    "prefetch_factor": 2
  },
  "inference": {
    "inference_mode": false,
    "cuda_graphs": false
  },
  "kernels": {
    "enabled": false,
    "allow_list": [],
    "fallback_on_error": true
  }
}
```

---

## 2. รายละเอียด Fields และ Validation Rules

### `runtime`
| Field | Type | Allowed / Range | คำอธิบาย |
| :--- | :--- | :--- | :--- |
| `tf32` | `bool` | `true` / `false` | เปิด `allow_tf32` และ `float32_matmul_precision("high")` (ต้องเป็น CUDA `sm >= 80`) |
| `cudnn_benchmark` | `bool` | `true` / `false` | เปิด `torch.backends.cudnn.benchmark` (เหมาะกับ static input shapes) |
| `alloc_conf` | `str \| null` | เช่น `"expandable_segments:True"` | ค่าสำหรับ `PYTORCH_CUDA_ALLOC_CONF` (เฉพาะ backend CUDA) |
| `omp_threads` | `int` | `>= 0` (`0` = auto/unchanged) | ค่า `OMP_NUM_THREADS` / `MKL_NUM_THREADS` |
| `jemalloc` | `bool` | `true` / `false` | preload `libjemalloc.so` ถ้าพบในระบบ |

### `amp`
| Field | Type | Allowed / Range | คำอธิบาย |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | `true` / `false` | เปิด Auto Mixed Precision (ข้ามอัตโนมัติถ้า user เปิดเองอยู่แล้ว) |
| `dtype` | `str` | `"fp32"`, `"fp16"`, `"bf16"` | ชนิด precision; `bf16` ใช้ได้เมื่อ GPU รองรับ (`sm >= 80` หรือ CPU BF16) |
| `grad_scaler` | `bool` | `true` / `false` | ใช้ `GradScaler` (เปิดเฉพาะเมื่อ `enabled=true` และ `dtype="fp16"`) |

### `compile`
| Field | Type | Allowed / Range | คำอธิบาย |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | `true` / `false` | เปิด auto `torch.compile` |
| `mode` | `str` | `"default"`, `"reduce-overhead"`, `"max-autotune"` | โหมดของ `torch.compile` |
| `dynamic` | `bool \| null` | `true`, `false`, `null` (auto) | รองรับ dynamic shapes |
| `cache_dir` | `str` | path string | ตำแหน่ง `TORCHINDUCTOR_CACHE_DIR` |

### `optimizer`
| Field | Type | Allowed / Range | คำอธิบาย |
| :--- | :--- | :--- | :--- |
| `fused` | `bool` | `true` / `false` | ฉีด `fused=True` สำหรับ `Adam`/`AdamW`/`SGD` บน CUDA |
| `foreach` | `bool` | `true` / `false` | ฉีด `foreach=True` เมื่อไม่ใช้ `fused` |

### `attention`
| Field | Type | Allowed / Range | คำอธิบาย |
| :--- | :--- | :--- | :--- |
| `flash_sdp` | `bool` | `true` / `false` | เปิด FlashAttention SDPA (`sm >= 80` + fp16/bf16) |
| `mem_efficient_sdp` | `bool` | `true` / `false` | เปิด Memory-Efficient SDPA (รองรับ Turing `sm_75` ขึ้นไป) |
| `math_sdp` | `bool` | `true` / `false` | Fallback math SDPA (ควรเปิดไว้เสมอเพื่อความปลอดภัย) |

### `dataloader`
| Field | Type | Allowed / Range | คำอธิบาย |
| :--- | :--- | :--- | :--- |
| `auto_tune` | `bool` | `true` / `false` | เปิด auto-tuning ให้ `DataLoader.__init__` (ต้องผ่าน `/dev/shm` และ pickle guard) |
| `pin_memory` | `bool` | `true` / `false` | เปิด `pin_memory=True` เมื่อ target device เป็น CUDA |
| `persistent_workers` | `bool` | `true` / `false` | เปิดเมื่อ `num_workers > 0` |
| `num_workers` | `int \| null` | `>= 0` หรือ `null` | `null` = ใช้ค่าเดิมหรือคำนวณจาก CPU cores + `/dev/shm` cap |
| `prefetch_factor` | `int \| null` | `>= 1` หรือ `null` | ใช้เมื่อ `num_workers > 0` |

### `inference`
| Field | Type | Allowed / Range | คำอธิบาย |
| :--- | :--- | :--- | :--- |
| `inference_mode` | `bool` | `true` / `false` | ครอบ `torch.inference_mode()` เมื่ออยู่ใน eval path ที่ไม่ขอ grad |
| `cuda_graphs` | `bool` | `true` / `false` | ใช้ CUDA Graphs / `reduce-overhead` สำหรับ static-shape inference |

### `kernels`
| Field | Type | Allowed / Range | คำอธิบาย |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | `true` / `false` | เปิดใช้งาน Custom Triton/CUDA Kernels |
| `allow_list` | `list[str]` | เช่น `["rmsnorm", "cross_entropy"]` | ว่าง = ใช้ทุก kernel ที่ผ่าน capability gate |
| `fallback_on_error` | `bool` | `true` / `false` | ถ้า kernel ไม่รองรับหรือ error ให้ fallback ไปใช้ PyTorch op เดิม |

---

## 3. Cross-Field Validation Constraints
1. หาก `backend == "cpu"`: บังคับ `runtime.tf32 = false`, `runtime.cudnn_benchmark = false`, `runtime.alloc_conf = null`, `optimizer.fused = false`, `inference.cuda_graphs = false`, `dataloader.pin_memory = false`
2. หาก `amp.dtype == "bf16"`: `amp.grad_scaler` ต้องเป็น `false`
3. หาก `amp.enabled == false`: `amp.grad_scaler` ต้องเป็น `false`
4. หาก `dataloader.num_workers == 0`: `dataloader.persistent_workers` ต้องเป็น `false` และ `prefetch_factor = null`

---

## 4. ลำดับชั้นการ Merge Config (Precedence Order)

เรียงจากความสำคัญต่ำสุดไปสูงสุด (ชั้นบนเขียนทับเฉพาะ field ที่ระบุ):

```text
1. base              -> ค่าปลอดภัยเริ่มต้น (ปิดฟีเจอร์เสี่ยงทั้งหมด)
2. host_arch         -> ปรับตาม OS, CPU arch (x86_64/aarch64/mps), /dev/shm guard, Colab runtime
3. gpu_arch          -> ปรับตาม backend + sm_XX (เช่น sm_75 Turing vs sm_80 Ampere vs sm_89 Ada vs sm_90 Hopper)
4. vram_tier         -> ปรับตาม VRAM (<12GB, 12-24GB, >=40GB)
5. user_profile      -> safe | balanced | aggressive
6. explicit_override -> CLI flags (--no-compile, --amp bf16) และ Environment Variables (PTOPT_*)
```

---

## 5. ตัวอย่าง Resolved Config บน Google Colab (`profile = balanced`)

| Field | Colab T4 (`sm_75`, 15GB) | Colab L4 (`sm_89`, 22.5GB) | Colab A100 (`sm_80`, 40GB) | Colab CPU-only |
| :--- | :--- | :--- | :--- | :--- |
| `runtime.tf32` | `false` | `true` | `true` | `false` |
| `amp.enabled / dtype` | `true / fp16` | `true / bf16` | `true / bf16` | `false / fp32` |
| `amp.grad_scaler` | `true` | `false` | `false` | `false` |
| `attention.flash_sdp` | `false` | `true` | `true` | `false` |
| `attention.mem_efficient_sdp` | `true` | `true` | `true` | `false` |
| `optimizer.fused` | `true` | `true` | `true` | `false` |
| `compile.enabled / mode` | `true / default` | `true / default` | `true / default` | `false` |
| `kernels.enabled` | `false` (เฉพาะ `aggressive` ถ้าผ่าน gate) | `false` (เปิดใน `aggressive`) | `false` (เปิดใน `aggressive`) | `false` |
