# TensorRT Inference Without Ultralytics

## Motivation

The current Docker image is ~20 GB, largely due to PyTorch (~6.5 GB) and TorchVision (~600 MB).
These are required not for inference logic, but because `ultralytics` imports them unconditionally
at package initialization — even when running a pre-built `.engine` file.

Replacing Ultralytics with a direct TensorRT Python API implementation would allow removing:

| Package | Saved |
|---------|-------|
| `torch` + `torchvision` (cu128) | ~6.5 GB |
| `ultralytics` + `matplotlib` | ~450 MB |
| `onnxruntime-gpu` + `onnxslim` + `onnx` | ~550 MB |
| **Total** | **~7.5 GB** |

Estimated image after: **~5–6 GB** (down from ~20 GB).

---

## Open Source Projects

### 1. [TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO) ⭐ Best fit

- Explicitly supports **YOLO11** (v3 through v11, PP-YOLOE)
- Pure Python inference via `TRTYOLO` class — zero Ultralytics dependency
- Optimized preprocessing via CUDA kernels; inference via CUDA graphs
- Installable as a pip package: `pip install tensorrt-yolo`

```python
from tensorrt_yolo import TRTYOLO

model = TRTYOLO("yolo11x.engine", task="detect")
results = model.predict(image)
```

### 2. [TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)

- Supports YOLOv11, v10, v9, v8, v7, v6, YOLOX, v5
- Python + C++
- Includes NMS plugin support

### 3. [YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT)

- YOLOv8-focused (same anchor-free architecture as YOLOv11)
- Explicitly designed to break away from PyTorch/Ultralytics at inference time

---

## ⚠️ Engine Compatibility Warning

These projects typically require **re-exporting ONNX with their own NMS plugin**
(e.g. `EfficientNMS`), then converting to `.engine` with `trtexec`.

They are **not** guaranteed to load `.engine` files exported by Ultralytics,
because the output tensor layout differs.

### Required workflow (not compatible with current Ultralytics-exported engines)
```
.pt  →  (project's export tool)  →  .onnx (with EfficientNMS)
     →  trtexec  →  .engine  →  project's inference API
```

### Current workflow (Ultralytics)
```
.pt  →  ultralytics model.export(format='engine')  →  .engine  →  YOLO('model.engine')
```

Adopting one of these projects requires re-exporting all `.engine` files and
updating `entrypoint.py` + `utils/yolo_inference.py`.

---

## Implementation Scope (if adopted)

Files that would need to change:

| File | Change |
|------|--------|
| `entrypoint.py` | Replace `YOLO(pt_path).export(...)` with project's export tool |
| `utils/yolo_inference.py` | Replace `YOLO(engine_path)(img)` with project's inference API |
| `docker_build/Dockerfile` | Remove `torch`, `torchvision`, `ultralytics`, ONNX packages; add `tensorrt-yolo` |

Post-processing for YOLOv11 anchor-free output `[1, 4+num_classes, 8400]`:
- Decode `cx, cy, w, h` → `x1, y1, x2, y2`
- Filter by confidence threshold
- Apply NMS (either in-engine via EfficientNMS plugin, or Python-side)

See also: [`ultralytics_free_inference_feasibility.md`](ultralytics_free_inference_feasibility.md)
