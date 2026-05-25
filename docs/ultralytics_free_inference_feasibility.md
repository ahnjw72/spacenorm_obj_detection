# Feasibility Analysis: TensorRT Export and Inference Without Ultralytics

## Background

Ultralytics YOLO11 is licensed under **AGPL-3.0** for open-source use. Commercial deployment
of a networked service (e.g., a CCTV occupancy detection service that reports to a paid API)
requires either:

- **Open-sourcing all service code** under AGPL-3.0, or
- Purchasing an **Ultralytics Enterprise license** (pricing not publicly listed; contact
  [https://ultralytics.com/license](https://ultralytics.com/license))

The current implementation uses `from ultralytics import YOLO` at both export time and inference
time, so both stages are subject to this requirement.

This document analyses the feasibility of removing the Ultralytics runtime dependency so that
the production inference service does not use Ultralytics code.

---

## Pipeline Overview

```
yolo11x_set01-0148.pt
        │
        │  Stage 1: PT → ONNX  (one-time, offline)
        ▼
yolo11x_set01-0148.onnx
        │
        │  Stage 2: ONNX → TRT engine  (one-time, per GPU arch)
        ▼
yolo11x_set01-0148.engine
        │
        │  Stage 3: Runtime inference
        │    3a. Pre-processing
        │    3b. TRT forward pass
        │    3c. Post-processing (bbox decode + NMS)
        ▼
boxes, confidences, classes
```

---

## Stage-by-Stage Analysis

### Stage 1: PT → ONNX Export

**Ultralytics dependency: unavoidable at this stage.**

The `.pt` file contains Ultralytics model architecture class definitions baked into the
checkpoint via Python's `pickle`. Loading it requires the Ultralytics package to be present
so that Python can reconstruct the model object:

```python
# This implicitly requires Ultralytics to define the model classes
import torch
model = torch.load('yolo11x_set01-0148.pt')   # fails without ultralytics installed
```

To avoid Ultralytics entirely at export time, the YOLOv11 architecture would need to be
re-implemented from scratch in pure PyTorch — not practical for a production project.

**Pragmatic conclusion:** Accept Ultralytics for the one-time, offline export step. Focus
on eliminating it from the runtime inference service, where the AGPL-3.0 network service
clause applies.

---

### Stage 2: ONNX → TRT Engine

**No Ultralytics needed. Easy.**

NVIDIA's `trtexec` CLI tool (ships with TensorRT) handles this step entirely:

```bash
trtexec \
    --onnx=yolo11x_set01-0148.onnx \
    --saveEngine=yolo11x_set01-0148.engine \
    --fp16 \
    --workspace=4096
```

Alternatively, the TensorRT Python API can be used programmatically. The resulting
`.engine` file contains only compiled GPU kernels — no Ultralytics code.

---

### Stage 3a: Pre-processing

**No Ultralytics needed. Easy.**

Standard OpenCV + NumPy operations:

```python
import cv2
import numpy as np

def preprocess(img_bgr: np.ndarray, imgsz: int = 640):
    img = cv2.resize(img_bgr, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0          # normalise to [0, 1]
    img = img.transpose(2, 0, 1)[np.newaxis]      # HWC → NCHW
    return np.ascontiguousarray(img)
```

---

### Stage 3b: TRT Forward Pass

**No Ultralytics needed. Medium effort.**

The TensorRT Python API (`tensorrt` + `pycuda`) runs inference directly on the engine:

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TRTInferencer:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        # Allocate host + device buffers for each binding
        self.bindings, self.host_inputs, self.host_outputs, \
            self.cuda_inputs, self.cuda_outputs = self._allocate_buffers()

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        np.copyto(self.host_inputs[0], input_array.ravel())
        cuda.memcpy_htod(self.cuda_inputs[0], self.host_inputs[0])
        self.context.execute_v2(self.bindings)
        cuda.memcpy_dtoh(self.host_outputs[0], self.cuda_outputs[0])
        return self.host_outputs[0]
```

The buffer allocation boilerplate is ~50 lines of well-documented code; NVIDIA provides
complete examples in the TensorRT Developer Guide.

---

### Stage 3c: Post-processing (BBox Decode + NMS)

**No Ultralytics needed. High effort — the most complex stage.**

YOLOv11 uses an **anchor-free** detection head. The raw engine output has shape
`[1, 84, 8400]` (4 box coordinates + 80 class scores × 8400 candidate detections).

A complete implementation:

```python
import cv2
import numpy as np

def postprocess(
    raw_output: np.ndarray,
    orig_shape: tuple,          # (H, W) of the original image
    imgsz: int = 640,
    conf_thresh: float = 0.6,
    iou_thresh: float = 0.45,
    target_class: int = 0,      # 0 = person
):
    # raw_output shape: [1, 84, 8400]
    preds = raw_output[0].T     # → [8400, 84]

    boxes_cxcywh = preds[:, :4]
    scores       = preds[:, 4:]

    # Class confidence = max score across all classes
    class_ids  = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    # Filter by confidence and target class
    mask = (confidences >= conf_thresh) & (class_ids == target_class)
    boxes_cxcywh = boxes_cxcywh[mask]
    confidences  = confidences[mask]
    class_ids    = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return [], [], []

    # cx,cy,w,h → x1,y1,x2,y2  (still in imgsz coordinates)
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    # Scale back to original image size
    scale_x = orig_shape[1] / imgsz
    scale_y = orig_shape[0] / imgsz
    x1, x2 = x1 * scale_x, x2 * scale_x
    y1, y2 = y1 * scale_y, y2 * scale_y

    # NMS via OpenCV (Apache 2.0 license — no commercial restriction)
    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(),
        confidences.tolist(),
        conf_thresh,
        iou_thresh,
    )
    indices = indices.flatten() if len(indices) > 0 else []

    final_boxes = [[int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])] for i in indices]
    final_confs  = [float(confidences[i]) for i in indices]
    final_classes = [int(class_ids[i]) for i in indices]

    return final_boxes, final_confs, final_classes
```

The output tensor format must be verified against the actual engine output by comparing
results with the Ultralytics implementation on the same test images before switching.

---

## Summary

| Stage | Ultralytics required | Effort | Key dependency |
|-------|---------------------|--------|----------------|
| PT → ONNX | **Yes** (unavoidable) | — | `ultralytics` |
| ONNX → TRT engine | No | Low | `trtexec` (NVIDIA) |
| Pre-processing | No | Low | `opencv-python`, `numpy` |
| TRT forward pass | No | Medium | `tensorrt`, `pycuda` |
| Post-processing | No | **High** | `opencv-python` (NMS) |

**Total estimated effort to replace runtime Ultralytics:** 3–5 engineering days for
implementation, plus testing time to validate output parity.

---

## License Status After Replacement

| Component | Used in production service | License |
|-----------|---------------------------|---------|
| `ultralytics` | Export only (one-time, offline) | AGPL-3.0 — legal risk reduced |
| `tensorrt` | Yes | [NVIDIA TensorRT EULA](https://docs.nvidia.com/deeplearning/tensorrt/latest/general/license.html) — commercial use permitted |
| `pycuda` | Yes | MIT — commercial use permitted |
| `opencv-python` | Yes | Apache 2.0 — commercial use permitted |
| `numpy` | Yes | BSD — commercial use permitted |

Whether the one-time use of Ultralytics for the offline export step still triggers the
AGPL-3.0 obligation is a legal question. For a definitive answer, consult a lawyer or
contact Ultralytics directly.

---

## Recommendation

1. **Short term:** Purchase an Ultralytics Enterprise license. Cheaper and faster than
   re-implementing post-processing; eliminates all license uncertainty.

2. **Medium term (if commercial scale justifies it):** Implement the Ultralytics-free
   inference path described above, keeping Ultralytics only for the offline export step.
   Validate output parity thoroughly before switching production workloads.

3. **Long term (if avoiding Ultralytics entirely):** Evaluate Apache 2.0 licensed
   alternatives (RT-DETR, YOLO-NAS, Detectron2) for new model training, which removes
   the export-time dependency as well.
