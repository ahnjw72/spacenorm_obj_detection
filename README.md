# spacenorm_obj_detection

Real-time human occupancy detection service for CCTV RTSP streams, powered by [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics). Detects people in each camera frame, applies ROI masking and background subtraction, and reports occupancy counts to the Spacenorm API via REST and MQTT.

This project is an inference-only service refactored from `spacenorm_obj_detectionv7`. Training and file-based (video/image/NVR) inference are out of scope — see `offline_spacenorm_obj_detection.py` for offline batch video processing.

---

## Features

- Multi-threaded RTSP stream processing (one thread per camera)
- Ultralytics YOLO inference with configurable model and resolution
- **TensorRT engine export and inference** — automatic per-node engine build and caching for ~3× faster inference
- Polygon ROI masking, small-object filtering, MOG2 background subtraction
- Hysteresis to suppress false-positive flicker
- Occupancy reporting via Spacenorm REST API and MQTT
- Web UI live stream (Flask, port 8081)
- Prometheus metrics (port 9000)
- Docker Swarm deployment on edge GPU nodes
- Offline batch video processing (`offline_spacenorm_obj_detection.py`)

---

## TensorRT Engine Inference

### Why TensorRT

Running inference with a TensorRT `.engine` file instead of a PyTorch `.pt` file
is approximately **3× faster** on NVIDIA GPUs with Tensor Cores. The speed gain
comes from the TRT forward pass — Ultralytics handles pre/post-processing in both
cases.

| Mode | RTX 3070 latency (approx.) |
|------|---------------------------|
| `.pt` (PyTorch FP32) | ~25–35 ms / frame |
| `.engine` (TensorRT FP16) | ~8–12 ms / frame |

The accuracy difference between FP32 and FP16 is negligible (< 0.1% mAP).

### Automatic Engine Build on First Start (`entrypoint.py`)

The container entrypoint (`entrypoint.py`) handles engine management automatically:

1. Computes the **SHA256 hash** of the `.pt` model file
2. Derives a unique engine filename: `<stem>_<hash12>_imgsz<N>_<fp16|fp32>.engine`
3. If the engine file does not exist in `SPACENORM_ENGINE_CACHE_DIR` → exports it
   (takes ~2–5 minutes on first run per node)
4. If the engine already exists → skips export (subsequent starts are near-instant)
5. Launches the main service with `--model <engine_path>`

The SHA256 hash in the filename means the engine is **automatically rebuilt** when
the model weights change — even if the filename stays the same.

### Engine Cache Directory

Engines are stored in a host-mounted bind volume so they persist across container
restarts and are shared between the main service and the offline service running on
the same node:

```
/var/lib/spacenorm_obj_detection/
└── yolo11x_set01-0148_9db5996b9201_imgsz640_fp16.engine
```

One-time setup on each worker node:

```bash
sudo mkdir -p /var/lib/spacenorm_obj_detection
sudo chmod 777 /var/lib/spacenorm_obj_detection   # or chown to the container user
```

### TensorRT Environment Variables

Configured in `docker_swarm/stack.yml.template`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SPACENORM_MODEL_PT` | `yolo11x_set01-0148.pt` | Source `.pt` file path inside the container |
| `SPACENORM_ENGINE_CACHE_DIR` | `/var/lib/spacenorm_obj_detection` | Host-mounted engine cache directory |
| `SPACENORM_IMGSZ` | `640` | Image size used when building the engine |
| `SPACENORM_TRT_HALF` | `true` | FP16 mode (`true` / `false`) |
| `SPACENORM_DEVICE` | `0` | GPU device index |
| `SPACENORM_TRT_WORKSPACE` | `4` | TensorRT workspace size in GiB |

### Manual Engine Export (`export_trt.py`)

To export a `.engine` file manually outside the container:

```bash
python export_trt.py \
    --model /path/to/yolo11x_set01-0148.pt \
    --imgsz 640 \
    --half \
    --device 0 \
    --workspace 4
```

The engine is written next to the `.pt` file. To use it in the service, update
`SPACENORM_MODEL_PT` or the `model` field in `default.json`.

### GPU Architecture Compatibility

A TensorRT engine is compiled for a **specific GPU architecture** (e.g., Ampere,
Ada, Hopper) and cannot be shared between nodes with different GPU models.
The per-node caching in `entrypoint.py` handles this automatically — each node
builds its own engine on first start.

---

## Offline Batch Video Processing

`offline_spacenorm_obj_detection.py` processes pre-recorded video files using the
same detection pipeline as the live service. It also uses a TensorRT engine via
`offline_entrypoint.py`.

### How It Works

Watches a `todos/` directory for task folders. Each folder must contain:
- A video file (`.mp4`, `.avi`, `.mov`, `.mkv`)
- A `start` flag file (triggers processing)
- An optional `settings.json` with `ROI` and `min_obj_size_ratio`

Results are written to `done/{id}/`:
- Snapshot JPEG files for frames with detections
- `result.json` with per-frame bounding-box data
- A `done` flag file when processing is complete

### Running Locally

```bash
python -m spacenorm_obj_detection.offline_spacenorm_obj_detection \
    --common_config ./spacenorm_cfg/offline_ai_detector/offline_spacenorm_obj_detection.json
```

To override the model path (e.g., use a pre-built engine):

```bash
python -m spacenorm_obj_detection.offline_spacenorm_obj_detection \
    --common_config ./spacenorm_cfg/offline_ai_detector/offline_spacenorm_obj_detection.json \
    --model /var/lib/spacenorm_obj_detection/<engine_filename>.engine
```

---

## Quick Start

### Prerequisites

```bash
pip install ultralytics flask paho-mqtt prometheus-client
```

### Run Locally (PyTorch `.pt` — no TensorRT required)

```bash
SPACENORM_SERVER_ID=cym \
SPACENORM_DEFAULT_CFG_FILE=./spacenorm_cfg/behavior/default.json \
SPACENORM_SERVER_CFG_FILE=./spacenorm_cfg/behavior/overrides/cym.json \
SPACENORM_SENSOR_CFG_FILE=./spacenorm_cfg/cctv/cctv_cym.json \
SPACENORM_DEVICEKEY_FILE=./spacenorm_cfg/cctv/cctv_cym.keys \
python -u -m spacenorm_obj_detection.spacenorm_obj_detection
```

---

## Detection Pipeline

```
RTSP Stream → GStreamer Frame Queue
    → Ultralytics YOLO inference (.pt or .engine)
    → filter_only_person()        # keep person class only
    → filter_small_objects()      # drop boxes below size threshold
    → remove_outside_ROI()        # polygon ROI masking per camera
    → check_bb_on_background()    # MOG2 background subtraction
    → Hysteresis                  # N consecutive frames required
    → Spacenorm API report + MQTT publish + Web UI frame
```

**Background detection types (kept as real detections):**
- Type 0 — high confidence, high motion
- Type -1 — medium confidence, medium motion
- Type -2 — high confidence, low motion

Types 1, 2, 3 are filtered as background.

---

## Project Structure

```
spacenorm_obj_detection/
├── entrypoint.py                       # Container entrypoint: TRT engine build + launch main service
├── offline_entrypoint.py               # Container entrypoint: TRT engine build + launch offline service
├── offline_spacenorm_obj_detection.py  # Offline batch video processing
├── export_trt.py                       # Manual TRT engine export script
├── spacenorm_obj_detection/
│   ├── spacenorm_obj_detection.py      # Main service (Flask, RTSP, MQTT)
│   ├── __init__.py
│   └── utils/
│       ├── yolo_inference.py           # Ultralytics YOLO wrapper (.pt and .engine)
│       ├── post_processing.py          # ROI, background, size filtering
│       ├── cctv_camera.py              # RTSP stream handling (GStreamer)
│       ├── config_loader.py            # Hierarchical config loading
│       ├── config_manager.py           # Simple JSON config + CLI override loader
│       ├── spacenorm_api.py            # Spacenorm REST API client
│       ├── kakao_messaging.py          # Kakao Talk alerts
│       ├── visualization.py            # Bounding box drawing
│       ├── display.py                  # FPS overlay
│       ├── yolo_classes.py             # Class index definitions
│       └── gateway_api.py              # Gateway API client
├── spacenorm_cfg/
│   ├── behavior/
│   │   ├── default.json                # Global defaults
│   │   └── overrides/                  # Per-server overrides
│   ├── cctv/
│   │   ├── cctv_<server>.json          # Camera definitions (RTSP URIs, ROI)
│   │   └── cctv_<server>.keys          # Device keys
│   └── offline_ai_detector/
│       └── offline_spacenorm_obj_detection.json  # Offline service config
├── docker_build/                       # Dockerfile and build scripts
├── docker_swarm/                       # Docker Swarm stack templates and deploy scripts
├── docs/
│   ├── licensing.md                    # AGPL-3.0 compliance guide
│   └── ultralytics_free_inference_feasibility.md  # TRT-only inference analysis
└── templates/ static/                  # Flask web UI assets
```

---

## Configuration

Config is merged in priority order (last wins): `default.json` → server override JSON → environment/CLI.

### Key parameters in `default.json`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | — | Path to YOLO weights (`.pt` or `.engine`) |
| `img_size` | 1280 | Inference resolution |
| `conf_thresh` | 0.6 | Minimum person confidence |
| `conf_thresh1_background` | 0.7 | Confidence threshold for medium-motion check |
| `conf_thresh2_background` | 0.95 | Confidence threshold for low-motion check |
| `background_thresh1/2/3` | — | MOG2 motion area thresholds |
| `hysteresis` | — | Consecutive frames required before reporting |
| `web_streaming_port` | 8081 | Flask web UI port |

### Camera entry (`cctv_<server>.json`)

Each camera must have:
- `device_id` — Spacenorm device identifier
- `uri` — RTSP stream URL
- `monitor_id` — NVR monitor identifier
- `ROI` (optional) — `{ "img_w", "img_h", "vertices": [[[x,y], ...]] }`
- `min_obj_size_ratio` (optional) — minimum bounding box area as % of frame
- `mqtt_broker_addr` / `mqtt_broker_port` (optional) — MQTT broker

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `SPACENORM_SERVER_ID` | Server identifier (e.g., `cym`, `kumho`) |
| `SPACENORM_DEFAULT_CFG_FILE` | Path to `default.json` |
| `SPACENORM_SERVER_CFG_FILE` | Path to server override JSON |
| `SPACENORM_SENSOR_CFG_FILE` | Path to `cctv_<server>.json` |
| `SPACENORM_DEVICEKEY_FILE` | Path to `.keys` file |
| `SPACENORM_LOG_DIR` | Log output directory |
| `SPACENORM_RECORD_DETECTION_RESULT_DIR` | Directory for saved detection frames |
| `SPACENORM_MODEL_PT` | Source `.pt` file for TRT engine export |
| `SPACENORM_ENGINE_CACHE_DIR` | Host-mounted engine cache directory |
| `SPACENORM_IMGSZ` | Image size for TRT engine build |
| `SPACENORM_TRT_HALF` | FP16 mode for TRT engine (`true`/`false`) |
| `SPACENORM_DEVICE` | GPU device index |
| `SPACENORM_TRT_WORKSPACE` | TRT workspace size in GiB |

---

## Docker Deployment

```bash
# Build image
cd docker_build
./docker_build.sh

# Push to ECR
aws ecr get-login-password --region ap-northeast-2 \
  | docker login --username AWS --password-stdin \
    159552820182.dkr.ecr.ap-northeast-2.amazonaws.com
docker push 159552820182.dkr.ecr.ap-northeast-2.amazonaws.com/spacenorm_obj_detection:cu128

# Deploy to a swarm node
cd docker_swarm
./deploy_obj_detection.sh cym

# Monitor
docker service ls
docker service ps spacenorm_obj_detection_cym
docker service logs -f spacenorm_obj_detection_cym
```

- Registry: AWS ECR (`159552820182.dkr.ecr.ap-northeast-2.amazonaws.com`)
- Image tag: `spacenorm_obj_detection:cu128`
- Stack template: `docker_swarm/stack.yml.template`
- Node constraints: `server=<name>` and `spacenorm_obj_detection=true`
- Ports: 8081 (web UI), 9000 (Prometheus)

### One-time worker node setup

```bash
# Shared log directory
sudo mkdir -p /var/log/spacenorm_obj_detection/detection_results

# TensorRT engine cache (shared with offline_ai_detector if co-located)
sudo mkdir -p /var/lib/spacenorm_obj_detection
sudo chmod 777 /var/lib/spacenorm_obj_detection
```

---

## Licensing

This project is licensed under **GNU Affero General Public License v3.0 (AGPL-3.0)**,
inherited from the Ultralytics YOLO11 dependency.

See [`LICENSE`](LICENSE) and [`docs/licensing.md`](docs/licensing.md) for:
- Full compliance requirements and procedure
- What must and need not be open-sourced
- Commercial license alternative (Ultralytics Enterprise)
- Commercially-friendly model alternatives (RT-DETR)
