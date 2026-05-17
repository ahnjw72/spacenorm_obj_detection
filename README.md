# spacenorm_yolo

Real-time human occupancy detection service for CCTV RTSP streams, powered by [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics). Detects people in each camera frame, applies ROI masking and background subtraction, and reports occupancy counts to the Spacenorm API via REST and MQTT.

This project is an inference-only service refactored from `spacenorm_yolov7`. Training and file-based (video/image/NVR) inference are out of scope.

---

## Features

- Multi-threaded RTSP stream processing (one thread per camera)
- Ultralytics YOLO inference with configurable model and resolution
- Polygon ROI masking, small-object filtering, MOG2 background subtraction
- Hysteresis to suppress false-positive flicker
- Occupancy reporting via Spacenorm REST API and MQTT
- Web UI live stream (Flask, port 8081)
- Prometheus metrics (port 9000)
- Docker Swarm deployment on edge GPU nodes

---

## Quick Start

### Prerequisites

```bash
pip install ultralytics flask paho-mqtt prometheus-client pymediainfo
```

### Run Locally

```bash
SPACENORM_SERVER_ID=cym \
SPACENORM_DEFAULT_CFG_FILE=./spacenorm_cfg/behavior/default.json \
SPACENORM_SERVER_CFG_FILE=./spacenorm_cfg/behavior/overrides/cym.json \
SPACENORM_SENSOR_CFG_FILE=./spacenorm_cfg/cctv/cctv_cym.json \
SPACENORM_DEVICEKEY_FILE=./spacenorm_cfg/cctv/cctv_cym.keys \
python -u -m spacenorm_yolo.spacenorm_yolo
```

---

## Detection Pipeline

```
RTSP Stream → GStreamer Frame Queue
    → Ultralytics YOLO inference
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
spacenorm_yolo/
├── spacenorm_yolo.py          # Main service entry point
├── __init__.py
├── utils/
│   ├── yolo_inference.py      # Ultralytics YOLO model wrapper
│   ├── post_processing.py     # ROI, background, size filtering
│   ├── cctv_camera.py         # RTSP stream handling (GStreamer)
│   ├── config_loader.py       # Hierarchical config loading
│   ├── spacenorm_api.py       # Spacenorm REST API client
│   ├── kakao_messaging.py     # Kakao Talk alerts
│   ├── visualization.py       # Bounding box drawing
│   ├── display.py             # FPS overlay
│   ├── yolo_classes.py        # Class index definitions
│   └── gateway_api.py         # Gateway API client
├── spacenorm_cfg/
│   ├── behavior/
│   │   ├── default.json       # Global defaults
│   │   └── overrides/         # Per-server overrides
│   └── cctv/
│       ├── cctv_<server>.json # Camera definitions (RTSP URIs, ROI)
│       └── cctv_<server>.keys # Device keys
├── templates/                 # Flask HTML templates
├── static/                    # Static web assets
└── docker_swarm/              # Docker Swarm stack files
```

---

## Configuration

Config is merged in priority order (last wins): `default.json` → server override JSON → environment/CLI.

### Key parameters in `default.json`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | — | Path to Ultralytics YOLO weights (`.pt`) |
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

---

## Docker Deployment

```bash
# Build image
cd docker_build
./docker_build.sh

# Deploy to a swarm node
cd docker_swarm
./deploy_yolov7.sh cym

# Monitor
docker service ls
docker service ps spacenorm_detector
docker service logs -f spacenorm_detector
```

- Registry: AWS ECR (`159552820182.dkr.ecr.ap-northeast-2.amazonaws.com`)
- Image: `spacenorm_yolo:latest`
- Stack template: `docker_swarm/stack.yml.template`
- Node constraints: `server=<name>` and `spacenorm_yolo=true`
- Ports: 8081 (web UI), 9000 (Prometheus)
