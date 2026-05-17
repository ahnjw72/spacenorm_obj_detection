# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**spacenorm_yolo** is a real-time human occupancy detection system based on Ultralytics YOLOv11, processing CCTV RTSP streams to detect people and report occupancy events to the Spacenorm API. It supports multiple client sites and deploys as a Docker Swarm service on edge GPU nodes.

This project is a focused inference-only service. Training, model export, and file-based (video/image/NVR) inference are out of scope — see the sibling project `spacenorm_yolov7` for those workflows.

## Commands

### Running the Service Locally

```bash
SPACENORM_SERVER_ID=cym \
SPACENORM_DEFAULT_CFG_FILE=./spacenorm_cfg/behavior/default.json \
SPACENORM_SERVER_CFG_FILE=./spacenorm_cfg/behavior/overrides/cym.json \
SPACENORM_SENSOR_CFG_FILE=./spacenorm_cfg/cctv/cctv_cym.json \
SPACENORM_DEVICEKEY_FILE=./spacenorm_cfg/cctv/cctv_cym.keys \
python -u -m spacenorm_yolo.spacenorm_yolo
```

### Docker Deployment

```bash
# Build image
cd docker_build
./docker_build.sh

# Deploy via Docker Swarm
cd docker_swarm
./deploy_yolov7.sh cym
./deploy_yolov7.sh kumho

# Service management
docker service ls
docker service ps spacenorm_detector
docker service logs -f spacenorm_detector
```

## Architecture

### Detection Pipeline

```
RTSP Stream → GStreamer Frame Queue
    → Ultralytics YOLO (configurable resolution, 7 classes) inference
    → Post-processing:
        filter_only_person()          # Drop non-person detections
        remove_outside_ROI()          # Polygon ROI masking per camera
        filter_small_objects()        # Drop small bounding boxes
        check_bb_on_background()      # MOG2 background subtraction
        Hysteresis (N consecutive frames required)
    → MQTT publish + Spacenorm API + Web UI (port 8081)
```

**Background detection types:** Type 0 (high conf, high motion) → type -1 (medium conf, medium motion) → type -2 (high conf, low motion) are kept as real detections. Types 1/2/3 are filtered as background.

### Key Source Files

| File | Purpose |
|------|---------|
| `spacenorm_yolo.py` | Main Flask service: multi-threaded RTSP processing, MQTT reporting, web streaming |
| `utils/yolo_inference.py` | Ultralytics YOLO model wrapper and annotation |
| `utils/post_processing.py` | ROI filtering, background detection, size filtering |
| `utils/cctv_camera.py` | RTSP camera stream handling (GStreamer) |
| `utils/config_loader.py` | Hierarchical config loading |
| `utils/spacenorm_api.py` | REST API client for the Spacenorm server |
| `utils/kakao_messaging.py` | Kakao Talk alert notifications |
| `utils/visualization.py` | Bounding box drawing on frames |
| `utils/display.py` | FPS overlay on frames |

### Configuration System

Config is loaded in priority order (highest last wins): `default.json` → server override JSON → CLI args.

```
spacenorm_cfg/
├── behavior/
│   ├── default.json                  # Global defaults (thresholds, ports, model path)
│   └── overrides/<server>.json       # Per-server behavioral overrides
└── cctv/
    ├── cctv_<server>.json            # Camera definitions: RTSP URIs, ROI polygons, device IDs
    └── cctv_<server>.keys            # Device keys (hex-encoded)
```

**Critical config parameters** in `default.json`:
- `conf_thresh` (0.6) — minimum person confidence
- `conf_thresh1_background` / `conf_thresh2_background` — thresholds for high/low motion scenarios
- `background_thresh1/2/3` — MOG2 motion thresholds (pixel area ratio)
- `hysteresis` — consecutive frames required before reporting
- `img_size` (1280) — inference resolution

Each camera entry in `cctv_<server>.json` specifies `device_id`, `uri` (RTSP), `monitor_id`, and an `ROI` with polygon `vertices` in image coordinates.

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `SPACENORM_SERVER_ID` | Server identifier (e.g., `cym`, `kumho`) |
| `SPACENORM_DEFAULT_CFG_FILE` | Path to `default.json` |
| `SPACENORM_SERVER_CFG_FILE` | Path to server override JSON |
| `SPACENORM_SENSOR_CFG_FILE` | Path to `cctv_<server>.json` |
| `SPACENORM_DEVICEKEY_FILE` | Path to `.keys` file |
| `SPACENORM_LOG_DIR` | Log output directory |
| `SPACENORM_RECORD_DETECTION_RESULT_DIR` | Saved detection frames directory |

### Docker Swarm Deployment

- Image registry: AWS ECR (`159552820182.dkr.ecr.ap-northeast-2.amazonaws.com`)
- Standard image: `spacenorm_yolo:latest`
- Stack template: `docker_swarm/stack.yml.template` — rendered per-server by `deploy_yolov7.sh`
- Service runs `global` mode, constrained to nodes labeled `server=<name>` and `spacenorm_yolo=true`
- Configs injected at `/app/spacenorm_yolo/spacenorm_cfg/...` inside the container
- Ports: 8081 (web streaming), 9000 (Prometheus metrics)
- Logs: json-file driver, max 100MB × 3 files

### Model

- Framework: Ultralytics YOLO (YOLOv11 or compatible)
- Input: configurable via `img_size` (default 1280×1280)
- Classes: 7 (person, bird, cat, dog, horse, sheep, cow) — only `person` is used for reporting
- Model path: configured via `model` field in `default.json`
