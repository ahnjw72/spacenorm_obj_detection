"""offline_spacenorm_obj_detection.py

Offline batch video processing using Ultralytics YOLOv11.

Watches a todos/ directory for new task folders. Each folder must contain:
  - A video file (.mp4 / .avi / .mov / .mkv)
  - A start  flag file  (triggers processing)
  - An optional settings.json  with ROI and min_obj_size_ratio

Results are written to done/{id}/:
  - Snapshot JPEG files for frames with detections
  - result.json  with per-frame bounding-box data
  - A done flag file when processing is complete

Usage:
    python -m spacenorm_obj_detection.offline_spacenorm_obj_detection \\
        --common_config /path/to/common_config.json
"""

import os
import sys
import time
import json
import cv2
from pathlib import Path
import argparse

# Disable Ultralytics telemetry before any YOLO import
from ultralytics.utils import SETTINGS
SETTINGS.update({'sync': False})

from .utils.yolo_inference import create_yolo_model, yolo_inference_image
from .utils.config_manager import ConfigManager
from .utils.post_processing import (
    remove_outside_ROI,
    filter_only_person,
    filter_small_objects,
    check_bb_on_background,
)

import logging

logger = logging.getLogger('offline_spacenorm_obj_detection')
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Result recording
# ---------------------------------------------------------------------------

def record_result(img, frame_count, frame_idx, frame_sec, boxes, confs, clss, out_dir):
    """Save a snapshot JPEG and return its result dict entry.

    Return format:
      { name: "00042.jpg", elapsed: 12.34, person: 2,
        bb: [[x1/W, y1/H, x2/W, y2/H], ...] }
    """
    width = len(str(frame_count))
    snapshot_filename = f"{frame_idx:0{width}d}.jpg"
    snapshot_filepath = os.path.join(out_dir, snapshot_filename)

    (H, W) = img.shape[:2]

    bb = []
    for box in boxes:
        x1, y1, x2, y2 = box
        bb.append([x1 / W, y1 / H, x2 / W, y2 / H])

    cv2.imwrite(snapshot_filepath, img)

    return {
        'name':    snapshot_filename,
        'elapsed': frame_sec,
        'person':  len(boxes),
        'bb':      bb,
    }


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(video_file, model, imgsz, roi, min_obj_size_ratio, args, out_dir):
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 0
    key = args.spacenorm_device_key

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg.setVarThreshold(50)
    fgbg.setDetectShadows(False)

    results = []

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_sec = frame_idx / fps
        logger.info(f"        - Processing frame {frame_idx}/{frame_count} (Time: {frame_sec:.2f}s)")

        # YOLOv11 inference
        boxes, confs, clss = yolo_inference_image(img, model, imgsz, args.conf_thresh)
        logger.info(f"            - Detected {len(boxes)} objects before post-processing")

        if len(boxes) > 0:
            # Post-process 1: keep only person detections
            boxes, confs, clss = filter_only_person(None, boxes, confs, clss, img, key, save_result=False)

            # Post-process 2: remove boxes smaller than min_obj_size_ratio
            if min_obj_size_ratio:
                (H, W) = img.shape[:2]
                boxes, confs, clss = filter_small_objects(key, boxes, confs, clss, min_obj_size_ratio, (H, W))

            # Post-process 3: remove detections outside the ROI polygon
            if roi:
                boxes, confs, clss = remove_outside_ROI(boxes, confs, clss, imgsz, roi, img, key, save_result=False)

            # Post-process 4: discard stationary/background detections via MOG2
            boxes, confs, clss, types, motionesses = check_bb_on_background(
                img, boxes, confs, clss, True, args, kernel, fgbg
            )

            if len(boxes) > 0:
                result = record_result(img, frame_count, frame_idx, frame_sec, boxes, confs, clss, out_dir)
                results.append(result)

    cap.release()
    return results


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_settings(settings_path):
    """Read per-task settings.json and return (roi, min_obj_size_ratio)."""
    with open(settings_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    roi = cfg.get('ROI', None)
    min_obj_size_ratio = cfg.get('min_obj_size_ratio', None)
    return roi, min_obj_size_ratio


# ---------------------------------------------------------------------------
# Per-task-folder processing
# ---------------------------------------------------------------------------

def process_id_folder(id_folder, model, imgsz, args):
    id_name = os.path.basename(id_folder)
    logger.info(f"[+] New task detected: {id_name}")
    logger.info(f"    - Processing folder: {id_folder}")

    # 1) Find the video file
    video_file = None
    for f in os.listdir(id_folder):
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_file = os.path.join(id_folder, f)
            logger.info(f"        - Video file found: {video_file}")
            break
    if not video_file:
        logger.info(f"    [ERROR] No video file found in {id_folder}")
        return

    # 2) Touch done flag in the task folder to record start time
    done_flag = os.path.join(id_folder, "done")
    Path(done_flag).touch()
    logger.info(f"    - Start-time flag created: {done_flag}")

    # 3) Load optional per-task settings (ROI, min_obj_size_ratio)
    settings_path = os.path.join(id_folder, "settings.json")
    if not os.path.exists(settings_path):
        roi = None
        min_obj_size_ratio = None
        logger.info(f"        settings.json missing in {id_folder} — skipping ROI and size filter")
    else:
        roi, min_obj_size_ratio = load_settings(settings_path)
        logger.info(f"    - ROI loaded: {roi}")
        logger.info(f"    - min_obj_size_ratio: {min_obj_size_ratio}")

    # 4) Create output directory
    done_dir = os.path.join(args.root_dir, "done")
    out_dir = os.path.join(done_dir, id_name)
    os.makedirs(out_dir, exist_ok=True)

    # 5) Run inference on every frame
    logger.info(f"    - Running YOLOv11 inference on video...")
    results = process_video(video_file, model, imgsz, roi, min_obj_size_ratio, args, out_dir)

    logger.info("====================================================")
    logger.info(results)
    logger.info("====================================================")

    # 6) Write result.json
    result_json_path = os.path.join(out_dir, "result.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump({"response_code": "성공", "files": results}, f, ensure_ascii=False, indent=2)

    # 7) Touch done flag in the output folder to signal completion
    time.sleep(5)  # brief wait for filesystem stability
    Path(os.path.join(out_dir, "done")).touch()
    logger.info(f"    - Detection finished for {id_name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Offline batch video detection with YOLOv11")
    parser.add_argument("--common_config", required=True, help="Path to common runtime configuration JSON file")
    parser.add_argument("--hot_reload", action="store_true", help="Reload config on each scan (not yet implemented)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    logger.info("=== Offline spacenorm_obj_detection started ===")

    args = parse_args()

    # Load config and flatten into args namespace
    cfg = ConfigManager(args)
    for key, val in vars(cfg.config).items():
        if isinstance(val, dict) and "value" in val:
            setattr(args, key, val["value"])
        else:
            setattr(args, key, val)

    # Set up console logging now that we have the config
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    scan_interval = args.scan_interval
    logger.info(f"[INFO] Scan interval set to {scan_interval} seconds")

    todos_dir = os.path.join(args.root_dir, "todos")

    # Load the YOLO model once and reuse across all tasks
    model = create_yolo_model(args.model)
    imgsz = args.img_size

    # Main scan loop
    while True:
        if not os.path.exists(todos_dir):
            logger.info(f"[INFO] Todos directory '{todos_dir}' does not exist..")
        else:
            for name in os.listdir(todos_dir):
                try:
                    folder = os.path.join(todos_dir, name)
                    if os.path.isdir(folder):
                        start_flag = os.path.join(folder, "start")
                        if os.path.exists(start_flag):
                            detect_done_flag = os.path.join(folder, "done")
                            if not os.path.exists(detect_done_flag):
                                process_id_folder(folder, model, imgsz, args)
                            else:
                                logger.info(f"[INFO] Task {name} already processed.")
                except Exception as e:
                    logger.error(f"[ERROR] Exception while processing task '{name}': {e}")

        time.sleep(scan_interval)


if __name__ == "__main__":
    main()
