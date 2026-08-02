#!/usr/bin/env python3
"""monitor_detection_result.py

Standalone diagnostic tool: runs YOLO inference plus the production
post-processing pipeline (utils/post_processing.py) over an input video,
frame by frame (decimated to args.report_period by default, for
reproducibility with the live per-camera processing cadence), and writes one
annotated JPG per processed frame.

By default, all detection parameters (conf_thresh, background thresholds,
img_size, model, report_period, ...) are loaded exactly as the live
spacenorm_obj_detection service would for a given --site: default.json,
merged with spacenorm_cfg/behavior/overrides/<site>.json, plus
spacenorm_cfg/cctv/cctv_<site>.json for ROI/min_obj_size_ratio when --camera
is also given. --model, --conf_thresh and --report_period may be passed to
override those reproduced values for a specific investigation.

By default, only detections at or above conf_thresh are shown (i.e. exactly
what production's inference call would ever see). Pass --conf_thresh_low
with a lower value (e.g. 0.001) to additionally surface sub-conf_thresh
detections for investigation.

Every detection that reaches the pipeline is retained in the output image,
color coded by the reason it was filtered (or kept) at each stage, so
filtering decisions can be inspected visually:

    raw detections (>= conf_thresh_low, default conf_thresh)
        -> below conf_thresh                      [gray] (only if conf_thresh_low < conf_thresh)
        -> filter_only_person()    -> not person   [magenta]
        -> filter_small_objects()  -> too small    [yellow]
        -> remove_outside_ROI()    -> outside ROI  [red]
        -> check_bb_on_background()
            -> background type 1/2                [orange]
            -> background type 3                   [gray]
            -> kept, low motion (type -1/-2)       [cyan]
            -> kept, reported (type 0)             [green]

The ROI polygon(s) (if any) are drawn in white.
"""

import argparse
import json
import logging
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np

# Make the project's top-level `utils` package importable when this script is
# run directly from scripts/cctv/monitor_detection_result/.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ultralytics.utils import SETTINGS
SETTINGS.update({'sync': False})

from utils.yolo_inference import create_yolo_model, yolo_inference_image
from utils.post_processing import (
    filter_only_person,
    filter_small_objects,
    remove_outside_ROI,
    check_bb_on_background,
    make_counterclockwise,
)

logger = logging.getLogger("monitor_detection_result")
logger.setLevel(logging.INFO)


# Custom 7-class model: person, bird, cat, dog, horse, sheep, cow (see CLAUDE.md)
CLASS_NAMES = {0: 'person', 1: 'bird', 2: 'cat', 3: 'dog', 4: 'horse', 5: 'sheep', 6: 'cow'}

# Colors are BGR (OpenCV convention).
COLOR_BELOW_CONF = (105, 105, 105)      # dark gray
COLOR_NOT_PERSON = (255, 0, 255)        # magenta
COLOR_TOO_SMALL = (0, 255, 255)         # yellow
COLOR_OUTSIDE_ROI = (0, 0, 255)         # red
COLOR_BG_TYPE12 = (0, 165, 255)         # orange
COLOR_BG_TYPE3 = (128, 128, 128)        # gray
COLOR_KEPT_LOW_MOTION = (255, 255, 0)   # cyan
COLOR_KEPT = (0, 255, 0)                # green
COLOR_ROI_POLYGON = (255, 255, 255)     # white

LEGEND = [
    ("below conf_thresh", COLOR_BELOW_CONF),
    ("not person", COLOR_NOT_PERSON),
    ("too small", COLOR_TOO_SMALL),
    ("outside ROI", COLOR_OUTSIDE_ROI),
    ("background (t1/t2)", COLOR_BG_TYPE12),
    ("background (t3)", COLOR_BG_TYPE3),
    ("kept, low motion", COLOR_KEPT_LOW_MOTION),
    ("kept, reported", COLOR_KEPT),
]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_behavior_config(default_cfg_path, server_cfg_path=None):
    """Flatten default.json (+ optional server override) into an argparse.Namespace.

    Mirrors utils/config_loader.py's merge semantics: server_cfg[category][param]['value']
    overrides base_cfg[category][param]['value']; unknown keys raise.
    """
    with open(default_cfg_path, "r", encoding="utf-8") as f:
        cfg_tree = json.load(f)

    if server_cfg_path:
        with open(server_cfg_path, "r", encoding="utf-8") as f:
            server_cfg = json.load(f)
        for category, params in server_cfg.items():
            if category not in cfg_tree:
                raise KeyError(f"Unknown category in server config: {category}")
            for name, meta in params.items():
                if name not in cfg_tree[category]:
                    raise KeyError(f"Unknown parameter in server config: {category}.{name}")
                cfg_tree[category][name]["value"] = meta["value"]

    flat = {}
    for category, params in cfg_tree.items():
        for name, meta in params.items():
            flat[name] = meta["value"]

    return argparse.Namespace(**flat)


def resolve_config_paths(project_root, site):
    """Resolve the (default_cfg, server_cfg, roi_cfg) paths the live service
    would use for `site`, mirroring the layout described in CLAUDE.md:
        spacenorm_cfg/behavior/default.json                  (always)
        spacenorm_cfg/behavior/overrides/<site>.json         (if it exists)
        spacenorm_cfg/cctv/cctv_<site>.json                  (if it exists)
    """
    default_cfg_path = project_root / "spacenorm_cfg" / "behavior" / "default.json"

    server_cfg_path = None
    roi_cfg_path = None
    if site:
        candidate = project_root / "spacenorm_cfg" / "behavior" / "overrides" / f"{site}.json"
        if candidate.is_file():
            server_cfg_path = candidate
        else:
            logger.info(f"No behavior override found for site '{site}' at {candidate}; using default.json only")

        candidate = project_root / "spacenorm_cfg" / "cctv" / f"cctv_{site}.json"
        if candidate.is_file():
            roi_cfg_path = candidate
        else:
            logger.info(f"No cctv config found for site '{site}' at {candidate}")

    return default_cfg_path, server_cfg_path, roi_cfg_path


def load_camera_config(roi_cfg_path, camera_name):
    """Look up ROI and min_obj_size_ratio for `camera_name` in a cctv_<server>.json file."""
    with open(roi_cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for company in data.get("companies", []):
        cctv = company.get("CCTV", {})
        if camera_name in cctv:
            entry = cctv[camera_name]
            return entry.get("ROI"), entry.get("min_obj_size_ratio")

    raise KeyError(f"Camera '{camera_name}' not found in {roi_cfg_path}")


# ---------------------------------------------------------------------------
# ROI drawing
# ---------------------------------------------------------------------------

def build_normalized_polygons(roi):
    """Normalize ROI polygon vertices to [0,1] x [0,1], same logic as remove_outside_ROI()."""
    img_w_roi = roi["img_w"]
    img_h_roi = roi["img_h"]
    vertices_sorted = roi.get("vertices_sorted", 0)

    polygons = []
    for polygon in roi["vertices"]:
        if vertices_sorted == 0:
            normalized = make_counterclockwise([[x / img_w_roi, y / img_h_roi] for x, y in polygon])
        else:
            normalized = [[x / img_w_roi, y / img_h_roi] for x, y in polygon]
        polygons.append(normalized)
    return polygons


def draw_roi_polygons(img, normalized_polygons):
    h, w = img.shape[:2]
    for normalized in normalized_polygons:
        pts = np.array([[int(x * w), int(y * h)] for x, y in normalized], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=COLOR_ROI_POLYGON, thickness=2)


# ---------------------------------------------------------------------------
# Box / legend drawing
# ---------------------------------------------------------------------------

def _text_color_for_bg(bgr):
    b, g, r = bgr
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def draw_label(img, text, topleft, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    margin = 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(topleft[0], 0)
    y = max(topleft[1], th + margin * 2)
    cv2.rectangle(img, (x, y - th - margin * 2), (x + tw + margin * 2, y), color, -1)
    cv2.putText(img, text, (x + margin, y - margin), font, scale, _text_color_for_bg(color), thickness, cv2.LINE_AA)


def draw_legend(img):
    row_h = 20
    swatch_w = 18
    pad = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    max_text_w = max(cv2.getTextSize(text, font, scale, 1)[0][0] for text, _ in LEGEND)
    panel_w = swatch_w + pad * 3 + max_text_w
    panel_h = row_h * len(LEGEND) + pad * 2

    # Top-right corner: CCTV OSD overlays (timestamp, camera name) most commonly
    # sit top-left or bottom, so this position is least likely to collide.
    img_w = img.shape[1]
    x0 = img_w - panel_w - 10
    y0 = 10

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    for idx, (text, color) in enumerate(LEGEND):
        y = y0 + pad + idx * row_h
        cv2.rectangle(img, (x0 + pad, y + 3), (x0 + pad + swatch_w, y + row_h - 3), color, -1)
        cv2.rectangle(img, (x0 + pad, y + 3), (x0 + pad + swatch_w, y + row_h - 3), (255, 255, 255), 1)
        cv2.putText(img, text, (x0 + pad * 2 + swatch_w, y + row_h - 6), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Per-object identity tracking across filter stages
# ---------------------------------------------------------------------------

def split_by_identity(originals, survivors):
    """originals: list passed into a filter function; survivors: the list it
    returned. post_processing.py filter functions only ever append references
    from their input lists, so `survivors` is an order-preserving subsequence
    of `originals` by object identity. Returns (kept_positions, dropped_positions),
    positions being indices into `originals`.
    """
    kept_positions = []
    dropped_positions = []
    si = 0
    for i, item in enumerate(originals):
        if si < len(survivors) and item is survivors[si]:
            kept_positions.append(i)
            si += 1
        else:
            dropped_positions.append(i)
    return kept_positions, dropped_positions


# ---------------------------------------------------------------------------
# Frame processing
# ---------------------------------------------------------------------------

def process_frame(img, model, imgsz, args, roi, min_obj_size_ratio, kernel, fgbg, conf_thresh_low):
    """Run inference + full post-processing pipeline, tagging every raw
    detection with the reason it was filtered/kept and a display color.

    Returns (raw_boxes, reasons, colors, labels) - four parallel lists.
    """
    raw_boxes, raw_confs, raw_clss = yolo_inference_image(img, model, imgsz, conf_thresh_low)
    n = len(raw_boxes)

    reasons = [None] * n
    colors = [None] * n
    labels = [None] * n

    def active_lists(active_idx):
        return ([raw_boxes[i] for i in active_idx],
                [raw_confs[i] for i in active_idx],
                [raw_clss[i] for i in active_idx])

    # Stage 0: production conf_thresh gate (yolo_inference_image() is called
    # with conf=conf_thresh_low above so this stage is visible at all).
    active_idx = []
    for i in range(n):
        conf_val = float(raw_confs[i])
        if conf_val < args.conf_thresh:
            reasons[i] = "below_conf_thresh"
            colors[i] = COLOR_BELOW_CONF
            labels[i] = f"cf={conf_val:.2f}"
        else:
            active_idx.append(i)

    # Stage 1: keep only 'person' class
    boxes, confs, clss = active_lists(active_idx)
    new_boxes, _, _ = filter_only_person(None, boxes, confs, clss, img, args.spacenorm_device_key, save_result=False)
    kept_pos, dropped_pos = split_by_identity(boxes, new_boxes)
    for p in dropped_pos:
        i = active_idx[p]
        cls_id = int(raw_clss[i])
        reasons[i] = "not_person"
        colors[i] = COLOR_NOT_PERSON
        labels[i] = f"{CLASS_NAMES.get(cls_id, 'cls%d' % cls_id)} cf={float(raw_confs[i]):.2f}"
    active_idx = [active_idx[p] for p in kept_pos]

    # Stage 2: minimum object size (area ratio) filter
    if min_obj_size_ratio:
        boxes, confs, clss = active_lists(active_idx)
        (H, W) = img.shape[:2]
        new_boxes, _, _ = filter_small_objects(args.spacenorm_device_key, boxes, confs, clss, min_obj_size_ratio, (H, W))
        kept_pos, dropped_pos = split_by_identity(boxes, new_boxes)
        for p in dropped_pos:
            i = active_idx[p]
            reasons[i] = "too_small"
            colors[i] = COLOR_TOO_SMALL
            labels[i] = f"cf={float(raw_confs[i]):.2f}"
        active_idx = [active_idx[p] for p in kept_pos]

    # Stage 3: ROI filter
    if roi is not None:
        boxes, confs, clss = active_lists(active_idx)
        new_boxes, _, _ = remove_outside_ROI(boxes, confs, clss, imgsz, roi, img, args.spacenorm_device_key, save_result=False)
        kept_pos, dropped_pos = split_by_identity(boxes, new_boxes)
        for p in dropped_pos:
            i = active_idx[p]
            reasons[i] = "outside_roi"
            colors[i] = COLOR_OUTSIDE_ROI
            labels[i] = f"cf={float(raw_confs[i]):.2f}"
        active_idx = [active_idx[p] for p in kept_pos]

    # Stage 4: background / motion check. Always called with remove_member=False
    # so every surviving box gets a `type`, regardless of args.remove_background_bb;
    # we apply that config flag ourselves below purely to choose the label/reason.
    if args.background:
        boxes, confs, clss = active_lists(active_idx)
        _, _, _, types, motionesses = check_bb_on_background(img, boxes, confs, clss, False, args, kernel, fgbg)
        for pos, i in enumerate(active_idx):
            t = types[pos]
            m = motionesses[pos]
            if t == 0:
                reasons[i] = "kept"
                colors[i] = COLOR_KEPT
            elif t in (-1, -2):
                reasons[i] = "kept_low_motion"
                colors[i] = COLOR_KEPT_LOW_MOTION
            else:
                assert t in (1, 2, 3)
                colors[i] = COLOR_BG_TYPE3 if t == 3 else COLOR_BG_TYPE12
                reasons[i] = f"background_t{t}" if args.remove_background_bb else f"kept_background_t{t}"
            labels[i] = f"cf={float(raw_confs[i]):.2f} m={m:.2f}"
    else:
        for i in active_idx:
            reasons[i] = "kept"
            colors[i] = COLOR_KEPT
            labels[i] = f"cf={float(raw_confs[i]):.2f}"

    return raw_boxes, reasons, colors, labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO + post-processing over a video and dump annotated JPGs "
                    "with every raw detection color-coded by its filtering reason.")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--site", default=None,
                        help="Site identifier (e.g. cym, jaeil_cr, kumho) used to reproduce that site's live "
                            "configuration: spacenorm_cfg/behavior/overrides/<site>.json (if present) is merged "
                            "over default.json, and spacenorm_cfg/cctv/cctv_<site>.json supplies ROI / "
                            "min_obj_size_ratio when --camera is also given.")
    parser.add_argument("--camera", default=None,
                        help="Camera name key to look up in spacenorm_cfg/cctv/cctv_<site>.json (requires --site)")
    parser.add_argument("--report_period", type=float, default=None,
                        help="Seconds between sampled frames (decimated using the video's own FPS). "
                            "Default: the site's configured report_period (matches the live per-camera "
                            "processing cadence). Use 0 to process every frame.")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for annotated JPGs. Default: <video_stem>_monitor/ next to the video.")
    parser.add_argument("--conf_thresh_low", type=float, default=None,
                        help="Confidence used for the raw YOLO inference pass. Default: same as "
                            "conf_thresh (config value, or --conf_thresh if given), so only detections "
                            "that would actually be reported in production are shown. Pass a lower value "
                            "(e.g. 0.001) to additionally surface sub-conf_thresh detections for "
                            "investigation.")
    parser.add_argument("--conf_thresh", type=float, default=None, help="Override conf_thresh from config")
    parser.add_argument("--model", default=None, help="Override model path from config")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)

    cli = parse_args()

    if cli.camera and not cli.site:
        raise SystemExit("--site is required when --camera is given")

    default_cfg_path, server_cfg_path, roi_cfg_path = resolve_config_paths(_PROJECT_ROOT, cli.site)
    logger.info(f"Behavior config: default={default_cfg_path}, override={server_cfg_path or '(none)'}")

    args = load_behavior_config(default_cfg_path, server_cfg_path)
    if cli.conf_thresh is not None:
        args.conf_thresh = cli.conf_thresh

    conf_thresh_low = cli.conf_thresh_low if cli.conf_thresh_low is not None else args.conf_thresh
    logger.info(f"conf_thresh = {args.conf_thresh}, conf_thresh_low = {conf_thresh_low}")

    # Model weights are expected to sit next to this script; --model can point
    # elsewhere explicitly. Never fall back to ultralytics' auto-download --
    # fail fast with a clear message instead.
    if cli.model is not None:
        model_path = Path(cli.model)
    else:
        model_path = _SCRIPT_DIR / Path(args.model).name

    if not model_path.is_file():
        raise SystemExit(
            f"Model file not found: {model_path}\n"
            f"Place the .pt weights file at that path, or pass --model <path> to point "
            f"at one elsewhere. This script never downloads model weights automatically."
        )
    args.model = str(model_path)

    video_path = Path(cli.video)
    video_stem = video_path.stem
    args.spacenorm_device_key = cli.camera or cli.site or video_stem

    roi = None
    normalized_polygons = []
    min_obj_size_ratio = None
    if cli.camera:
        if roi_cfg_path is None:
            raise SystemExit(f"--camera given but no cctv config found for site '{cli.site}' "
                            f"(expected spacenorm_cfg/cctv/cctv_{cli.site}.json)")
        roi, min_obj_size_ratio = load_camera_config(roi_cfg_path, cli.camera)
        if roi is not None:
            normalized_polygons = build_normalized_polygons(roi)
        else:
            logger.info(f"No ROI defined for camera '{cli.camera}' in {roi_cfg_path}")

    output_dir = Path(cli.output_dir) if cli.output_dir else video_path.parent / f"{video_stem}_monitor"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {args.model}")
    model = create_yolo_model(args.model)
    imgsz = args.img_size

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    report_period = cli.report_period if cli.report_period is not None else args.report_period

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if report_period:
        if fps <= 0:
            raise SystemExit("Could not determine video FPS; cannot use --report_period")
        frame_step = max(1, round(report_period * fps))
    else:
        frame_step = 1
    logger.info(f"Video FPS = {fps:.2f}, report_period={report_period}s -> frame_step = {frame_step}")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg.setVarThreshold(50)
    fgbg.setDetectShadows(False)

    frame_idx = 0
    saved_files = []
    while True:
        ret, img = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            raw_boxes, reasons, colors, labels = process_frame(
                img, model, imgsz, args, roi, min_obj_size_ratio, kernel, fgbg, conf_thresh_low)

            out_img = img.copy()
            draw_roi_polygons(out_img, normalized_polygons)
            for box, reason, color, label in zip(raw_boxes, reasons, colors, labels):
                x1, y1, x2, y2 = box
                cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
                text = f"{reason} {label}" if label else reason
                draw_label(out_img, text, (x1, y1), color)
            draw_legend(out_img)

            t_sec = frame_idx / fps if fps > 0 else float(frame_idx)
            out_name = f"{video_stem}_frame{frame_idx:06d}_t{t_sec:.2f}s.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), out_img)
            saved_files.append(out_path)
            logger.info(f"[{args.spacenorm_device_key}] frame {frame_idx} (t={t_sec:.2f}s): "
                        f"{len(raw_boxes)} raw detection(s) -> {out_name}")

        frame_idx += 1

    cap.release()
    logger.info(f"Done. {len(saved_files)} frame(s) written to {output_dir}")

    archive_path = output_dir / f"{video_stem}_monitor.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for jpg_path in saved_files:
            tar.add(jpg_path, arcname=jpg_path.name)
    logger.info(f"Archived {len(saved_files)} JPG(s) to {archive_path}")


if __name__ == "__main__":
    main()
