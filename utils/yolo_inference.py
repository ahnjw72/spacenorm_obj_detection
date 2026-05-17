#################################################################
# Methods and utilities for YOLOv11 (Ultralytics) inference
#################################################################

import cv2
import datetime
import os
import numpy as np
from ultralytics import YOLO

import logging
logger = logging.getLogger(__name__)


def make_yolo_annotation_file(annotation_filepath, boxes, clss, types, img_width, img_height):
    # boxes is a list of box coordinates: bb[0], bb[1], bb[2], bb[3] : x_min, y_min, x_max, y_max
    # YOLO format: (label, x_center, y_center, width, height)  (<-- normalized to the image size)
    with open(annotation_filepath, 'w') as annotation_file:
        for bb, cl, type in zip(boxes, clss, types):
            x_center = (bb[0] + bb[2]) / 2.0
            y_center = (bb[1] + bb[3]) / 2.0
            dw = 1. / img_width
            dh = 1. / img_height
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]

            x_center = x_center * dw
            y_center = y_center * dh
            width = width * dw
            height = height * dh

            assert (x_center <= 1.0) and (x_center >= 0.0)
            assert (y_center <= 1.0) and (y_center >= 0.0)

            if type == 1 or type == 2 or type == 3:
                label = 3
            elif type == -1:
                label = 1
            elif type == -2:
                label = 2
            else:
                assert type == 0
                label = 0

            annotation_file.write(f"{label} {x_center} {y_center} {width} {height}\n")


def create_yolo_model(model_path):
    model = YOLO(model_path)
    logger.info(f"Ultralytics YOLO model loaded from {model_path}")
    return model


def yolo_inference_image(img0, model, imgsz, conf_thresh):
    """Run Ultralytics YOLO inference on a single BGR image.

    Returns boxes (list of [x1,y1,x2,y2] ints), confs, clss — same contract as
    the old yolov7_inference_image().
    """
    results = model(img0, imgsz=imgsz, conf=conf_thresh, verbose=False)

    boxes = []
    confs = []
    clss = []

    if results and len(results[0].boxes):
        result = results[0]
        for i in range(len(result.boxes)):
            xyxy = result.boxes.xyxy[i].cpu().tolist()
            boxes.append([int(x) for x in xyxy])
            confs.append(result.boxes.conf[i].cpu())
            clss.append(result.boxes.cls[i].cpu())

    return boxes, confs, clss


def record_detection_result(key, boxes, confs, clss, types, motionesses, detect_csv_writer, img, vis, args, frame_idx):

    org_images_dir = os.path.join(args.record_detection_result_dir, "org")
    detected_images_dir = os.path.join(args.record_detection_result_dir, "detected")

    assert vis is not None
    assert img is not None
    assert os.path.exists(org_images_dir)
    assert os.path.exists(detected_images_dir)

    now = datetime.datetime.now()

    time_str = "{}{:02d}{:02d}{:02d}{:02d}{:02d}".format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)

    org_img_filename = f"org_{key}_{frame_idx}.jpg"
    org_img_filepath = f"{org_images_dir}/{org_img_filename}"
    cv2.imwrite(org_img_filepath, img)

    result_img_filename = f"{key}_{frame_idx}.jpg"
    result_img_filepath = f"{detected_images_dir}/{result_img_filename}"
    result_img = vis.draw_bboxes(img, boxes, confs, clss, types, motionesses)
    cv2.imwrite(result_img_filepath, result_img)

    for box, conf, typ, motioness in zip(boxes, confs, types, motionesses):
        detect_csv_writer.writerow([key, time_str, typ, motioness, conf, result_img_filename])
