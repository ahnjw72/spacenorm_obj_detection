"""spacenorm_obj_detection.py

This script demonstrates how to do real-time object detection with
Ultralytics YOLOv11 model.

"""
import sys
import faulthandler
import signal

faulthandler.enable(file=sys.stderr)
faulthandler.register(signal.SIGUSR1, all_threads=True)
faulthandler.dump_traceback_later(60, repeat=True)

import os, subprocess, threading
import time
import datetime
import logging
import logging.handlers as handlers
import copy
import csv
import cv2
import traceback
import json
import paho.mqtt.client as mqtt
from pathlib import Path
import socket

from flask import Response, request, Flask, render_template

from .utils.yolo_classes import get_cls_dict
from .utils.cctv_camera import RTSP_Camera
from .utils.display import show_fps
from .utils.visualization import BBoxVisualization
from .utils.spacenorm_api import Spacenorm_API
from .utils.kakao_messaging import refreshToken, kakaoMsgSend

from .utils.post_processing import remove_outside_ROI, select_car_related_results, filter_only_person, filter_small_objects, check_bb_on_background
from ultralytics.utils import SETTINGS
SETTINGS.update({'sync': False})

from .utils.yolo_inference import make_yolo_annotation_file, create_yolo_model, yolo_inference_image, record_detection_result
from .utils.config_loader import SpacenormConfigLoader

from prometheus_client import Gauge

import urllib.parse

cctv_ID = {}
RTSP_Camera_dict = {}

outputFrame = None
lock = threading.Lock()

lock_kakao_code = threading.Lock()

selected_key = None
cctv_space_name = None

# initialize a flask object
app = Flask(__name__)
app.secret_key = 'abcde'

SOURCE_CODE_URL = "https://github.com/ahnjw72/spacenorm_obj_detection"

@app.after_request
def add_source_code_header(response):
    """AGPL-3.0 compliance: expose source code URL to all HTTP responses."""
    response.headers["X-Source-Code"] = SOURCE_CODE_URL
    return response

PKG_NAME = Path(__file__).parent.name

WINDOW_NAME = 'TrtYOLODemo' 
 
CONF_THRESH = 0.0
CONF_THRESH1_BACKGROUND = 0.0
CONF_THRESH2_BACKGROUND = 0.0

BACKGROUND_THRESH1 = 0.0 # if the normalized(by the size of bb) sum of foreground pixel values in a bounding box is greater than this threshold, the bb is assumed to be foreground object.
BACKGROUND_THRESH2 = 0.0
BACKGROUND_THRESH3 = 0.0 # smaller than this value means the bb is surely background.

"""
   'motioness' (refer to check_bb_on_background())
    ^
    |                              |
    |              P(type 0)       |
 m1 |-------------------------------
    |                  |           |
    |                  |           |
    |       B(type 1)  |P(type -1) |   
    |                  |           |
    |                  |           |
 m2 |-------------------------------
    |         B(type 2)   |P(type -2)   
    |                     |        |
 m3 |-------------------------------
    |              B(type 3)       |   
    |                              |
    |---------------------------------> threshold
    ^                  ^  ^       1.0
    |                  |  |   
    |                  |  |  
 conf_thresh           t1 t2

 where,
    m1: BACKGROUND_THRESH1 (1.0 --> 5.0(2023.06.20))
    m2: BACKGROUND_THRESH2 (0.2 --> 2.0(2023.06.20))
    m3: BACKGROUND_THRESH3 (0.11 --> 1.0(2023.06.20)) (<-- when motioness is below this value, detected obj is always reported as background to reduce false positive)
    t1: CONF_THRESH1_BACKGROUND (0.7)
    t2: CONF_THRESH2_BACKGROUND (0.95)

 BB colors in nvr_file_detection
    Person(type  0): RED
    Person(type -1): GREEN
    Person(type -2): GREEN
    Background(type 1): BLUE
    Background(type 2): BLUE
    Background(type 3): BLACK

"""
 
SPACENORM_REPORT_PERIOD_SEC = 3.0 # default target interval between report checks; overridden from cfg.report_period in main() before any thread starts, and never mutated afterwards

SPACENORM_HEARTBEAT_PERIOD_SEC = 300 # every 5 minutes

KAKAO_CODE_JSON_FILE = f"./{PKG_NAME}/utils/kakao_code.json"

spacenorm_key_list = []

# initialize logging system -- ahnjw,2020.12.03
MAX_LOG_BYTES = 500*1024*1024

# logger = logging.getLogger('spacenorm_person_detect')

# 참고 : https://chatgpt.com/share/6963203e-8938-8009-8444-b44a55c38ac7
APP_LOGGER_NAME = __package__ or __name__.split(".", 1)[0]
logger = logging.getLogger(APP_LOGGER_NAME)

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s') 



def process_spacenorm_devicekey_file(args):
    args_dict = {}
    with open(args.spacenorm_devicekey_file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            if line[0] == '#':
                continue
            new_args = copy.copy(args)
            new_args.spacenorm_device_key = line
            args_dict[line] = new_args

    return args_dict
 
       
def report_via_mqtt(num_detected_boxes, mqtt_client, mqtt_topic, key):
    mqtt_msg = f"{{\"v\":{num_detected_boxes}}}"
    result_mqtt = mqtt_client.publish(mqtt_topic, mqtt_msg)
    if result_mqtt.rc == mqtt.MQTT_ERR_SUCCESS:
        logger.info(f"[{key}] Publish succeeded : {mqtt_msg}")
    else:
        logger.error(f"[{key}] Publish failed with error code: {result_mqtt.rc}")

def detector_per_cam(cam, key, model, imgsz, args, vis, yolo_lock, detect_csv_writer, detect_csv_writer_lock):
    """
    Thread function for each cam (ex. of key: 'B1F_Food')
    """
    # grab global references to the output frame, and lock variables
    global outputFrame, lock, selected_key

    assert(key == args.spacenorm_device_key)
    assert(vis)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # FIXME: (2025.11.17) 아래의 값은 조명 변화에 의한 오인식을 완화하기 위한 테스트 값임.
    fgbg.setVarThreshold(50)
    fgbg.setDetectShadows(False)
    
    logger.info(f"[{key}] MOG2 : varThreshold = {fgbg.getVarThreshold()}")
    logger.info(f"[{key}] MOG2 : detectShadows = {fgbg.getDetectShadows()}")

    # logger.info(f"[{key}] Creat an RTSP_Camera..")
    # cam = RTSP_Camera(args, cctv_ID)
    
    # logger.info(f"[{key}] Camera is successfully opened")
    
    fps = 0.0
    temp_cnt = 0
    KAKAO_MSG_SEND_THRESHOLD = 200 # send kakao msg for every this number of 'read None's.
    prev_reported_boxes = -1
    consecutive_read_none = 0
    accumulated_time = 0.0    
    tic_toc = 0
    frame_idx = 0

    prevent_report = args.prevent_report
    
    consecutive_frame_with_persons = 0

    mqtt_broker_addr = None
    mqtt_broker_port = None
    mqtt_client = None
    if 'mqtt_broker_addr' in cctv_ID[key] and 'mqtt_broker_port' in cctv_ID[key]:
        # Initialize mqtt client
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        # print(f"{cctv_ID[key]['mqtt_broker_addr']}, {cctv_ID[key]['mqtt_broker_port']}")
        mqtt_broker_addr = cctv_ID[key]['mqtt_broker_addr']
        if mqtt_broker_addr:
            if mqtt_broker_addr == 'localhost':
                mqtt_broker_addr = '172.17.0.1' # FIXME: should be made more general
            mqtt_broker_port = cctv_ID[key]['mqtt_broker_port']
            mqtt_topic = f"spacenorm/{cctv_ID[key]['device_id']}/PS" # spacenorm/<디바이스ID>/PS
            mqtt_client.connect(mqtt_broker_addr, int(mqtt_broker_port))
            mqtt_client.subscribe(mqtt_topic)
            mqtt_client.loop_start()
            logger.info(f"[{key}] MQTT client connected to {mqtt_broker_addr}:{mqtt_broker_port} and subscribed to {mqtt_topic}")

    while not cam.isOpened(): # When the RTSP is not opened yet, we try to open here before entering the main detection loop below.
        logger.info(f"[{key}] cam.isOpened() = {cam.isOpened()} --> try to open RTSP")
        cam._open()
        time.sleep(2.0)

    prev_tic = time.time()
    while True:
        tic = time.time()

        duration_betn_read_calls = tic - prev_tic
        logger.debug(f"[{key}] duration bet'n cam.read() calls: {duration_betn_read_calls:.2f}s (report_period = {SPACENORM_REPORT_PERIOD_SEC}s)")
        prev_tic = tic

        img = cam.read()

        if img is None:
            logger.error(f"[{key}] cam.read() to return None --> retry to read() after short sleep")
            time.sleep(1.0)  # Short sleep before retrying
            continue

        (H, W) = img.shape[:2]

        toc4 = time.time()
        with yolo_lock:
            tic_yolo = time.time()
            
            boxes, confs, clss = yolo_inference_image(img, model, imgsz, args.conf_thresh)
            # format of boxes (list of xyxy):  [[566, 37, 600, 62], [710, 48, 787, 155]]

            toc_yolo = time.time()
            tic_toc_yolo = toc_yolo - tic_yolo
            logger.debug(f"[{key}] time taken in waiting for yolo_lock: {(tic_yolo-toc4)*1000:.2f}ms")
            logger.debug(f"[{key}] time taken by yolo_inference_image( ): {tic_toc_yolo*1000:.2f}ms ({1/tic_toc_yolo:.1f}fps)")

        toc2 = time.time()
        
        if args.detect_car_kumho:
            logger.debug(f"[{key}] before select_car_related_results() : boxes = {boxes}, confs = {confs}, clss = {clss}")
            boxes, confs, clss = select_car_related_results(boxes, confs, clss)
            logger.debug(f"[{key}] after  select_car_related_results() : boxes = {boxes}, confs = {confs}, clss = {clss}")
        else:
            boxes, confs, clss = filter_only_person(vis, boxes, confs, clss, img, key, False)

        if (len(boxes)>0) and ('min_obj_size_ratio' in cctv_ID[key]): # if there is min_obj_size_ratio, we remove boxes whose area is smaller than the ratio
            min_obj_size_ratio = cctv_ID[key]['min_obj_size_ratio'] # unit is %
            assert(min_obj_size_ratio > 0.0)
            logger.debug(f"[{key}] Before filter_small_objects(): num boxes = {len(boxes)}, boxes = {boxes}, min_obj_size_ratio = {min_obj_size_ratio}")
            boxes, confs, clss = filter_small_objects(key, boxes, confs, clss, min_obj_size_ratio, (H,W))

        if (len(boxes)>0) and ('ROI' in cctv_ID[key]): # if there is ROI information, we remove boxes outside the ROIs (polygons)
            logger.debug(f"[{key}] Before remove_outside_ROI(): num boxes = {len(boxes)}, boxes = {boxes}, ROI = {cctv_ID[key]['ROI']}")
            
            new_boxes, new_confs, new_clss = remove_outside_ROI(boxes, confs, clss, imgsz, cctv_ID[key]['ROI'], img, key, True) # cctv_ID[key]['ROI'] : {'img_w': 1920, 'img_h': 1080, 'vertices': [[[38, 528], [391, 491], [159, 1062], [521, 1041]]]}
            
            assert(len(new_boxes) == len(new_confs))
            assert(len(new_boxes) == len(new_clss))

            if (len(new_boxes) != len(boxes)):
                assert(len(new_boxes) < len(boxes))

                logger.debug(f"[{key}] After remove_outside_ROI(): num boxes = {len(new_boxes)}, boxes = {new_boxes}")

            boxes = new_boxes
            confs = new_confs
            clss = new_clss

        if args.background:
            if args.remove_background_bb:
                remove_member = True
            else:
                remove_member = False
            # remove or mark all bb's type in types list that is on the (assumed) background - ahnjw
            tic_check_bb = time.time()
            boxes, confs, clss, types, motionesses = check_bb_on_background(img, boxes, confs, clss, remove_member, args, kernel, fgbg)
            toc_check_bb = time.time()
            tic_toc_check_bb = toc_check_bb - tic_check_bb
        else:
            types = [0]*len(boxes) # when we do not check background bb - ahnjw,2020.11.22

        assert(cam.spacenorm_api) # following code is only valid when spacenorm_api is not None

        toc3 = time.time()

        num_detected_boxes = len(boxes)
        if temp_cnt < 5: # do not report for the first five frames for background bb detection
            temp_cnt += 1
            logger.info(f"[{key}] skip initial {temp_cnt}-th frame")
            continue

        elif num_detected_boxes != prev_reported_boxes:
        # We only report num_detected_boxes when it is different from prev_reported_boxes.
        # (2023.06.22) But, when num_detected_boxes != 0, we skip this frame if there are fewer consecutively 
        # reported frames with person(s) than args.hysteresis since it is highly probable that this frame's
        # detected person is false positive. (removing 'detection flicker')

            if (num_detected_boxes != 0) and (consecutive_frame_with_persons < int(args.hysteresis)):
                logger.debug(f"[{key}] {num_detected_boxes} objetect detected, but do not report {consecutive_frame_with_persons}-th frame due to hysteresis ({args.hysteresis})")
                consecutive_frame_with_persons += 1
                continue

            if num_detected_boxes == 0:
                consecutive_frame_with_persons = 0

            if not prevent_report:
                
                # If MQTT is enabled, publish the number of detected boxes to the MQTT broker
                if mqtt_broker_addr: # if MQTT is enabled
                    report_via_mqtt(num_detected_boxes, mqtt_client, mqtt_topic, key)
                
                r = cam.spacenorm_api.report('PS', cctv_ID[cam.spacenorm_device_key]['device_id'], num_detected_boxes, key)                
                logger.info("[{}] report # {}".format(key, num_detected_boxes))

                try: # to deal with the case that 'camera_snapshots' field is missing in r.json()['result'][0]
                    if 'camera_snapshots' in r.json()['result'][0]:            
                        if (len(r.json()['result'][0]['camera_snapshots']) > 0):                

                            # logger.info(f"[{key}] Try to add AR comment with files")
                            
                            img_for_AR = vis.draw_bboxes(img, boxes, confs, clss, types, motionesses)
                            
                            # timestamp = time.time()
                            # image_filename = f'./{PKG_NAME}/AR_images/{key}_{timestamp}.jpg'
                            
                            image_filename = f'./{PKG_NAME}/AR_images/{key}_{tic_yolo}.jpg'
                            cv2.imwrite(image_filename, img_for_AR)
            
                            logger.info(f"[{key}] --> AR coment with files: {image_filename}")
                            if args.detect_car_kumho:
                                msg = f"{num_detected_boxes} car(s) detected. detected_at: {tic_yolo}"
                            else:
                                msg = f"{num_detected_boxes} person(s) detected. detected_at: {tic_yolo}"

                            f = open(image_filename, 'rb')
                            
                            # cam.spacenorm_api.AR_comment_with_files(r, f, image_filename, msg, key)
                            # To deal with Korean charaters in image_filename: (ref: https://chat.openai.com/share/6bb6b9c4-163e-40ca-a7a6-549a37ae5e63)
                            cam.spacenorm_api.AR_comment_with_files(r, f, urllib.parse.quote(image_filename), msg, key)
                            
                            f.close()
                            logger.debug(f"[{key}] --> Try to remove {image_filename} ({f.name})")
                            os.remove(f.name)
                            logger.debug("[{key}] --> Succeeded")
                
                except Exception as e:
                    logger.error("[{key}] Bypassing exception from len(r.json()['result'][0]['camera_snapshots'])")
                    logger.error(f"[{key}] Exception type: {type(e).__name__}")
                    logger.error(f"[{key}] Exception value: {str(e)}")

                # # If MQTT is enabled, publish the number of detected boxes to the MQTT broker
                # if mqtt_broker_addr: # if MQTT is enabled
                #     report_via_mqtt(num_detected_boxes, mqtt_client, mqtt_topic, key)

            else:
                logger.info("[{}] detected # {}".format(key, num_detected_boxes))
                logger.info(f"[{key}] But reporting is prevented for testing ...")

            prev_reported_boxes = num_detected_boxes

        else:
            logger.info("[{}] same as before # {} -> do not report..".format(key, num_detected_boxes))

        # Record detection result for further investigation ------------------------------------
        if (num_detected_boxes > 0) and args.record_detection_result:

            with detect_csv_writer_lock:
                record_detection_result(key, boxes, confs, clss, types, motionesses, detect_csv_writer, img, vis, args, frame_idx)

            frame_idx = (frame_idx + 1) % args.max_images_in_output
        # --------------------------------------------------------------------------------------

        if args.web_streaming_port and selected_key == key:
            assert(vis is not None)
            img = vis.draw_bboxes(img, boxes, confs, clss, types, motionesses)
            img = show_fps(img, fps, key)
            with lock:
                outputFrame = img.copy()

        toc = time.time()
        tic_toc = toc-tic

        if (tic_toc < 0):
            logger.warning(f"[{key}] Error!! tic = {tic}, toc = {toc} --> tic_toc = toc-tic = {tic_toc}")
            assert(tic_toc > 0)
            
        curr_fps = 1.0 / tic_toc 
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)

        # tic : cam.read() 직전
        # toc2 : yolo_inference_image() 직후
        # toc3 : postprocessing 직후
        # toc : report 및 AR comment 전송 직후
        time_for_inference = toc2 - tic
        time_for_postprocessing = toc3 - toc2
        time_for_report = toc - toc3

        logger.info(f"[{key}] Total duration   : {tic_toc*1000:.2f}ms")
        logger.info(f"[{key}]   inference({time_for_inference*1000:.2f}ms) + postprocessing({time_for_postprocessing*1000:.2f}ms) + report({time_for_report*1000:.2f}ms)")
        logger.info(f"[{key}] Total frame rate : {fps:.2f} fps\n")
        
        remaining_time = SPACENORM_REPORT_PERIOD_SEC - tic_toc
        if remaining_time > 0:
            time.sleep(remaining_time)
        else:
            logger.warning("[{}] iteration took {:.2f}s, exceeding SPACENORM_REPORT_PERIOD_SEC (= {}s) -- proceeding without extra sleep".format(key, tic_toc, SPACENORM_REPORT_PERIOD_SEC))

        accumulated_time += time.time() - tic

        if (accumulated_time > SPACENORM_HEARTBEAT_PERIOD_SEC):
            logger.info(f"[{key}] Sending heartbeat..")
            cam.spacenorm_api.heartbeat([cctv_ID[key]['device_id']], key)

            # if do_not_report: # force reporting even in case of "do not report"
            #     if prev_reported_boxes >= 0: # to screen the case of just re-opened CCTV without any detection
            #         cam.spacenorm_api.report('PS', cctv_ID[key][0], prev_reported_boxes)
            if not prevent_report:
                if prev_reported_boxes >= 0: # to screen the case of just re-opened CCTV without any detection
                    cam.spacenorm_api.report('PS', cctv_ID[key]['device_id'], prev_reported_boxes, key)

                    # If MQTT is enabled, publish the number of detected boxes to the MQTT broker
                    if mqtt_broker_addr: # if MQTT is enabled
                        report_via_mqtt(prev_reported_boxes, mqtt_client, mqtt_topic, key)
            
            accumulated_time = 0            

# ------------------------- end of detector_per_cam() ---------------------------------------

def rtsp_read_watchdog(cameras, hang_sec=20):
    while True:
        now = time.time()
        logger.debug(f"(now = {now}) rtsp_read_watchdog is checking cam.read() status... ")
        for key in cameras:
            cam = cameras[key]
            with cam.read_lock:
                if cam.read_in_progress and (cam.last_read_enter_ts > cam.last_read_return_ts):
                    hung_for = now - cam.last_read_enter_ts
                    if hung_for >= hang_sec and not cam.read_hang_reported:
                        cam.read_hang_reported = True
                        logger.warning(
                            f"[{key}] cap.read() hung for {hung_for:.1f}s"
                        )
                else:
                    cam.read_hang_reported = False
        time.sleep(2)

def detect_multithreaded(model, imgsz, args, args_dict, vis, detect_csv):
    """Continuously capture images from multiple RTSP cameras and do object detection with a single YOLO model.
        - multithreaded version - ahnjw,2022.11.01

    # Arguments
      model: Ultralytics YOLO model instance
      args: input arguments containing various thresholds for detection and background removal.
      vis: for visualization.
      detect_csv: csv file for recording detection result for further investigation
    """

    logger.info("Start detect_multithreaded()........")

    if args.record_detection_result:
        assert(detect_csv is not None)
        detect_csv_writer = csv.writer(detect_csv)
        detect_csv_writer.writerow(['key', 'time', 'type', 'motion', 'conf', 'image'])
        #logger.info("detect_csv header is written")
        detect_csv_writer_lock = threading.Lock()
    else:
        detect_csv_writer = None
        detect_csv_writer_lock = None
    
    yolo_lock = threading.Lock()

    # Create a thread for each camera stream
    threads = []
    for key in args_dict: # ex. of key: 'B1F_Food'
        cam = RTSP_Camera(args_dict[key], cctv_ID)
        RTSP_Camera_dict[key] = cam
        detector_thread = threading.Thread(name=f"detector_per_cam[{key}]", target=detector_per_cam, args=(cam, key, model, imgsz, args_dict[key], vis, yolo_lock, detect_csv_writer, detect_csv_writer_lock))
        detector_thread.daemon = True
        detector_thread.start()
        threads.append(detector_thread)
        logger.info(f"   Thread for {key} started..")
        
    # Start a watchdog thread to monitor the status of cam.read() for each camera stream and to release and re-open the stream when cam.read() is detected to be hung for a certain period of time.
    watchdog_thread = threading.Thread(
        target=rtsp_read_watchdog,
        args=(RTSP_Camera_dict,),
        daemon=True,
        name="rtsp_read_watchdog"
    )
    watchdog_thread.start()
    logger.info(f"   rtsp_read_watchdog thread started..")


    thread_count_metric = Gauge('thread_count', 'Number of threads running')

    # process_name = args.cfg.split('/')[-1].split('.')[0] + '_재실감지' # ex: "cym_재실감지"
    # process_name = args.process_name

    num_keys = len(args_dict)
    while True:
        time.sleep(120) # sleep for 120 seconds before checking thread status
        logger.info('='*20+' thread info '+'='*20)                    
        
        thread_count=0
        grab_img_thread_count = 0
        
        for t in threading.enumerate():
            logger.info(f"thread {thread_count} : {t.name}")
            thread_count += 1
            if 'grab_img' in t.name:
                grab_img_thread_count += 1
        
        logger.info(f"--> {grab_img_thread_count} grab_img threads running (num_keys = {num_keys})")
        logger.info(f"--> Total {thread_count} threads running")
        
        if num_keys != grab_img_thread_count:
            logger.warning(f"Number of grab_img threads ({grab_img_thread_count}) is different from number of keys ({num_keys})!!")

        logger.info("="*53)
        # thread_count_metric.labels(process=process_name).set(thread_count)
        thread_count_metric.set(thread_count)

    # for t in threads:
    #     t.join()

def detector_YOLO_multithreaded(args):
    logger.info("detector_YOLO_multithreaded()...")

    if args.category_num <= 0:
        raise ValueError('ERROR: bad category_num (%d)!' % args.category_num)

    if not os.path.isfile(args.model):
        logger.error('ERROR: file (%s) not found!' % args.model)
        sys.exit(1)

    if getattr(args, "web_streaming_port", None):
        if not args.no_display:
            logger.error('ERROR: web streaming requires no_display')
            sys.exit(1)

    if not getattr(args, "spacenorm_devicekey_file", None):
        logger.error("args.spacenorm_devicekey_file must be set")
        sys.exit(1)

    args_dict = process_spacenorm_devicekey_file(args)
    for key in args_dict:
        spacenorm_key_list.append(key)
    logger.info(f"spacenorm_key_list = {spacenorm_key_list}")

    cls_dict = get_cls_dict(args.category_num)
    logger.debug(f"cls_dict = {cls_dict}")

    logger.info("Loading YOLO model...")
    model = create_yolo_model(args.model)
    imgsz = args.img_size

    vis = BBoxVisualization(cls_dict)

    with open(os.path.join(args.log_dir, "traceback.log"), "w") as error_log:
        try:
            if getattr(args, "record_detection_result", None):
                org_images_dir = os.path.join(args.record_detection_result_dir, "org")
                detected_images_dir = os.path.join(args.record_detection_result_dir, "detected")
                os.makedirs(org_images_dir, exist_ok=True)
                os.makedirs(detected_images_dir, exist_ok=True)
                detect_csv_path = os.path.join(args.record_detection_result_dir, "detect.csv")
                with open(detect_csv_path, "w", newline='') as detect_csv:
                    detect_multithreaded(model, imgsz, args, args_dict, vis, detect_csv)
            else:
                detect_multithreaded(model, imgsz, args, args_dict, vis, detect_csv=None)
        except Exception:
            traceback.print_exc()
            traceback.print_exc(file=error_log)


@app.route("/", methods=["GET"])
def index():
    global outputFrame, selected_key, lock

    with lock:
        outputFrame = None
    selected_key = request.args.get('cctv_key')
    logger.info(f"index(): selected_key = {selected_key}")

    return render_template("index.html", keys=spacenorm_key_list,
        where_str=cctv_space_name, when_str=None)


def generate():
    global outputFrame, lock

    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encodedImage) + b'\r\n')
        time.sleep(SPACENORM_REPORT_PERIOD_SEC)


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


ACCESS_TOKEN_REFRESH_INTERVAL_HOUR = 5 # access toekn for kakaotalk REST API is valid only for 6 hours
def refresh_kakao_access_token():
    while(1):

        print("refresh kakao access token")    
        logger.info("refresh kakao access token")

        with open(KAKAO_CODE_JSON_FILE,"r") as fp:
            tokens = json.load(fp)

        refresh_token = tokens['refresh_token']
        
        new_tokens = refreshToken(refresh_token)

        assert('access_token' in new_tokens)
        tokens['access_token'] = new_tokens['access_token']

        if ('refresh_token' in new_tokens):
            tokens['refresh_token'] = new_tokens['refresh_token']

        with lock_kakao_code:
            with open(KAKAO_CODE_JSON_FILE,"w") as fp:
                json.dump(tokens, fp)

        print(f"--> new access token is written to {KAKAO_CODE_JSON_FILE}")
        logger.info(f"--> new access token is written to {KAKAO_CODE_JSON_FILE}")

        time.sleep(ACCESS_TOKEN_REFRESH_INTERVAL_HOUR * 3600)

# construct cctv_ID from json configuration file to use legacy code
def init_cctv_data(cfg_filepath):
    global cctv_ID

    with open(cfg_filepath, "r") as fp:
        companies_cfg = json.load(fp)

        # companies_cfg['companies'] is a list of dictionaries
        # company is a dictionary with keys "company_name", "gateway_id", "access_token", "refresh_token", "CCTV".
        for company in companies_cfg['companies']: 
            company_name = company['company_name']
            access_token = company['access_token']
            refresh_token = company['refresh_token']
            
            for cctv in company['CCTV']:
                logger.debug(cctv)
                cctv_key = company_name + '_' + cctv # ex of cctv_key : "정양산업_B1F_food1"
                cctv_ID[cctv_key] = {} #{'device_id': None, 'uri': None, 'monitor_id': None, 'access_token': None, 'refresh_token': None, 'ROI': None, 'min_obj_size_ratio': None, 'mqtt_broker_addr': None, 'mqtt_broker_port': None}
                cctv_ID[cctv_key]['device_id'] = company['CCTV'][cctv]['device_id']
                cctv_ID[cctv_key]['uri'] = company['CCTV'][cctv]['uri']
                cctv_ID[cctv_key]['monitor_id'] = company['CCTV'][cctv]['monitor_id'] # FIXME: monitor_id가 없는 경우의 처리 필요
                cctv_ID[cctv_key]['access_token'] = access_token
                cctv_ID[cctv_key]['refresh_token'] = refresh_token
                if 'ROI' in company['CCTV'][cctv]:
                    cctv_ID[cctv_key]['ROI'] = company['CCTV'][cctv]['ROI'] # ex)) {'img_w': 1920, 'img_h': 1080, 'vertices': [[[38, 528], [391, 491], [159, 1062], [521, 1041]]]}
                if 'min_obj_size_ratio' in company['CCTV'][cctv]:
                    cctv_ID[cctv_key]['min_obj_size_ratio'] = company['CCTV'][cctv]['min_obj_size_ratio']

                if 'mqtt_broker_addr' in company['CCTV'][cctv]:
                    cctv_ID[cctv_key]['mqtt_broker_addr'] = company['CCTV'][cctv]['mqtt_broker_addr']
                if 'mqtt_broker_port' in company['CCTV'][cctv]:
                    cctv_ID[cctv_key]['mqtt_broker_port'] = company['CCTV'][cctv]['mqtt_broker_port']

    # print(cctv_ID)
    # exit(0)
#=======================================================================================================

def main():
    global spacenorm_key_list, SPACENORM_REPORT_PERIOD_SEC

    loader = SpacenormConfigLoader()
    cfg = loader.load()

    SPACENORM_REPORT_PERIOD_SEC = cfg.report_period

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(cfg.log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info(f"package name: '{PKG_NAME}'")
    logger.info("\nLoaded configuration:")
    logger.info(cfg)

    init_cctv_data(cfg.spacenorm_sensor_cfg_file)

    t1 = threading.Thread(name='detector_YOLO_multithreaded', target=detector_YOLO_multithreaded, args=(cfg,))
    t1.daemon = True
    t1.start()

    port_number = getattr(cfg, "web_streaming_port", None)
    if port_number is not None:
        def get_default_host_ip():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
            except Exception:
                ip = "127.0.0.1"
            finally:
                s.close()
            return ip

        host_ip = getattr(cfg, "host_ip", None) or get_default_host_ip()
        logger.info(f"Web streaming host IP: {host_ip}")
        app.run(host=host_ip, port=port_number, debug=True, threaded=True, use_reloader=False)

    t1.join()
        
if __name__ == '__main__':
    main()
