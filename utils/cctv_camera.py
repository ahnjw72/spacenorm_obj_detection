"""camera.py

This code implements the Camera class, which encapsulates code to
handle IP CAM, USB webcam or the Jetson onboard camera.  In
addition, this Camera class is further extended to take a video
file or an image file as input.

"""
# NOTE: added functionality of handling image list txt file -- ahnjw,2020.11.01

import logging
import threading
import subprocess

import numpy as np
import cv2
import os
import time

from .gateway_api import Gateway # spacenorm Gateway API - ahnjw,2020.11.11

#from queue import Queue
#IMG_QUEUE_SIZE = 1


# The following flag ise used to control whether to use a GStreamer
# pipeline to open USB webcam source.  If set to False, we just open
# the webcam using cv2.VideoCapture(index) machinery. i.e. relying
# on cv2's built-in function to capture images from the webcam.
USB_GSTREAMER = True

#logger = logging.getLogger('spacenorm_person_detect')
logger = logging.getLogger(__name__)

cctv_ID_deprecated = {     
    # device ID, uri, monitor ID, access token, refresh token for each CCTV IP Camera - ahnjw,2023.03.28
    'B2F_machine_room':('73706163656e6f726d5f63616d657260','rtsp://cym-gamcheon.iptime.org:11087/profile4/media.smp', 'Gdvu1tX6Eq', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'B1F_food1':('73706163656e6f726d5f63616d657261','rtsp://cym-gamcheon.iptime.org:11082/profile4/media.smp', 'Gdvv1tX6Eq', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'B1F_food2':('73706163656e6f726d5f63616d65727d','rtsp://cym-gamcheon.iptime.org:11137/profile4/media.smp', '5hjVTQ5Hpe', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'B1F_food3':('73706163656e6f726d5f63616d65727e','rtsp://cym-gamcheon.iptime.org:11142/profile4/media.smp','XqGLqxsmfc', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'B1F_B101':('73706163656e6f726d5f63616d657262','rtsp://cym-gamcheon.iptime.org:11052/profile4/media.smp', 'QjEwMQ', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '1F_dock-roadside1':('73706163656e6f726d5f63616d657263','rtsp://cym-gamcheon.iptime.org:11077/profile4/media.smp', 'Gdvu1tX6Ep', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '1F_dock-roadside2':('73706163656e6f726d5f63616d657272','rtsp://cym-gamcheon.iptime.org:11097/profile4/media.smp', '7ZWY7Jet7J6l64E66GcMg', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '1F_dock-roadside3':('73706163656e6f726d5f63616d65727f','rtsp://cym-gamcheon.iptime.org:11147/profile4/media.smp','cuQrHcCSJj', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '1F_dock-seaside1':('73706163656e6f726d5f63616d657264','rtsp://cym-gamcheon.iptime.org:11072/profile4/media.smp', 'Gevu1tX6Eq', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    
    '1F_dock-seaside2':('73706163656e6f726d5f63616d657273','rtsp://cym-gamcheon.iptime.org:11092/profile4/media.smp', '7ZWY7Jet7J6l67CU64ukMg', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    # 333

    '1F_dock-seaside3':('73706163656e6f726d5f63616d657280','rtsp://cym-gamcheon.iptime.org:11152/profile4/media.smp','vu3SG8jJXO', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '2F_office':('73706163656e6f726d5f63616d657265','rtsp://cym-gamcheon.iptime.org:11067/profile5/media.smp', 'Kdvu1tX6Eq', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '2F_201':('73706163656e6f726d5f63616d657266','rtsp://cym-gamcheon.iptime.org:11022/profile4/media.smp', 'MjAx', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '2F_202':('73706163656e6f726d5f63616d657267','rtsp://cym-gamcheon.iptime.org:11057/profile4/media.smp', 'MjAy', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '3F_301':('73706163656e6f726d5f63616d657268','rtsp://cym-gamcheon.iptime.org:11017/profile4/media.smp', 'MzAx', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '3F_302':('73706163656e6f726d5f63616d657269','rtsp://cym-gamcheon.iptime.org:11062/profile4/media.smp', 'MzAy', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '4F_401':('73706163656e6f726d5f63616d65726a','rtsp://cym-gamcheon.iptime.org:11027/profile4/media.smp', 'NDAx', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '4F_402':('73706163656e6f726d5f63616d65726b','rtsp://cym-gamcheon.iptime.org:11047/profile4/media.smp', 'NDAy', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '5F_501':('73706163656e6f726d5f63616d65726c','rtsp://cym-gamcheon.iptime.org:11007/profile4/media.smp', 'NTAx', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '5F_502':('73706163656e6f726d5f63616d65726d','rtsp://cym-gamcheon.iptime.org:11042/profile4/media.smp', 'NTAy', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '6F_601':('73706163656e6f726d5f63616d65726e','rtsp://cym-gamcheon.iptime.org:11012/profile4/media.smp', 'NjAx', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '6F_602':('73706163656e6f726d5f63616d65726f','rtsp://cym-gamcheon.iptime.org:11032/profile4/media.smp', 'NjAy', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '7F_701':('73706163656e6f726d5f63616d657270','rtsp://cym-gamcheon.iptime.org:11002/profile4/media.smp', 'NzAx', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '7F_702':('73706163656e6f726d5f63616d657271','rtsp://cym-gamcheon.iptime.org:11037/profile4/media.smp', 'NzAy', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'ROOF_greenhouse':('73706163656e6f726d5f63616d657274','rtsp://cym-gamcheon.iptime.org:11102/profile4/media.smp', '7Jil7IOB7ZWY7Jqw7Iqk64K067aACg', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
 
    'openfield_1':('73706163656e6f726d5f63616d657275','rtsp://admin:Qwert12%23@openfield.iptime.org:11094/profile4/media.smp', '64yA7KCA64W47KeAXzE', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    # 380

    'openfield_2':('73706163656e6f726d5f63616d657276','rtsp://admin:Qwert12%23@openfield.iptime.org:11097/profile4/media.smp', '64yA7KCA64W47KeAXzI', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    # 381

    'openfield_3':('73706163656e6f726d5f63616d657281','rtsp://admin:Qwert12%23@openfield.iptime.org:11100/profile4/media.smp', 'WrcmrL1kiq', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    # 460

    'openfield_4':('73706163656e6f726d5f63616d657282','rtsp://admin:Qwert12%23@openfield.iptime.org:11103/profile4/media.smp', 'hVwIRxQylW', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    # 461 (밭 < 대저동 < 정양산업)

    'kiosk_roadside':('73706163656e6f726d5f63616d657277','rtsp://cym-gamcheon.iptime.org:11107/profile4/media.smp', 'UOHbV6ApMH', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    
    'kiosk_seaside':('73706163656e6f726d5f63616d657278','rtsp://cym-gamcheon.iptime.org:11112/profile4/media.smp', 'm9QgiP1miU', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    # 435

    'B1F_food_warehouse1':('73706163656e6f726d5f63616d657279','rtsp://cym-gamcheon.iptime.org:11117/profile4/media.smp', '7x9XBfVQN9', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'B1F_food_warehouse2':('73706163656e6f726d5f63616d65727a','rtsp://cym-gamcheon.iptime.org:11122/profile4/media.smp', 'sbGLiPa5wx', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'B1F_food_warehouse3':('73706163656e6f726d5f63616d65727b','rtsp://cym-gamcheon.iptime.org:11127/profile4/media.smp', '6HFRgFsbuA', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'B1F_food_warehouse4':('73706163656e6f726d5f63616d65727c','rtsp://cym-gamcheon.iptime.org:11132/profile4/media.smp', 'Oe7eTFZIjO', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),

    'Press_scrap_paper' : ('73706163656e6f726d5f63616d657283','rtsp://admin:Qwert12%23@119.207.239.126:11097/profile4/media.smp', 'RJvaOtwrHS', 'gMivuUBsxb6MG_nuAYWoDaYK-tc_YrJ_rIILc6noOVM', 'VkpzjaKgHJ8KHHONHXmBC1rpEESjf7Optgs-8vyGWcI'),

    'KUMHO_security1' : ('73706163656e6f726d5f63616d657284','rtsp://admin:Qwert12%23@221.152.97.59:11094/profile4/media.smp', 'cPkg5uqXpC', 'W60YEc6Uf460_QRGLLcuoXRmHD6bX9YrqLn5NeN1fSs', 'olgkltf3j9vFZeU_sQCjTzWuyVT0iNt1RuNGZBrM6RI'),
    'KUMHO_security2' : ('73706163656e6f726d5f63616d657285','rtsp://admin:Qwert12%23@221.152.97.59:11097/profile4/media.smp', 'fwPLiCI39W', 'W60YEc6Uf460_QRGLLcuoXRmHD6bX9YrqLn5NeN1fSs', 'olgkltf3j9vFZeU_sQCjTzWuyVT0iNt1RuNGZBrM6RI'),
    'KUMHO_warehouse1' : ('73706163656e6f726d5f63616d657286','rtsp://admin:Qwert12%23@221.152.97.59:11100/profile4/media.smp', 'hGfYcInSXs', 'W60YEc6Uf460_QRGLLcuoXRmHD6bX9YrqLn5NeN1fSs', 'olgkltf3j9vFZeU_sQCjTzWuyVT0iNt1RuNGZBrM6RI'),
    'KUMHO_warehouse2' : ('73706163656e6f726d5f63616d657287','rtsp://admin:Qwert12%23@221.152.97.59:11103/profile4/media.smp', 'MiaPxO0p0c', 'W60YEc6Uf460_QRGLLcuoXRmHD6bX9YrqLn5NeN1fSs', 'olgkltf3j9vFZeU_sQCjTzWuyVT0iNt1RuNGZBrM6RI'),

}

def add_camera_args(parser):
    """Add parser augument for camera options."""
    parser.add_argument('--spacenorm_device_key', default=None,
                        help=('human occupancy detection sensor nodes for spacenorm [False]'))
    parser.add_argument('--image', type=str, default=None,
                        help='image file name, e.g. dog.jpg')
    parser.add_argument('--image_list', type=str, default=None,
                        help='image list txt file name, e.g. test_set01.txt')
    parser.add_argument('--video', type=str, default=None,
                        help='video file name, e.g. traffic.mp4')
    parser.add_argument('--video_looping', action='store_true',
                        help='loop around the video file [False]')
    parser.add_argument('--rtsp', type=str, default=None,
                        help=('RTSP H.264 stream, e.g. '
                              'rtsp://admin:123456@192.168.1.64:554'))
    parser.add_argument('--rtsp_latency', type=int, default=200,
                        help='RTSP latency in ms [200]')
    parser.add_argument('--usb', type=int, default=None,
                        help='USB webcam device id (/dev/video?) [None]')
    parser.add_argument('--onboard', type=int, default=None,
                        help='Jetson onboard camera [None]')
    parser.add_argument('--copy_frame', action='store_true',
                        help=('copy video frame internally [False]'))
    parser.add_argument('--do_resize', action='store_true',
                        help=('resize image/video [False]'))
    parser.add_argument('--width', type=int, default=640,
                        help='image width [640]')
    parser.add_argument('--height', type=int, default=480,
                        help='image height [480]')
    return parser


def open_cam_rtsp(uri, width, height, latency, key):
    """Open an RTSP URI (IP CAM)."""
    logger.info(f"[{key}] Open an RTSP URI ({uri})")

    """
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))

    
    if 'omxh264dec' in gst_elements:
        print("omxh264dec")
        logger.info("omxh264dec")
        # Use hardware H.264 decoder on Jetson platforms
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! omxh264dec ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! videoconvert ! '
                   'appsink').format(uri, latency, width, height)
        print("Use hardware H.264 decoder on Jetson platforms")
        logger.info("Use hardware H.264 decoder on Jetson platforms")
        capture = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    elif 'avdec_h264' in gst_elements:
        # Otherwise try to use the software decoder 'avdec_h264'
        # NOTE: in case resizing images is necessary, try adding
        #       a 'videoscale' into the pipeline
        print("avdec_h264")
        logger.info("avdec_h264")
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! avdec_h264 ! '
                   'videoconvert ! appsink').format(uri, latency)
        print("Try to use the software decoder avdec_h264")
        logger.info("try to use the software decoder avdec_h264")
        capture = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    else:
        print("using cv2 uri capture :",uri)
        logger.info(f"using cv2 uri capture :{uri}")
        capture = cv2.VideoCapture(uri)
    """

    logger.info(f"[{key}] using cv2 uri capture :{uri}")
    capture = cv2.VideoCapture(uri)

    if capture.isOpened():
        logger.info(f"[{key}] open_cam_rtsp(): successful cv2.VideoCapture")
        return capture
    else:
        logger.info(f"[{key}] open_cam_rtsp(): capture.isOpened() returns false")
        return None
        

"""
def open_cam_rtsp_cctv(uri): # avdec_h264 doesn't work -- ahnjw,2020.11.08
    return cv2.VideoCapture(uri)
"""

def get_gst_pipeline_old(uri, latency=200):
    """
    The 'Universal' GStreamer string.
    - protocols=tcp: Essential for Docker and stable streaming.
    - latency: Lower is better for real-time, higher is better for stability.
    - decodebin: Still used, but wrapped in a way to force video caps.
    """
    return (
        f"rtspsrc location={uri} protocols=tcp latency={latency} ! "
        "application/x-rtp, media=video ! " # Force the bin to ignore audio/metadata
        "decodebin ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! " # Pre-convert to BGR for OpenCV performance
        "appsink drop=true max-buffers=1 emit-signals=false" 
    )

def get_gst_pipeline_old2(uri, latency=200):
    """
    An adaptive GStreamer pipeline that forces TCP, auto-detects 
    the video codec (H264/H265), and pre-buffers for OpenCV BGR.
    """
    return (
        f"rtspsrc location={uri} protocols=tcp latency={latency} drop-on-latency=true ! "
        "application/x-rtp, media=video ! " # 1. Force GStreamer to ignore audio/metadata tracks
        "decodebin ! "                      # 2. Auto-detect H.264, H.265, MJPEG, etc.
        "videoconvert ! "                   # 3. Convert color spaces
        "video/x-raw, format=BGR ! "        # 4. Force BGR output (Fastest for OpenCV)
        "appsink drop=true max-buffers=1"   # 5. Drop old frames so you don't lag
    )

def get_gst_pipeline(uri, latency=200, timeout_ms=5000):
    """
    An adaptive GStreamer pipeline that forces TCP, auto-detects 
    the video codec (H264/H265), and pre-buffers for OpenCV BGR.
    """

    # Convert ms to microseconds for GStreamer
    tcp_timeout = timeout_ms * 1000

    return (
        f"rtspsrc location={uri} protocols=tcp latency={latency} tcp-timeout={tcp_timeout} ! "
        "application/x-rtp, media=video ! " # 1. Force GStreamer to ignore audio/metadata tracks
        "decodebin ! "                      # 2. Auto-detect H.264, H.265, MJPEG, etc.
        "videoconvert ! "                   # 3. Convert color spaces
        "video/x-raw, format=BGR ! "        # 4. Force BGR output (Fastest for OpenCV)
        "appsink drop=true max-buffers=1"   # 5. Drop old frames so you don't lag
    )

def open_rtsp_universal(key, uri, timeout_ms=10000): # Try to open RTSP stream with multiple backends (2026.01.09)
    # 1. Try GStreamer (Optimized)
    logger.debug(f"[{key}] Attempting GStreamer: {uri}")
    pipeline = get_gst_pipeline(uri)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    start_time = time.time()    
    logger.debug(f"[{key}] Waiting for GStreamer backend to open...")
    while time.time() - start_time < (timeout_ms / 1000):
        if cap.isOpened():
            # Verify we can actually grab a frame (GStreamer sometimes says 
            # opened but fails on first read)            
            if cap.grab():
                logger.info(f"[{key}] GStreamer backend initialized via TCP.")
                return cap
        # logger.info(f"[{key}] GStreamer backend not ready yet, retrying...")
        time.sleep(0.1)

    # 2. Fallback to FFmpeg (The 'Old Reliable')
    cap.release()
    cap = cv2.VideoCapture(uri) # Default OpenCV backend (FFmpeg)
    logger.info(f"[{key}] GStreamer failed or timed out. Falling back to FFmpeg...")
    
    if cap.isOpened():
        logger.info(f"[{key}] open_rtsp_universal(): successful cv2.VideoCapture")
        return cap
    else:
        logger.info(f"[{key}] open_rtsp_universal(): capture.isOpened() returns false")
        
    return None # Both failed


def open_cam_usb(dev, width, height):
    """Open a USB webcam."""
    if USB_GSTREAMER:
        gst_str = ('v4l2src device=/dev/video{} ! '
                   'video/x-raw, width=(int){}, height=(int){} ! '
                   'videoconvert ! appsink').format(dev, width, height)
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    else:
        return cv2.VideoCapture(dev)


def open_cam_onboard(width, height):
    """Open the Jetson onboard camera."""
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, you might need to add
        # 'flip-method=2' into gst_str below.
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_image_list(image_list_filepath):
    with open(image_list_filepath) as image_list_file:
        content = [line.rstrip() for line in image_list_file]
        #print(content)
        #sorted(content, reverse=True)
    return ImageFileListReader(sorted(content, reverse=True))

def grab_img_old5(cam):
    key = cam.spacenorm_device_key
    logger.info(f"[{key}] [grab_img() thread] starts.")

    with cam.thread_running_lock:
        thread_running = cam.thread_running

    while thread_running:
        img_handle = None

        try:
            local_cap = cam.cap
            if local_cap is not None and local_cap.isOpened():
                success, img_handle = local_cap.read()
                if not success:
                    img_handle = None
            else:
                img_handle = None
        except Exception as e:
            logger.error(f"[{key}] Exception in cap.read() in grab_img(): {e}")
            img_handle = None  
        
        with cam.read_lock:
            cam.img_handle = img_handle

        if img_handle is None:
            time.sleep(0.1)

        with cam.thread_running_lock:
            thread_running = cam.thread_running

    if local_cap is not None:
        logger.info(f"[{key}] [grab_img() thread] try to release local_cap")
        local_cap.release()  # Ensure we release the capture if we exit the loop
    else:
        logger.info(f"[{key}] [grab_img() thread] local_cap is None, no need to release.")

    logger.info(f"[{key}] [grab_img() thread] exiting.")

def grab_img_old4(cam):
    key = cam.spacenorm_device_key
    logger.info(f"[{key}] [grab_img() thread] starts.")

    with cam.thread_running_lock:
        thread_running = cam.thread_running

    while thread_running:
        img_handle = None

        locked = cam.cap_lock.acquire(timeout=1.0)
        if locked:
            try:
                local_cap = cam.cap
                if local_cap is None:
                    break

                if local_cap.isOpened():
                    success, img_handle = local_cap.read()
                    if not success:
                        img_handle = None
            except Exception as e:
                logger.error(f"[{key}] Exception in grab_img read: {e}")
                img_handle = None
            finally:
                cam.cap_lock.release()

        with cam.read_lock:
            cam.img_handle = img_handle

        if img_handle is None:
            time.sleep(0.1)

        with cam.thread_running_lock:
            thread_running = cam.thread_running

    logger.info(f"[{key}] [grab_img() thread] exiting.")

def grab_img_old3(cam):
    key = cam.spacenorm_device_key
    logger.info(f"[{key}] [grab_img() thread] starts.")

    with cam.thread_running_lock:
        thread_running = cam.thread_running
    
    while thread_running:
        # 1. Capture a local reference to the capture object.
        # This ensures that even if cam.release() sets cam.cap = None,
        # this thread still holds a valid reference until this loop iteration ends.
        local_cap = cam.cap
        
        if local_cap is None:
            break

        img_handle = None
        # 2. Use a timeout on the lock to avoid being stuck if release() is struggling
        locked = cam.cap_lock.acquire(timeout=1.0)
        if locked:
            try:
                if local_cap.isOpened():
                    # This is the line that can hang due to network/driver issues
                    success, img_handle = local_cap.read()
                    if not success:
                        img_handle = None
            except Exception as e:
                logger.error(f"[{key}] Exception in grab_img read: {e}")
                img_handle = None
            finally:
                cam.cap_lock.release()
        
        # 3. Update the shared frame handle
        with cam.read_lock:
            cam.img_handle = img_handle

        # Small sleep if read failed to prevent CPU spiking during disconnects
        if img_handle is None:
            time.sleep(0.1)

        with cam.thread_running_lock:
            thread_running = cam.thread_running

    logger.info(f"[{key}] [grab_img() thread] exiting.")


def grab_img_old2(cam):
    key = cam.args.spacenorm_device_key
    logger.info(f"[{key}] [grab_img() thread] starts.")

    while cam.thread_running:
        # Wrap the OpenCV call in the cap_lock
        with cam.cap_lock:
            if cam.cap and cam.cap.isOpened():
                try:
                    # This is the line that usually hangs
                    filename, img_handle = cam.cap.read()
                except Exception:
                    img_handle = None
            else:
                img_handle = None

        with cam.read_lock:
            cam.img_handle = img_handle


def grab_img_old(cam):
    """This 'grab_img' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab a new image and put it
    into the global 'img_handle', until 'thread_running' is set to False.
    """
    key = cam.args.spacenorm_device_key
    logger.info(f"[{key}] [grab_img() thread] starts.")

    wait_sec = 2
    while cam.thread_running:
        #print(f"cam.cap.isOpened() = {cam.cap.isOpened()}")
        if cam.cap.isOpened(): # ahnjw,2020.12.14

            try:
                filename, img_handle = cam.cap.read()
            except Exception as e:
                # Handle the exception here
                # logger.warning(f"[{key}] [grab_img() thread] Exception from cam.cap.read() : {e} --> wait {wait_sec}sec and retry..")
                logger.warning(f"[{key}] [grab_img() thread] Exception from cam.cap.read() : {e} --> img_handle becomes None")
                img_handle = None
                time.sleep(wait_sec)
                continue
            """
            if img_handle is None:
                
                logger.warning(f"[{key}] [grab_img() thread] Failed to read frame. Reinitializing capture object.")
                cam.release()
                time.sleep(wait_sec)
                cam._open()
                logger.warning(f"[{key}] [grab_img() thread] Capture object is reinitialized.")
                
                continue
            """
            if type(filename) == str:
                cam.img_file_name = filename
            else:
                cam.img_file_name = None

            # cam.read_lock.acquire()
            # cam.img_handle = img_handle
            # cam.read_lock.release()
            with cam.read_lock:
                cam.img_handle = img_handle

            if img_handle is None:
                logger.warning(f"[{key}] [grab_img() thread] img_handle is None --> break this loop to stop grab_img() thread.")
                break
                
        else:
            logger.warning(f"[{key}] [grab_img() thread] cam.cap.isOpened() returns False --> img_handle becomes None")
            img_handle = None
            time.sleep(wait_sec)

    logger.warning(f"[{key}] [grab_img() thread] exits..")
    print(f"[{key}] [grab_img() thread] exits..")

    cam.thread_running = False

def grab_img_old6(cam):
    key = cam.spacenorm_device_key
    sleep_sec = 1.0

    logger.info(f"[{key}] [grab_img() thread] starts.")

    with cam.thread_running_lock:
        thread_running = cam.thread_running

    while thread_running:
        img_handle = None

        try:
            local_cap = cam.cap
            if local_cap is not None and local_cap.isOpened():

                with cam.read_lock:
                    cam.last_read_enter_ts = time.time()
                    cam.read_in_progress = True

                success, img_handle = local_cap.read()

                with cam.read_lock:
                    cam.last_read_return_ts = time.time()
                    cam.read_in_progress = False                
                    if cam.last_read_return_ts - cam.last_read_enter_ts > 5.0:  # If read() takes more than 5 seconds, log a warning
                        logger.warning(f"[{key}] cap.read() took {cam.last_read_return_ts - cam.last_read_enter_ts:.2f} seconds (success: {success}), which is unusually long.")

                if not success:
                    logger.debug(f"[{key}] cap.read() returned False, setting img_handle to None")
                    img_handle = None
            else:
                logger.debug(f"[{key}] local_cap is None or not opened, setting img_handle to None")
                img_handle = None
        except Exception as e:
            logger.error(f"[{key}] Setting img_handle to None due to the Exception in cap.read() in grab_img(): {e}")
            img_handle = None  
        
        with cam.read_lock:
            if img_handle is None:
                logger.debug(f"[{key}] Updating cam.img_handle to None")
            cam.img_handle = img_handle

        if img_handle is None:
            logger.debug(f"[{key}] img_handle is None, sleeping for a short time ({sleep_sec} seconds) before retrying...")
            time.sleep(sleep_sec)

        with cam.thread_running_lock:
            thread_running = cam.thread_running

    if local_cap is not None:
        logger.info(f"[{key}] [grab_img() thread] try to release local_cap")
        local_cap.release()  # Ensure we release the capture if we exit the loop
    else:
        logger.info(f"[{key}] [grab_img() thread] local_cap is None, no need to release.")

    logger.info(f"[{key}] [grab_img() thread] exiting.")

def grab_img(cam):
    key = cam.spacenorm_device_key
    sleep_sec = 1.0

    logger.info(f"[{key}] [grab_img() thread] starts.")

    while True:
        img_handle = None
        
        local_cap = cam.cap
        if local_cap is not None and local_cap.isOpened():

            with cam.read_lock:
                cam.last_read_enter_ts = time.time()
                cam.read_in_progress = True

            success, img_handle = local_cap.read()

            with cam.read_lock:
                cam.last_read_return_ts = time.time()
                cam.read_in_progress = False                
                if cam.last_read_return_ts - cam.last_read_enter_ts > 5.0:  # If read() takes more than 5 seconds, log a warning
                    logger.warning(f"[{key}] cap.read() took {cam.last_read_return_ts - cam.last_read_enter_ts:.2f} seconds (success: {success}), which is unusually long.")

            if not success:
                logger.debug(f"[{key}] cap.read() returned False -> set img_handle to None and release the capture object to free resources, then try to reconnect in the next iteration.")
                cam.cap.release()  # Release the capture object if it's not opened to free resources
                cam.cap = None  # Set to None to indicate we need to recreate it
                continue  # Skip the rest of the loop and try again to get a new capture object in the next iteration
        else:
            logger.debug(f"[{key}] local_cap is None or not opened --> recreate the capture object.")
            local_cap = open_rtsp_universal(key, cam.cctv_info[key]['uri'])
            if local_cap is None:
                logger.info("Retrying connection in 5 seconds...")
                time.sleep(5)
            else:
                logger.info(f"[{key}] Successfully reconnected to the RTSP stream.")

            cam.cap = local_cap
            
            continue
        
        with cam.read_lock:
            if img_handle is None:
                logger.debug(f"[{key}] img_handle is None, try to read again after sleeping for a short time ({sleep_sec} seconds)...")
                time.sleep(sleep_sec)  # Sleep before retrying to read
            else:
                cam.img_handle = img_handle


class RTSP_Camera():
    def __init__(self, args, cctv_info):
        # print("\n\nargs = {}".format(args))
        self.args = args
        self.cctv_info = cctv_info
        self.is_opened = False
        self.spacenorm_device_key = args.spacenorm_device_key
        self.spacenorm_api = None
        self.thread_running = False
        self.img_handle = None
        self.copy_frame = args.copy_frame
        self.do_resize = args.do_resize
        self.img_width = args.width
        self.img_height = args.height
        self.cap = None
        self.thread = None
        #self.img_queue = Queue(IMG_QUEUE_SIZE)
        self.read_lock = threading.Lock()
        self.thread_running_lock = threading.Lock()
        # self.cap_lock = threading.RLock()
        self.frame_cnt = -1 # number of frames in case of video file
        self.frame_rate = -1 # frame rate of mp4 file if this cam is connected to the mp4 file - 2021.11.04, ahnjw

        self.retry_count = 0
        self.max_wait_sec = 60  # Don't wait longer than 1 minute between retries

        self.last_read_enter_ts = 0.0
        self.last_read_return_ts = 0.0
        self.last_frame_ts = 0.0
        self.read_in_progress = False
        self.read_hang_reported = False
        self.read_hang_count = 0
        self.grab_thread_ident = None
        self.grab_thread_name = None

        self._open()

    def _open(self):

        key = self.args.spacenorm_device_key
        access_token = self.cctv_info[key]['access_token']
        refresh_token = self.cctv_info[key]['refresh_token']

        if self.spacenorm_api is None:
            self.spacenorm_api = Gateway(access_token, refresh_token)
        
        # Clean up any "zombie" cap before opening
        if self.cap is not None:
            self.release()

        self.cap = open_rtsp_universal(key, self.cctv_info[key]['uri'])
        
        if self.cap and self.cap.isOpened():
            self._start()
        else:
            self.is_opened = False
        
    def isOpened(self):
        return self.is_opened

    def _start(self):
        key = self.args.spacenorm_device_key
        
        # 1. Verification: Don't start if the capture object failed to initialize
        if self.cap is None or not self.cap.isOpened():
            logger.warning(f"[{key}] _start() called but self.cap is not valid.")
            self.is_opened = False
            return

        # 2. Synchronous Probe: Try to grab the 1st image to confirm the stream is alive
        # This happens in the calling thread (main_thread) before the sub-thread starts.
        try:
            success, img_handle = self.cap.read()
        except Exception as e:
            logger.warning(f"[{key}] Exception during first read in _start: {e}")
            success = False

        if not success or img_handle is None:
            logger.warning(f"[{key}] Failed to grab initial frame. Stream might be empty.")
            self.is_opened = False
            return

        # 3. Metadata Setup
        self.is_opened = True
        self.img_height, self.img_width, _ = img_handle.shape
        
        # Update the handle safely using the lock
        with self.read_lock:
            self.img_handle = img_handle

        # 4. Thread Launch: Start the background 'grab_img' thread
        with self.thread_running_lock:
            if not self.thread_running:
                self.thread_running = True
                # Create a new thread object
                self.thread = threading.Thread(
                    name=f'grab_img({key})', 
                    target=grab_img, 
                    args=(self,)
                )
                # Set as daemon so it dies if the main process is force-killed
                self.thread.daemon = True 
                self.thread.start()
                logger.info(f"[{key}] [grab_img() thread] successfully started.")
            else:
                logger.warning(f"[{key}] _start() called but thread_running is already True!")

    def _start_old(self):
        key = self.args.spacenorm_device_key

        while self.cap is None:
            self.cap = open_rtsp_universal(a.spacenorm_device_key, self.cctv_info[a.spacenorm_device_key]['uri'])
            if not self.cap:
                logger.info(f"[{key}] open_rtsp_universal() failed.. --> retry after 1 second..")
                time.sleep(1)

        # Try to grab the 1st image and determine width and height
        self.img_file_name, self.img_handle = self.cap.read()
        if type(self.img_file_name) is not str:
            self.img_file_name = None

        if self.img_handle is None:
            logger.warning(f"[{key}] Camera: cap.read() returns no image!")
            self.is_opened = False
            return

        self.is_opened = True
        
        self.img_height, self.img_width, _ = self.img_handle.shape
        assert not self.thread_running
        self.thread_running = True
        self.thread = threading.Thread(name=f'grab_img({self.spacenorm_device_key})', target=grab_img, args=(self,))
        self.thread.start()

    def read(self):
        """Returns a deep copy of the most recent frame in a thread-safe manner."""
        key = self.args.spacenorm_device_key
        
        # 1. Quick check for open status
        if not self.is_opened:
            # Use a subtle log level or a counter to avoid spamming the console 
            # if the stream is in the middle of a reconnection.
            return None

        # 2. Critical Section: Protect the shared img_handle resource
        with self.read_lock:
            if self.img_handle is None: # <-- This should not happen anymore with modified grab_img() that does not set img_handle to None on read failure, but we keep this check just in case
                logger.error(f"[{key}] ERROR: It's weird for cam.read() to return None")

                return None
            try:
                # We MUST return a copy. If we return the original, the grab_img 
                # thread might update the pixels while YOLOv7 is mid-inference.
                return self.img_handle.copy()
            except Exception as e:
                logger.error(f"[{key}] Error copying img_handle: {e}")
                return None

    def release(self):
        key = self.spacenorm_device_key
        logger.info(f"[{key}] release() initiated.")

        with self.thread_running_lock:
            logger.info(f"[{key}] Signaling grab_img thread to stop...")
            self.thread_running = False

        if self.thread and self.thread.is_alive():
            if threading.current_thread() is not self.thread:
                self.thread.join(timeout=10.0)
                if self.thread.is_alive():
                    logger.warning(f"[{key}] grab_img thread hung; proceeding without forced release.")
            else:
                logger.warning(f"[{key}] release() called from same thread; skip join.")

        logger.info(f"[{key}] after joining grab_img thread.")

        self.is_opened = False
        with self.read_lock:
            self.img_handle = None

        logger.info(f"[{key}] release() completed.")

    def release_old3(self):
        key = self.spacenorm_device_key
        logger.info(f"[{key}] release() initiated.")

        with self.thread_running_lock:
            logger.info(f"[{key}] Signaling grab_img thread to stop...")
            self.thread_running = False

        if self.thread and self.thread.is_alive():
            if threading.current_thread() is not self.thread:
                self.thread.join(timeout=10.0)
                if self.thread.is_alive():
                    logger.warning(f"[{key}] grab_img thread hung; proceeding without forced release.")
            else:
                logger.warning(f"[{key}] release() called from same thread; skip join.")

        logger.info(f"[{key}] after joining grab_img thread.")

        acquired = self.cap_lock.acquire(timeout=3.0)
        if not acquired:
            logger.warning(f"[{key}] cap_lock acquire timeout; skip cap.release()")
        else:
            try:
                if self.cap is not None:
                    try:
                        logger.info(f"[{key}] before cap.release()")
                        self.cap.release()
                        logger.info(f"[{key}] after cap.release()")
                    except Exception as e:
                        logger.error(f"[{key}] Error during cap.release(): {e}")
                    finally:
                        self.cap = None
            finally:
                self.cap_lock.release()

        logger.info(f"[{key}] release() step1 completed.")

        self.is_opened = False
        with self.read_lock:
            self.img_handle = None

        logger.info(f"[{key}] release() step2 completed.")

    def release_old2(self):
        key = self.spacenorm_device_key
        logger.info(f"[{key}] release() initiated.")

        # 1. Signal thread to stop
        logger.info(f"[{key}] Signaling grab_img thread to stop...")
        with self.thread_running_lock:
            self.thread_running = False
        
        # 2. Wait for thread to exit normally (non-blocking for the lock)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10.0)
            if self.thread.is_alive():
                logger.warning(f"[{key}] grab_img thread hung; proceeding with forced release.")

        # 3. Safe Lock Acquisition
        # We use a timeout so the Main Thread NEVER hangs forever.
        acquired = self.cap_lock.acquire(timeout=10.0)
        try:
            if self.cap is not None:
                # Even if we didn't get the lock, we attempt release or nullify
                try:
                    self.cap.release()
                    logger.info(f"[{key}] VideoCapture released.")
                except Exception as e:
                    logger.error(f"[{key}] Error during cap.release(): {e}")
                finally:
                    self.cap = None
        finally:
            if acquired:
                self.cap_lock.release()

        logger.info(f"[{key}] release() step1 completed.")

        self.is_opened = False
        with self.read_lock:
            self.img_handle = None

        logger.info(f"[{key}] release() step2 completed.")

    def release_old(self):
        key = self.args.spacenorm_device_key
        
        # 1. Signal the grab_img thread to stop looping
        if self.thread_running:
            logger.info(f"[{key}] Signaling grab_img thread to stop...")
            self.thread_running = False
        
        # 2. Join with a TIMEOUT (Crucial to prevent main_thread hang)
        if self.thread and self.thread.is_alive():
            # We wait 2 seconds. If the thread is stuck in a C++ read(), 
            # it won't join, so we stop waiting to save the main thread.
            self.thread.join(timeout=2.0)
            
            if self.thread.is_alive():
                logger.warning(f"[{key}] grab_img thread is HANGING in read(). Proceeding to force release.")

        # 3. Forcefully release the OpenCV/GStreamer object
        # Wrapping this in a try-block ensures self.cap = None always happens.
        with self.cap_lock:
            if self.cap:
                try:
                    logger.debug(f"[{key}] Try to self.cap.release()")
                    # Some drivers unblock the 'read()' call once 'release()' is called
                    self.cap.release()
                    logger.info(f"[{key}] --> Succeeded in releasing..")
                except Exception as e:
                    logger.warning(f"[{key}] --> Failed during release call: {e}")
                finally:
                    # ALWAYS nullify the object so _open() doesn't trigger a RuntimeError
                    self.cap = None
                    self.is_opened = False
            else:
                logger.debug(f"[{key}] self.cap is already None.")

        # 4. Final state cleanup
        self.is_opened = False
        with self.read_lock:
            self.img_handle = None

    def __del__(self):
        key = self.args.spacenorm_device_key

        self.release()
        self.spacenorm_api.release()

        logger.debug(f"[{key}] return from Camera::__del__()..")


class Camera_old():
    """Camera class which supports reading images from theses video sources:

    1. Image (jpg, png, etc.) file, repeating indefinitely
    2. Video file
    3. RTSP (IP CAM)
    4. USB webcam
    5. Jetson onboard camera
    6. Image file list -- ahnjw,2020.11.01
    """

    def __init__(self, args, cctv_info):
        # print("\n\nargs = {}".format(args))
        self.args = args
        self.cctv_info = cctv_info
        self.is_opened = False
        self.spacenorm_device_key = args.spacenorm_device_key
        self.spacenorm_api = None
        self.video_file = ''
        self.video_output_file = ''
        self.video_looping = args.video_looping
        self.thread_running = False
        self.img_handle = None
        self.img_file_name = None
        self.copy_frame = args.copy_frame
        self.do_resize = args.do_resize
        self.img_width = args.width
        self.img_height = args.height
        self.cap = None
        self.thread = None
        #self.img_queue = Queue(IMG_QUEUE_SIZE)
        self.read_lock = threading.Lock()
        self.frame_cnt = -1 # number of frames in case of video file
        self.frame_rate = -1 # frame rate of mp4 file if this cam is connected to the mp4 file - 2021.11.04, ahnjw

        #print("try to open the camera")
        self._open()  # try to open the camera

    def _open(self):
        """Open camera based on command line arguments."""
        a = self.args

        # print(f"[{a.spacenorm_device_key}] Open camera based on command line arguments.")
        # print(f"[{a.spacenorm_device_key}] self.cap = {self.cap}")
        if self.cap is not None:
            print(f"[{a.spacenorm_device_key}] camera is already opened!")
            raise RuntimeError('camera is already opened!')

        # print("a.spacenorm_device_key = ", a.spacenorm_device_key)
        if a.spacenorm_device_key:
            key = a.spacenorm_device_key
            logger.info(f"[{key}] Camera: using a cctv for spacenorm")
            # print(f"[{key}] Camera: using a cctv for spacenorm")
            
            access_token = self.cctv_info[a.spacenorm_device_key]['access_token']
            refresh_token = self.cctv_info[a.spacenorm_device_key]['refresh_token']
            #access_token = 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE'
            #refresh_token = '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'

            if self.spacenorm_api is None:
                # print("Create API interface for 'gateways'..")
                self.spacenorm_api = Gateway(access_token, refresh_token)
                # print("Successful creation of API interface for 'gateways'..")
            
            # self.cap = open_cam_rtsp(self.cctv_info[a.spacenorm_device_key]['uri'], a.width, a.height, a.rtsp_latency, a.spacenorm_device_key)
            self.cap = open_rtsp_universal(a.spacenorm_device_key, self.cctv_info[a.spacenorm_device_key]['uri'])
            if not self.cap:
                logger.info(f"[{key}] Camera._open() failed..")
                self.is_opened = False
            else:
                logger.info(f"[{key}] Camera._open() succeeded..")
                self._start()

        elif a.image:
            logger.info('  Camera: using an image file %s' % a.image)
            self.cap = 'image'
            self.img_handle = cv2.imread(a.image)
            if self.img_handle is not None:
                if self.do_resize:
                    self.img_handle = cv2.resize(
                        self.img_handle, (a.width, a.height))
                self.is_opened = True
                self.img_height, self.img_width, _ = self.img_handle.shape
        elif a.image_list:
            logger.info('  Camera: using an image list file %s' % a.image_list)
            self.cap = open_image_list(a.image_list) 
            self._start()

        elif a.video:
            print(f"a.video = {a.video}")
            logger.info('  Camera: using a video file %s' % a.video)
            self.video_file = a.video
            self.cap = cv2.VideoCapture(a.video)

            if self.cap.isOpened(): 
                self.is_opened = True
                self.frame_cnt = int(cv2.VideoCapture.get(self.cap, int(cv2.CAP_PROP_FRAME_COUNT)))
                self.frame_rate = (cv2.VideoCapture.get(self.cap, int(cv2.CAP_PROP_FPS)))
                print(f"\n=============== ({a.video}) frame_cnt = {self.frame_cnt}, frame_rate = {self.frame_rate}")
            else:
                self.is_opened = False

            if a.video_output:
                self.video_output_file = a.video.split('.')[0] + "_out.mp4"
                print("video_output_file = ", self.video_output_file)

            self._start()
        elif a.rtsp:
            logger.info('  Camera: using RTSP stream %s' % a.rtsp)
            # self.cap = open_cam_rtsp(a.rtsp, a.width, a.height, a.rtsp_latency)
            self.cap = open_rtsp_universal(a.spacenorm_device_key, a.rtsp)
            if not self.cap:
                logger.info(f"[{key}] Camera._open() failed..")
                self.is_opened = False
            else:
                logger.info(f"[{key}] successful Camera._open()..")
                self._start()
        elif a.usb is not None:
            logger.info('  Camera: using USB webcam /dev/video%d' % a.usb)
            self.cap = open_cam_usb(a.usb, a.width, a.height)
            self._start()
        elif a.onboard is not None:
            logger.info('  Camera: using Jetson onboard camera')
            self.cap = open_cam_onboard(a.width, a.height)
            self._start()
        else:
            raise RuntimeError('no camera type specified!')

    def isOpened(self):
        return self.is_opened

    def _start(self):
        key = self.args.spacenorm_device_key
        if not self.cap.isOpened():
            logger.warning(f"[{key}] Camera: starting while cap is not opened!")            
            return

        # Try to grab the 1st image and determine width and height
        self.img_file_name, self.img_handle = self.cap.read()
        if type(self.img_file_name) is not str:
            self.img_file_name = None
        if self.img_handle is None:
            logger.warning(f"[{key}] Camera: cap.read() returns no image!")
            self.is_opened = False
            return

        self.is_opened = True
        #if self.video_file:
        if self.video_file or self.args.image_list:
            if not self.do_resize:
                self.img_height, self.img_width, _ = self.img_handle.shape
            #print(f"\nimg_width = {self.img_width}, img_height = {self.img_height}\n")
        else:
            self.img_height, self.img_width, _ = self.img_handle.shape
            # start the child thread if not using a video file source
            # i.e. rtsp, usb or onboard
            assert not self.thread_running
            self.thread_running = True
            self.thread = threading.Thread(name=f'grab_img({self.spacenorm_device_key})', target=grab_img, args=(self,))
            # print("Just before starting grab_img() thread..")
            self.thread.start()

    def _stop(self):
        #print("in _stop(): self.thread_running = ", self.thread_running)
        if self.thread_running:
            self.thread_running = False
            self.thread.join()
        #print("--> self.thread_running = ", self.thread_running)
        #time.sleep(1)

    def read(self, key=None):
        """Read a frame from the camera object.

        Returns None if the camera runs out of image or error.
        """
        if not self.is_opened:
            print(f"[{key}] ERROR: read returns None since it is not opened")
            logger.error(f"[{key}] ERROR: read returns None since it is not opened")
            return None

        if self.video_file: # does not use multithreading - ahnjw,2020.11.05
            _, img = self.cap.read()
            if img is None:
                logger.info('  Camera: reaching end of video file')
                if self.video_looping:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.video_file)
                _, img = self.cap.read()
            if img is not None and self.do_resize:
                img = cv2.resize(img, (self.img_width, self.img_height))
            return img
        elif self.cap == 'image': # does not use multithreading - ahnjw,2020.11.05
            return np.copy(self.img_handle)
        elif self.args.image_list: # does not use multithreading - ahnjw,2020.11.05
            self.img_file_name, img = self.cap.read()
            return img
        else: # use multithreading(img_handle is continually updated by other thread(grab_img)) - ahnjw,2020.11.05
            if self.copy_frame:
                return self.img_handle.copy()
            else:
                self.read_lock.acquire()
                if self.img_handle is None: # should not copy when it is null - ahnjw,2021.04.03
                    self.read_lock.release()
                    return None
                else:
                    img_handle = self.img_handle.copy()
                    self.read_lock.release()                
                    return img_handle

    def release(self):
        
        self._stop() # stop the grab_img thread first
        
        key = self.args.spacenorm_device_key
        # logger.error(f"[{key}] ERROR: read returns None since it is not opened")
        
        logger.info(f"[{key}] self.release() is called")

        if self.cap:
            try:
                logger.info(f"[{key}] Try to self.cap.release()")
                self.cap.release()
                self.cap = None
                logger.info(f"[{key}] --> Succeded in releasing..")
            except:
                logger.info(f"[{key}] --> Failed in releasing..")
                pass
            self.is_opened = False
        else:
            logger.info(f"[{key}] self.cap is None --> do nothing in release()")

        print(f"[{key}] return from Camera::release()..")
        logger.info(f"[{key}] return from Camera::release()..")
        

    def __del__(self):
        key = self.args.spacenorm_device_key

        logger.info(f"[{key}] In Camera::__del__() is called..")
        self.release()

        if self.spacenorm_api:
            logger.info(f"[{key}] release self.spacenorm_api")
            #del self.spacenorm_api
            self.spacenorm_api.release()
            
        print(f"[{key}] return from Camera::__del__()..")
        logger.info(f"[{key}] return from Camera::__del__()..")

# ahnjw, 2020.11.02 -- class for processing multiple images
class ImageFileListReader():
    # ImageFileListReader class which supports reading images from the list of image filenames:

    def __init__(self, filepath_list):
        self.filepath_list = filepath_list
        # self.filepath_base = "/home/cym/Work/darknet"
        # print(self.filepath_list)
        print("number of files = {}".format(len(self.filepath_list)))

    def isOpened(self):
        if not self.filepath_list:
            return False
        else:
            return True

    def read(self):
        if len(self.filepath_list) == 0:
            print("No more image file...")
            return None, None
        else:
            # filepath = os.path.join(self.filepath_base, self.filepath_list.pop())
            filepath = self.filepath_list.pop()
            print("read(): {} ({})".format(filepath, len(self.filepath_list)))
            return filepath, cv2.imread(filepath)