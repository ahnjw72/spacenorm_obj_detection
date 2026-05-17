
"""display.py
"""

import numpy as np
import cv2

def open_window(window_name, title, width=None, height=None):
    """Open the display window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(window_name, title)
    if width and height:
        cv2.resizeWindow(window_name, width, height)


def show_help_text(img, help_text):
    """Draw help text on image."""
    cv2.putText(img, help_text, (11, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (32, 32, 32), 4, cv2.LINE_AA)
    cv2.putText(img, help_text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (240, 240, 240), 1, cv2.LINE_AA)
    return img


def show_fps(img, fps, key=None):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = 'FPS: {:.2f}'.format(fps)
    if key is not None:
        fps_text = fps_text + f" {key}"
    cv2.putText(img, fps_text, (11, 30), font, 1.5, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 30), font, 1.5, (240, 240, 240), 1, line)
    return img

def show_file_info(img, file_path, frame_cnt, cur_frame, toi_date, cur_date, detection_result):
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    progress_percent = 100*(cur_frame/frame_cnt)
    progress_text = "({:.1f} %)".format(progress_percent)
    #print("progress = ", progress_text)
    #cv2.putText(img, progress_text, (161, 30), font, 1.5, (32, 32, 32), 4, line)
    #cv2.putText(img, progress_text, (160, 30), font, 1.5, (240, 240, 240), 1, line)
    
    
    cv2.putText(img, file_path.split('/')[-1], (11, 40), font, 1.2, (32, 32, 32), 4, line)
    cv2.putText(img, file_path.split('/')[-1], (10, 40), font, 1.2, (240, 240, 240), 1, line)
    
    cv2.putText(img, 'TOI: '+str(toi_date), (11, 70), font, 1.2, (32, 32, 32), 4, line)
    cv2.putText(img, 'TOI: '+str(toi_date), (10, 70), font, 1.2, (240, 240, 240), 1, line)

    #cv2.putText(img, str(cur_date.strftime('%Y-%m-%d %H:%M:%S'))+f' ({round(cur_date.timestamp())})', (11, 100), font, 1.2, (32, 32, 32), 4, line)
    #cv2.putText(img, str(cur_date.strftime('%Y-%m-%d %H:%M:%S'))+f' ({round(cur_date.timestamp())})', (10, 100), font, 1.2, (240, 240, 240), 1, line)

    cv2.putText(img, str(cur_date.strftime('%Y-%m-%d %H:%M:%S')), (11, 100), font, 1.2, (32, 32, 32), 4, line)
    cv2.putText(img, str(cur_date.strftime('%Y-%m-%d %H:%M:%S')), (10, 100), font, 1.2, (240, 240, 240), 1, line)

    cv2.putText(img, f'({detection_result[0]} , {detection_result[1]})', (11, 130), font, 1.2, (32, 32, 32), 4, line)
    cv2.putText(img, f'({detection_result[0]} , {detection_result[1]})', (10, 130), font, 1.2, (240, 240, 240), 1, line)

    return img

def set_display(window_name, full_scrn):
    """Set disply window to either full screen or normal."""
    if full_scrn:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_NORMAL)

def combine_two_images(img1, img2, vertical=True):
    """
    ref site: https://stackoverflow.com/questions/7589012/combining-two-images-with-opencv
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if vertical:
        #create empty matrix
        vis = np.zeros((h1+h2, max(w1,w2),3), np.uint8)

        #combine 2 images
        vis[:h1, :w1,:3] = img1
        vis[h1:h1+h2, :w2,:3] = img2
    else:
        #create empty matrix
        vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

        #combine 2 images
        vis[:h1, :w1,:3] = img1
        vis[:h2, w1:w1+w2,:3] = img2

    return vis