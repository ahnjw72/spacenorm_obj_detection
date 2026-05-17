##################################################################
# Utility functions for post-processing YOLOv7 detection results
##################################################################
import datetime
import logging
import cv2
import numpy as np
from math import atan2

#logger = logging.getLogger('spacenorm_person_detect')
logger = logging.getLogger(__name__)

def make_counterclockwise(vertices):
    n = len(vertices)
    # Calculate the centroid of the polygon
    centroid_x = sum(x for x, y in vertices) / n
    centroid_y = sum(y for x, y in vertices) / n

    # Sort the vertices based on their polar angle with respect to the centroid
    vertices.sort(key=lambda vertex: (atan2(vertex[1] - centroid_y, vertex[0] - centroid_x), vertex))

    return vertices

# def is_point_inside_polygon(x, y, polygon):
def is_point_inside_polygon(xy, polygon):
    x = xy[0]
    y = xy[1]
    num_vertices = len(polygon)
    if num_vertices < 3:
        return False

    odd_nodes = False
    j = num_vertices - 1

    for i in range(num_vertices):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if yi < y and yj >= y or yj < y and yi >= y:
            if xi + (y - yi) / (yj - yi) * (xj - xi) < x:
                odd_nodes = not odd_nodes

        j = i

    return odd_nodes

# roi : {'img_w': 1920, 'img_h': 1080, 'vertices': [[[38, 528], [391, 491], [159, 1062], [521, 1041]], [[77, 500], [111, 40], [160, 1000], [900, 1050]]]}
# boxes (list of xyxy):  [[566, 37, 600, 62], [710, 48, 787, 155]] 
temp_img_count = 0
def remove_outside_ROI(boxes, confs, clss, imgsz, roi, img, key, save_result=False):
    
    global temp_img_count
    max_img_count = 1000

    if save_result and (temp_img_count < max_img_count):
        img_filepath = f"./temp/{key}_{datetime.datetime.now()}.jpg"
        output_img = img.copy()
        # img_w_output_img = img.shape[1]
        # img_h_output_img = img.shape[0]

        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2) # by default, each box is colored BLUE

    img_w_output_img = img.shape[1]
    img_h_output_img = img.shape[0]
    normalized_boxes = [[x1/img_w_output_img, y1/img_h_output_img, x2/img_w_output_img, y2/img_h_output_img] for x1, y1, x2, y2 in boxes]
    center_boxes = [[(x1+x2)/2, (y1+y2)/2] for x1, y1, x2, y2 in normalized_boxes]
    
    vertices_roi = roi['vertices']
    img_w_roi = roi['img_w']
    img_h_roi = roi['img_h']
    if 'vertices_sorted' in roi:
        vertices_sorted = roi['vertices_sorted']
    else:
        vertices_sorted = 0

    # assert(vertices_roi == 1) # due to uncertain behavior of make_counterclockwise(), we only support coordinates with sorted order (counterclockwise) now.

    new_boxes = []
    new_confs = []
    new_clss = []
    for polygon in vertices_roi: # a polygon (= a list of xy coordinates) : [[38, 528], [391, 491], [159, 1062], [521, 1041]]
        
        # if vertices_sorted is not zero, then the vertices are sorted in the counterclockwise order already and
        # so make_counterclockwise() is not applied.
        if vertices_sorted == 0:
            normalized_polygon = make_counterclockwise([[x/img_w_roi, y/img_h_roi] for x, y in polygon])
        else:
            normalized_polygon = [[x/img_w_roi, y/img_h_roi] for x, y in polygon]

        # normalized_polygons.append(normalized_polygon)

        if save_result and (temp_img_count < max_img_count):
            polygon_in_img = [[int(x*img_w_output_img), int(y*img_h_output_img)] for x,y in normalized_polygon]
            polygon_pts = np.array(polygon_in_img, np.int32)
            polygon_pts = polygon_pts.reshape((-1, 1, 2))
            cv2.polylines(output_img, [polygon_pts], isClosed=True, color=(0, 0, 255), thickness=2)

        for i, center_box in enumerate(center_boxes):
            # if is_point_inside_polygon(center_box[0], center_box[1], normalized_polygon):
            if is_point_inside_polygon((center_box[0], center_box[1]), normalized_polygon):
                new_boxes.append(boxes[i])
                new_confs.append(confs[i])
                new_clss.append(clss[i])

                if save_result and (temp_img_count < max_img_count):
                    x1, y1, x2, y2 = boxes[i]
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2) # the box whose center is inside the polygon becomes GREEN

    if (len(new_boxes) > 0) and save_result and (temp_img_count < max_img_count): # first 50 images are saved for further inspection
        cv2.imwrite(img_filepath, output_img)
        logger.info(f"{img_filepath} is written")
        temp_img_count += 1

    return new_boxes, new_confs, new_clss

# In case of car detection, only the classes of 2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck' are remained.
# FIXME: This routine is unnecessary if the model only detects the objects in the above four categories. 
#        On how to 'cut-out' the output of the model for the computation efficiency, refer to the following chat:
#        https://chat.openai.com/share/853c6203-8be8-4c44-9c72-0e1f8259f308
#        (CAUTION: The 'cut-out' method introduced in the above chat is not verified.)        
def select_car_related_results(boxes, confs, clss): # ex. of clss : [tensor(56., device='cuda:0'), tensor(53., device='cuda:0')]
    new_boxes = []
    new_confs = []
    new_clss = []
    for i, tensor in enumerate(clss):
        if tensor.item() in [2., 3., 5., 7.]:
            new_boxes.append(boxes[i])
            new_confs.append(confs[i])
            new_clss.append(clss[i])

    return new_boxes, new_confs, new_clss

def filter_only_person(vis, boxes, confs, clss, img, key, save_result=False):
    new_boxes = []
    new_confs = []
    new_clss = []
    temp_types = []
    temp_motionesses = []
    other_than_person = False
    # logger.info(f"[{key}] filter out non-person objec(s)")
    for i, tensor in enumerate(clss):
        if tensor.item() == 0: # person class
            new_boxes.append(boxes[i])
            new_confs.append(confs[i])
            new_clss.append(clss[i])
        else:
            other_than_person = True
            logger.info(f"[{key}] non-person object detected with class = {clss[i]}")

        if save_result:
            temp_types.append(1)
            temp_motionesses.append(-1)
            
    if other_than_person and save_result:
        img_filepath = f"./logs/temp/{key}_{datetime.datetime.now()}_non_person.jpg"
        img_for_investigation = vis.draw_bboxes(img, boxes, confs, clss, temp_types, temp_motionesses)
        cv2.imwrite(img_filepath, img_for_investigation)
        logger.info(f"[{key}] non-person object(s) detected and saved the image : {img_filepath}")
        
    return new_boxes, new_confs, new_clss

# min_ratio_in_percent : minimum area ratio(in %) of bounding box to image area
def filter_small_objects(key, boxes, confs, clss, min_ratio_in_percent, img_shape):
    new_boxes = []
    new_confs = []
    new_clss = []
    img_h, img_w = img_shape
    img_area = img_w * img_h

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        area_ratio_in_precent = (box_w*box_h / img_area) * 100.0
        if area_ratio_in_precent < min_ratio_in_percent:
            logger.info(f"[{key}] box area ratio {area_ratio_in_precent:.2f}% is smaller than min_ratio {min_ratio_in_percent:.2f}% -> remove this box")
            continue
        else:
            new_boxes.append(boxes[i])
            new_confs.append(confs[i])
            new_clss.append(clss[i])
               
    return new_boxes, new_confs, new_clss

def check_bb_on_background(img, boxes, confs, clss, remove_member, args, kernel, fgbg):
    fgmask = fgbg.apply(img)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('fgmask', fgmask)

    CONF_THRESH1_BACKGROUND = args.conf_thresh1_background 
    CONF_THRESH2_BACKGROUND = args.conf_thresh2_background 
    # check the order of threshold values
    thresholds = [CONF_THRESH1_BACKGROUND, CONF_THRESH2_BACKGROUND]
    assert(sorted(thresholds) == thresholds)

    BACKGROUND_THRESH1 = args.background_thresh1
    BACKGROUND_THRESH2 = args.background_thresh2 
    BACKGROUND_THRESH3 = args.background_thresh3 
    # check the order of threshold values
    thresholds = [BACKGROUND_THRESH3, BACKGROUND_THRESH2, BACKGROUND_THRESH1]
    assert(sorted(thresholds) == thresholds)

    key = args.spacenorm_device_key

    #logger.info("START: check_bb_on_background()------------------------------")

    new_boxes = []
    new_confs = []
    new_clss = []
    types = [] # 1 or 2 or 3 means background, -1 or -2 means static person, otherwise (0) normal person
    motionesses = []
    for bb, cf, cl in zip(boxes, confs, clss):
        # x_min, y_min, x_max, y_max = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
        # print(f"x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max}")

        # check if this box is on background (ahnjw,2020.10.12)
        ROI = fgmask[y_min:y_max, x_min:x_max]
        sum_ROI = cv2.sumElems(ROI)
        motioness = sum_ROI[0]/ROI.size

        #print("sum_ROI[0]/ROI.size = {}".format(sum_ROI[0]/ROI.size))
        #print("background = {}, cf = {}".format(sum_ROI[0]/ROI.size, cf))
        # logger.info("[{}] motioness = {}, cf = {}".format(key,sum_ROI[0]/ROI.size, cf))
        logger.info(f"[{key}] motioness = {motioness:.2f}, cf = {cf:.2f}")
        if (motioness < BACKGROUND_THRESH3): # this ROI is background
            logger.info(f"[{key}] Type 3: background")
            if remove_member == False:
                # print("background type 3(conf = {})--{}".format(cf, bb))
                logger.info("[{}] background type 2(conf = {})--{}".format(key, cf, bb))
                new_boxes.append(bb)
                new_confs.append(cf)
                new_clss.append(cl)
                types.append(3) # mark a bb as background type 3
                motionesses.append(motioness)
            else:
                #print("remove a background type 2 bb(conf = {})--{}".format(cf, bb))
                logger.info("[{}] remove a background type 3 bb(conf = {})--{}".format(key,cf, bb))

        elif (motioness > BACKGROUND_THRESH1): # this ROI is not background
            logger.info(f"[{key}] Type 0: object (conf = {cf})")
            new_boxes.append(bb)
            new_confs.append(cf)
            new_clss.append(cl)
            types.append(0)
            motionesses.append(motioness)

        elif (motioness < BACKGROUND_THRESH2):
            #print("background = {}, cf = {}".format(sum_ROI[0]/ROI.size, cf))
            #print("Type 2 or -2")
            #logger.info("Type 2 or -2")
            if cf < CONF_THRESH2_BACKGROUND:
                logger.info(f"[{key}] Type 2: background")
                if remove_member == False:
                    # print("background type 2(conf = {})--{}".format(cf, bb))
                    logger.info("[{}] background type 2(conf = {})--{}".format(key, cf, bb))
                    new_boxes.append(bb)
                    new_confs.append(cf)
                    new_clss.append(cl)
                    types.append(2) # mark a bb as background type 2
                    motionesses.append(motioness)
                else:
                    #print("remove a background type 2 bb(conf = {})--{}".format(cf, bb))
                    logger.info("[{}] remove a background type 2 bb(conf = {})--{}".format(key, cf, bb))
            else:
                #print("Type -2: do not remove a bb since its confidence = {} (> {})".format(cf, CONF_THRESH2_BACKGROUND))
                logger.info("[{}] Type -2: do not remove a bb since its confidence = {} (> {})".format(key, cf, CONF_THRESH2_BACKGROUND))
                new_boxes.append(bb)
                new_confs.append(cf)
                new_clss.append(cl)
                types.append(-2)
                motionesses.append(motioness)

        else:
            if cf < CONF_THRESH1_BACKGROUND:
                logger.info(f"[{key}] Type 1: background")
                if remove_member == False:
                    # print("background type 1(conf = {})--{}".format(cf, bb))
                    logger.info("[{}] background type 1(conf = {})--{}".format(key, cf, bb))
                    new_boxes.append(bb)
                    new_confs.append(cf) # mark a bb as pseudo background 
                    new_clss.append(cl)
                    types.append(1) # mark a bb as background type 1
                    motionesses.append(motioness)
                else:
                    #print("remove a background type 1 bb(conf = {})--{}".format(cf, bb))
                    logger.info("[{}] remove a background type 1 bb(conf = {})--{}".format(key, cf, bb))
            else:
                #print("Type -1: do not remove a bb since its confidence = {} (> {})".format(cf, CONF_THRESH1_BACKGROUND))
                logger.info("[{}] Type -1: do not remove a bb since its confidence = {} (> {})".format(key, cf, CONF_THRESH1_BACKGROUND))
                new_boxes.append(bb)
                new_confs.append(cf)
                new_clss.append(cl)
                types.append(-1) # mark a bb as person even its motion value is low
                motionesses.append(motioness)

    #logger.info("END: check_bb_on_background()------------------------------\n")

    return new_boxes, new_confs, new_clss, types, motionesses
