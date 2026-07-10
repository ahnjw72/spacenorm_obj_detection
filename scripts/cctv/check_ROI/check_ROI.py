import cv2
import json
import argparse
import numpy as np
from math import atan2

def make_counterclockwise(vertices):
    n = len(vertices)
    # Calculate the centroid of the polygon
    centroid_x = sum(x for x, y in vertices) / n
    centroid_y = sum(y for x, y in vertices) / n

    # Sort the vertices based on their polar angle with respect to the centroid
    vertices.sort(key=lambda vertex: (atan2(vertex[1] - centroid_y, vertex[0] - centroid_x), vertex))

    return vertices

def parse_args():
    """Parse input arguments."""
    desc = ("Draw polygon(s) as in ROI config over the CCTV's image")
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '-c', '--config', type=str, default="test.json",
        help=('JSON CCTV configuration input file'))
    parser.add_argument(
        '-i', '--img', type=str, default=None,
        help=('Image input file (not use RTSP)'))
    parser.add_argument(
        '--cctv', type=str, default=None,
        help=('CCTV name for which check ROI'))    
    
    args = parser.parse_args()
    return args

def draw_polygons(img, roi):
    
    output_img = img.copy()
    vertices_roi = roi['vertices']
    img_w_roi = roi['img_w']
    img_h_roi = roi['img_h']
    if 'vertices_sorted' in roi:
        vertices_sorted = roi['vertices_sorted']
    else:
        vertices_sorted = 0

    img_w = img.shape[1]
    img_h = img.shape[0]

    for polygon in vertices_roi: # a polygon (= list of xy coordinates) : [[38, 528], [391, 491], [159, 1062], [521, 1041]]
        if vertices_sorted == 0:
            normalized_polygon = make_counterclockwise([[x/img_w_roi, y/img_h_roi] for x, y in polygon])
        else:
            normalized_polygon = [[x/img_w_roi, y/img_h_roi] for x, y in polygon]

        polygon_in_img = [[int(x*img_w), int(y*img_h)] for x,y in normalized_polygon]
        polygon_pts = np.array(polygon_in_img, np.int32)
        polygon_pts = polygon_pts.reshape((-1, 1, 2))
        cv2.polylines(output_img, [polygon_pts], isClosed=True, color=(0, 0, 255), thickness=2)    

    return output_img

def main():

    args = parse_args()

    json_config_file = args.config
    with open(json_config_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    # print(data)

    for company in data['companies']:
        for cctv_name, cctv_config in company['CCTV'].items():
            if args.cctv is not None and args.cctv != cctv_name:
                continue

            if 'ROI' in cctv_config:
                uri = cctv_config['uri']
                roi = cctv_config['ROI']

                cap = cv2.VideoCapture(uri)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print(f"({cctv_name}) Can't get a frame")                        
                    else:
                        out_img = draw_polygons(frame, roi)        
                        cv2.imwrite(f"./ROI_images/{cctv_name}_ROI.jpg", out_img)
                        print(f"{cctv_name}_ROI.jpg is written..")
                        # continue
                cap.release()
            else:
                print(f"There is no ROI info for {cctv_name}")


if __name__=='__main__':
    main()