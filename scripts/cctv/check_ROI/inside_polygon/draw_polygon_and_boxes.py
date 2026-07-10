import cv2
import numpy as np

# Assuming you have your img, polygon, and boxes defined
img = cv2.imread('test.jpg')
# print(f"image size = {img.shape}")
img_w = img.shape[1]
img_h = img.shape[0]
print(f"img_w = {img_w}, img_h = {img_h}")

normalized_polygon = [[0.1,0.2], [0.2, 0.5], [0.4, 0.4], [0.3, 0.1]]
normalized_boxes = [[0.3,0.4,0.5,0.6], [0.5,0.7,0.6,0.9]]

polygon = [[int(x*img_w), int(y*img_h)] for x,y in normalized_polygon ]
print(f"normalized_polygon = {normalized_polygon}")
print(f"polygon = {polygon}")
boxes = [[int(x1*img_w), int(y1*img_h), int(x2*img_w), int(y2*img_h)] for x1,y1,x2,y2 in normalized_boxes]
print(f"normalized_boxes = {normalized_boxes}")
print(f"boxes = {boxes}")

# Create a copy of the image to avoid modifying the original
output_img = img.copy()

# Draw the polygon in red
# Convert the list of tuples to a NumPy array
polygon_pts = np.array(polygon, np.int32)
# Reshape the array to be a 1xN array of points
polygon_pts = polygon_pts.reshape((-1, 1, 2))
# Draw the polygon
cv2.polylines(output_img, [polygon_pts], isClosed=True, color=(0, 0, 255), thickness=2)

# Draw the boxes in green
for box in boxes:
    x1, y1, x2, y2 = box
    print(f"x1y1x2y2 = {x1}, {y1}, {x2}, {y2}")
    cv2.rectangle(output_img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

# Show the resulting image
cv2.imshow('Image with Polygon and Boxes', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
