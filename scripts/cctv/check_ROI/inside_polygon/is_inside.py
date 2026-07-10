import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from math import atan2
import random

def make_counterclockwise(vertices):
    n = len(vertices)
    # Calculate the centroid of the polygon
    centroid_x = sum(x for x, y in vertices) / n
    centroid_y = sum(y for x, y in vertices) / n

    # Sort the vertices based on their polar angle with respect to the centroid
    vertices.sort(key=lambda vertex: (atan2(vertex[1] - centroid_y, vertex[0] - centroid_x), vertex))

    return vertices

def is_point_inside_polygon(x, y, polygon):
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

# Example usage:
# polygon = [(1, 1), (2, 4), (5, 2), (4, 0), (7,2)]
polygon = []
# for i in range(0,10):
#     polygon.append((random.randint(0, 9), random.randint(0, 9)))
polygon = [(0.02,0.49),(0.2,0.45),(0.08,0.98),(0.27,0.96)]
print(f"{polygon}")
polygon = make_counterclockwise(polygon)
print(f"{polygon}\n")

# point = (3, 3)
# point = (random.randint(0, 9), random.randint(0, 9))
point = (0.1,0.1)
result = is_point_inside_polygon(point[0], point[1], polygon)

if result:
    print(f"The point {point} is inside the polygon.")
else:
    print(f"The point {point} is outside the polygon.")

# Plot the polygon and the test point
fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')

polygon_patch = Polygon(polygon, closed=True, fill=False, edgecolor='blue')
ax.add_patch(polygon_patch)

plt.plot(*point, 'ro', label='Test Point')

if result:
    plt.title(f"The point {point} is inside the polygon.")
else:
    plt.title(f"The point {point} is outside the polygon.")

plt.xlim(min(p[0] for p in polygon) - 1, max(p[0] for p in polygon) + 1)
plt.ylim(min(p[1] for p in polygon) - 1, max(p[1] for p in polygon) + 1)
plt.grid(True)
plt.legend()
plt.show()
