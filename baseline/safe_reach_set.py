" safe reachable set"

import numpy as np
from shapely.geometry import LineString, Polygon, Point
import matplotlib.pyplot as plt

# # Function to create polygon from line intersection with the rectangle
# def line_cut_rect(line, rect):
#     clipped_line = line.intersection(rect)
#     if clipped_line.is_empty:
#         return None
#     if clipped_line.geom_type == 'MultiLineString':
#         polygons = [Polygon([p.bounds[:2], p.bounds[2:4], (p.bounds[0], p.bounds[3]), (p.bounds[2], p.bounds[1])]) for p in clipped_line]
#         return unary_union(polygons)
#     elif clipped_line.geom_type == 'LineString':
#         assert len(clipped_line.coords)==2
#         a,b,c = line_equation(clipped_line.coords[0],clipped_line.coords[1])
#         linering = list(clipped_line.coords)
#         for pol in list(rect.exterior.coords)[:-1]: # cause it is a ring
#             if a* pol[0]+ b*pol[1]+ c >0:
#                 linering += [pol]
#         print(linering)
#         ret = sort_points_CCW(linering)
#         print(ret)
#         return Polygon(ret)
#     return None

    
def line_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    # return a, b, c
    return lambda x, y: a * x + b * y + c


def perpendicular_bisector(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    # Midpoint
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    
    # Slope of the line segment
    if x1 == x2:
        # Line is vertical, perpendicular bisector is horizontal
        a = 0
        b = 1
        c = -my
    elif y1 == y2:
        # Line is horizontal, perpendicular bisector is vertical
        a = 1
        b = 0
        c = -mx
    else:
        # Slope of the line segment
        slope = (y2 - y1) / (x2 - x1)
        # Perpendicular slope is the negative reciprocal
        perp_slope = -1 / slope
        # Line equation: y - my = perp_slope * (x - mx)
        # Rearranging to ax + by + c = 0
        a = perp_slope
        b = -1
        c = my - perp_slope * mx
    
    return a, b, c

def line_intersect_with_rect(a, b, c, rectangle):
    intersections = []
    
    # Line equation ax + by + c = 0
    # If b != 0, y = -(a/b)x - (c/b)
    if b != 0:
        line = LineString([(-10, -(a * -10 + c) / b), (10, -(a * 10 + c) / b)])
    else:
        # Vertical line
        line = LineString([(-c / a, -10), (-c / a, 10)])
    
    # Find intersection points
    intersection = rectangle.boundary.intersection(line)
    
    if intersection.is_empty:
        return []
    elif isinstance(intersection, Point):
        return [(intersection.x, intersection.y)]
    elif isinstance(intersection, LineString):
        return list(intersection.coords)
    else:
        return [point.coords[0] for point in intersection.geoms]

def sort_points_CCW(points):
    """Counter clockwise"""
    centroid = np.mean(points, axis=0)
    # Function to calculate angle with respect to the centroid
    def angle_from_centroid(point):
        vector = point - centroid
        return np.arctan2(vector[1], vector[0])
    # Sort points by angle
    sorted_points = sorted(points, key=angle_from_centroid)
    return np.array(sorted_points)

def line_cut_rect(a,b,c, rect, ref_point):
    _ref = a* ref_point[0] + b*ref_point[1] + c
    vertices = []
    l = line_intersect_with_rect(a,b,c, rect)
    vertices += l
    for pol in list(rect.exterior.coords)[:-1]: # cause it is a ring
        if (a* pol[0]+ b*pol[1]+ c) * _ref > 0:
            vertices += [pol]
    # print(vertices)
    vertices = sort_points_CCW(vertices)
    # print(vertices)
    return Polygon(vertices)


if __name__ == "__main__":
    from shapely.geometry import box
    from shapely import intersection_all
    
    # Define rectangular boundary
    rect = box(-1, -1, 1, 1)  # Rectangle from (0,0) to (10,10)
    evader = [0,0]
    pursuers = [[0,1], [-0.5,-0.5], [0.5,-0.5]]
    clipped_polygons = []
    for pursuer in pursuers:
        p1 = pursuer
        p2 = evader
        a,b,c = perpendicular_bisector(p1,p2)
        pg = line_cut_rect(a,b,c, rect, evader)
        clipped_polygons.append(pg)
    # Find the intersection of all clipped polygons
    common_region = intersection_all(clipped_polygons)
    print(common_region)
    print(common_region.area)

    # >>>>>>>>>>>>>>>> Plotting the results >>>>>>>>>>>>>>>
    fig, ax = plt.subplots()
    x, y = rect.exterior.xy
    ax.plot(x, y, 'k-', label='Boundary Rectangle')

    # for line in lines:
    #     ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 'b--', label='Lines')

    if not common_region.is_empty:
        if common_region.geom_type == 'Polygon':
            x, y = common_region.exterior.xy
            ax.fill(x, y, 'r', alpha=0.5, label='Common Intersection')
        elif common_region.geom_type == 'MultiPolygon':
            for poly in common_region:
                x, y = poly.exterior.xy
                ax.fill(x, y, 'r', alpha=0.5, label='Common Intersection')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()

    plt.savefig('baseline/srs.png', bbox_inches='tight', dpi=300)
    # plt.show()
    # <<<<<<<<<<<<<<<< Plotting the results <<<<<<<<<<<<<<<<
