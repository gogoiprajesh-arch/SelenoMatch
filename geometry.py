"""
Stage 2: Intersection Calculator
---------------------------------
Reprojects each image's four corners into South Polar Stereographic meters,
intersects the two footprints, then projects the intersection polygon back
into pixel space for *each* raw image, independently, so the Geo-Sampler
knows exactly which window to read from each file.
"""
import numpy as np
import cv2
from shapely.geometry import Polygon

MOON_RADIUS_M = 1737400


def latlon_to_stereographic(lat, lon, R=MOON_RADIUS_M):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    rho = 2 * R * np.tan(np.pi / 4 + lat_rad / 2)
    x = rho * np.sin(lon_rad)
    y = rho * np.cos(lon_rad)
    return x, y


def stereographic_to_latlon(x, y, R=MOON_RADIUS_M):
    rho = np.sqrt(x ** 2 + y ** 2)
    lat_rad = 2 * np.arctan(rho / (2 * R)) - np.pi / 2
    lat = np.degrees(lat_rad)
    lon = np.degrees(np.arctan2(x, y))
    if lon < 0:
        lon += 360
    return lat, lon


def corners_to_polygon_stereographic(corners):
    points = []
    for key in ['ul', 'ur', 'lr', 'll']:
        lat, lon = corners[key]
        points.append(latlon_to_stereographic(lat, lon))
    return Polygon(points)


def _pixel_coords_from_latlon(points_latlon, corners, img_shape):
    """Perspective-map the 4 known lat/lon corners -> pixel corners, then
    apply that mapping to arbitrary lat/lon points (used for XML products
    that have no affine transform of their own)."""
    n_rows, n_cols = img_shape

    src_pts = np.array([
        latlon_to_stereographic(*corners['ul']),
        latlon_to_stereographic(*corners['ur']),
        latlon_to_stereographic(*corners['lr']),
        latlon_to_stereographic(*corners['ll']),
    ], dtype=np.float32)

    dst_pts = np.array([
        [0, 0],
        [n_cols - 1, 0],
        [n_cols - 1, n_rows - 1],
        [0, n_rows - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    pixel_coords = []
    for lat, lon in points_latlon:
        px_m, py_m = latlon_to_stereographic(lat, lon)
        pt = np.array([[[px_m, py_m]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, matrix)
        col, row = transformed[0][0]
        row = int(np.clip(round(row), 0, n_rows - 1))
        col = int(np.clip(round(col), 0, n_cols - 1))
        pixel_coords.append((row, col))
    return pixel_coords


def _bbox_from_transform(points_xy, transform, img_shape):
    """Georeferenced (.tif) path: use the real affine transform."""
    inv = ~transform
    rows, cols = [], []
    for x, y in points_xy:
        col, row = inv * (x, y)
        rows.append(row)
        cols.append(col)
    r_min, r_max = int(min(rows)), int(max(rows))
    c_min, c_max = int(min(cols)), int(max(cols))
    return r_min, r_max, c_min, c_max


def _bbox_from_latlon(points_latlon, corners, img_shape):
    pixel_coords = _pixel_coords_from_latlon(points_latlon, corners, img_shape)
    rows = [p[0] for p in pixel_coords]
    cols = [p[1] for p in pixel_coords]
    return min(rows), max(rows), min(cols), max(cols)


def compute_overlap_windows(meta1, meta2):
    """
    meta1/meta2: dicts with keys 'corners', 'resolution', 'product_type',
    'shape' (rows, cols), and — only for .tif products — 'transform'.

    Returns None if the footprints don't overlap, otherwise a dict:
        bbox1, bbox2      -> (row_min, row_max, col_min, col_max) per image,
                              clamped to that image's own bounds
        target_res        -> coarser of the two resolutions (m/px)
        universal_out_shape-> (rows, cols) both crops should be resampled to
    """
    poly1 = corners_to_polygon_stereographic(meta1['corners'])
    poly2 = corners_to_polygon_stereographic(meta2['corners'])
    intersection = poly1.intersection(poly2)

    if intersection.is_empty:
        return None

    # minx, miny, maxx, maxy = intersection.bounds
    # cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    # span_m = 2000.0  # 20km x 20km localized patch region
    
    # sub_box = Polygon([
    #     (cx - span_m, cy - span_m),
    #     (cx + span_m, cy - span_m),
    #     (cx + span_m, cy + span_m),
    #     (cx - span_m, cy + span_m)
    # ])
    # intersection = intersection.intersection(sub_box)

    if intersection.is_empty:
        return None

    intersection_xy = list(intersection.exterior.coords)

    intersection_xy = list(intersection.exterior.coords)
    intersection_latlon = [stereographic_to_latlon(x, y) for x, y in intersection_xy]

    def bbox_for(meta):
        if meta['product_type'] == 'tif':
            return _bbox_from_transform(intersection_xy, meta['transform'], meta['shape'])
        return _bbox_from_latlon(intersection_latlon, meta['corners'], meta['shape'])

    r_min1, r_max1, c_min1, c_max1 = bbox_for(meta1)
    r_min2, r_max2, c_min2, c_max2 = bbox_for(meta2)

    h1, w1 = meta1['shape']
    h2, w2 = meta2['shape']
    r_min1, r_max1 = max(0, r_min1), min(h1, r_max1)
    c_min1, c_max1 = max(0, c_min1), min(w1, c_max1)
    r_min2, r_max2 = max(0, r_min2), min(h2, r_max2)
    c_min2, c_max2 = max(0, c_min2), min(w2, c_max2)

    target_res = max(meta1['resolution'], meta2['resolution'])
    min_x, min_y, max_x, max_y = intersection.bounds
    physical_width_m = max_x - min_x
    physical_height_m = max_y - min_y

    target_w = max(1, int(physical_width_m / target_res))
    target_h = max(1, int(physical_height_m / target_res))


    return {
        'bbox1': (r_min1, r_max1, c_min1, c_max1),
        'bbox2': (r_min2, r_max2, c_min2, c_max2),
        'target_res': target_res,
        'universal_out_shape': (target_h, target_w),
        'bounds': (min_x, min_y, max_x, max_y)  # <-- ADD THIS SO THE SAMPLER CAN WARP XMLs
    }



