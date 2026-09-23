import rasterio
from rasterio.windows import Window
import numpy as np
import cv2
from geometry import latlon_to_stereographic

def sample_overlap(path, bbox, out_shape):
    """Standard windowed memory-safe read for map-projected TIF data."""
    r_min, r_max, c_min, c_max = bbox
    height = max(1, r_max - r_min)
    width = max(1, c_max - c_min)

    with rasterio.open(str(path)) as src:
        # Memory Saver: out_shape forces Rasterio to compress the array during load!
        data = src.read(
            1,
            window=Window(col_off=c_min, row_off=r_min, width=width, height=height),
            out_shape=out_shape,
        )
    return data.astype(np.float32)

def sample_unprojected_xml(path, corners, out_shape, bounds, target_res, is_iirs):
    """Warps raw pushbroom XML data (TMC or IIRS) onto the universal grid."""
    with rasterio.open(str(path)) as src:
        if is_iirs:
            band_range = list(range(40, 61))
            full_band = np.mean(src.read(band_range), axis=0).astype(np.float32)
        else:
            full_band = src.read(1).astype(np.float32)

        min_x, min_y, max_x, max_y = bounds
        ul_x, ul_y = latlon_to_stereographic(*corners['ul'])
        ur_x, ur_y = latlon_to_stereographic(*corners['ur'])
        lr_x, lr_y = latlon_to_stereographic(*corners['lr'])
        ll_x, ll_y = latlon_to_stereographic(*corners['ll'])

        def meter_to_grid(x, y):
            col = (x - min_x) / target_res
            row = (max_y - y) / target_res
            return [col, row]

        dst_pts = np.array([
            meter_to_grid(ul_x, ul_y), meter_to_grid(ur_x, ur_y),
            meter_to_grid(lr_x, lr_y), meter_to_grid(ll_x, ll_y)
        ], dtype=np.float32)

        src_pts = np.array([
            [0, 0], [src.width - 1, 0],
            [src.width - 1, src.height - 1], [0, src.height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(full_band, matrix, (out_shape[1], out_shape[0]))
        
        if is_iirs:
            warped = cv2.GaussianBlur(warped, (5, 5), 0)
            
    return warped