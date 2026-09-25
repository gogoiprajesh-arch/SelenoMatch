"""
Stage 3: Geo-Sampler [Rasterio]
--------------------------------
Reads *only* the overlap window from each raw file (never the full image),
then resamples both crops onto the shared `universal_out_shape` grid so the
two rasters are pixel-for-pixel co-registered before any pixel processing
starts.
"""
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import numpy as np
import cv2
from geometry import latlon_to_stereographic


def sample_overlap(path, bbox, out_shape=None, band=1, read_full_ram=False):
    """
    bbox: (row_min, row_max, col_min, col_max) in this file's own pixel grid
    out_shape: (rows, cols) to resample the crop onto. If None, keeps native shape.
    """
    r_min, r_max, c_min, c_max = bbox
    height = max(1, r_max - r_min)
    width = max(1, c_max - c_min)

    with rasterio.open(str(path)) as src:
        if read_full_ram:
            # Bypass Rasterio 3D window bug by reading the full band into RAM
            full_band = src.read(band)
            crop = full_band[r_min:r_max, c_min:c_max]
            
            if crop.size > 0 and out_shape is not None:
                data = cv2.resize(
                    crop, 
                    (out_shape[1], out_shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                data = crop if crop.size > 0 else np.zeros((height, width), dtype=np.float32)
        else:
            # Standard windowed read for single-band optical data
            window = Window(col_off=c_min, row_off=r_min, width=width, height=height)
            if out_shape is not None:
                data = src.read(
                    band,
                    window=window,
                    out_shape=out_shape,
                    resampling=Resampling.nearest,
                )
            else:
                # FAST PATH: Native read without map-grid stretching
                data = src.read(band, window=window)
                
    return data.astype(np.float32)

def sample_iirs_aligned(iirs_path, iirs_corners, wac_transform, wac_bbox, out_shape):
    """Warps raw IIRS pushbroom data directly onto the WAC crop's pixel grid."""
    with rasterio.open(str(iirs_path)) as src:
        # Average bands 40-60 to eliminate hyperspectral sensor striping
        band_range = list(range(40, 61))
        full_band = np.mean(src.read(band_range), axis=0).astype(np.float32)
        
        ul_x, ul_y = latlon_to_stereographic(*iirs_corners['ul'])
        ur_x, ur_y = latlon_to_stereographic(*iirs_corners['ur'])
        lr_x, lr_y = latlon_to_stereographic(*iirs_corners['lr'])
        ll_x, ll_y = latlon_to_stereographic(*iirs_corners['ll'])

        inv_wac = ~wac_transform
        r_min, r_max, c_min, c_max = wac_bbox

        def physical_to_crop_pixel(x, y):
            col, row = inv_wac * (x, y)
            return [col - c_min, row - r_min]

        dst_pts = np.array([
            physical_to_crop_pixel(ul_x, ul_y),
            physical_to_crop_pixel(ur_x, ur_y),
            physical_to_crop_pixel(lr_x, lr_y),
            physical_to_crop_pixel(ll_x, ll_y)
        ], dtype=np.float32)

        src_pts = np.array([
            [0, 0],
            [src.width - 1, 0],
            [src.width - 1, src.height - 1],
            [0, src.height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(full_band, matrix, (out_shape[1], out_shape[0]))
        
        # Melt high frequency noise
        warped = cv2.GaussianBlur(warped, (5, 5), 0)
        
    return warped