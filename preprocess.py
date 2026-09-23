"""
Stage 4: Optical Preprocessor
-------------------------------
Log Transform -> Normalization -> Masked CLAHE.
Straight port of the three functions from image_procesiing.ipynb, chained
into a single `preprocess()` call.
"""
import numpy as np
import cv2


def log_transform(raw_array):
    valid_mask = raw_array > 0
    log_data = np.zeros_like(raw_array, dtype=np.float32)
    log_data[valid_mask] = np.log1p(raw_array[valid_mask])
    return log_data


def normalize_lunar_patch(crop_array):
    valid_mask = crop_array > 0
    raw_data = crop_array[valid_mask]

    if len(raw_data) == 0:
        return np.zeros(crop_array.shape, dtype=np.uint8)

    noise_floor = np.percentile(raw_data, 3)
    cleaned_data = np.where(crop_array > noise_floor, crop_array, 0)
    cleaned_mask = cleaned_data > 0

    if np.sum(cleaned_mask) == 0:
        return np.zeros(crop_array.shape, dtype=np.uint8)

    min_val = np.min(cleaned_data[cleaned_mask])
    max_val = np.max(cleaned_data[cleaned_mask])

    normalized_8bit = np.zeros(crop_array.shape, dtype=np.uint8)
    if max_val > min_val:
        stretched = (cleaned_data[cleaned_mask] - min_val) / (max_val - min_val) * 255.0
        normalized_8bit[cleaned_mask] = stretched.astype(np.uint8)
    return normalized_8bit


def apply_masked_clahe(norm_img, original_raw_image, clip_limit=2.0, tile_grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    clahe_img = clahe.apply(norm_img)
    valid_mask = original_raw_image > 0
    final_img = np.zeros_like(clahe_img)
    final_img[valid_mask] = clahe_img[valid_mask]
    return final_img


def preprocess(raw_array):
    """raw_array -> uint8 CLAHE-enhanced image, masking out no-data pixels."""
    logged = log_transform(raw_array)
    normalized = normalize_lunar_patch(logged)
    return apply_masked_clahe(normalized, raw_array)
