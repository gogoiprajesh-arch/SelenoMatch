"""
Stage 7: Match Fusion
------------------------
Merges the two branches' candidate matches into one coordinate-consistent
list, per tile, weighted by solar incidence (RIFT-2's phase-congruency
core is more robust at grazing illumination; LoFTR's learned features do
better under flat lighting). Tile-local (row, col) coordinates are
translated into the shared global overlap frame here, so everything
downstream (RANSAC, sub-pixel refine, visualization) works in one
coordinate system.

Also carries over `enforce_uniform_distribution` from aggregate_results.py
-- it existed there but was never actually called before the .npz was
written, so dense crater-rim texture could dominate the whole match set.
It's wired in here for real.
"""
import numpy as np


def solar_weights(solar_incidence_deg):
    """RIFT-2 favored at grazing/low-sun, LoFTR favored at mid/high sun."""
    w_rift = 1.2 if solar_incidence_deg > 70.0 else (0.8 if solar_incidence_deg < 40.0 else 1.0)
    w_loftr = 0.7 if solar_incidence_deg > 70.0 else (1.1 if solar_incidence_deg < 40.0 else 1.0)
    return w_rift, w_loftr


def fuse_tile_matches(pts1_rift, pts2_rift, conf_rift,
                       pts1_loftr, pts2_loftr, conf_loftr,
                       y_off, x_off, solar_incidence_deg=65.0):
    """
    Weights + concatenates both branches for one tile and shifts the
    coordinates into the global overlap frame.
    Returns pts1_global, pts2_global, conf (all float32 arrays), plus a
    per-point 'source' label array ('rift' / 'loftr') for diagnostics.
    """
    w_rift, w_loftr = solar_weights(solar_incidence_deg)

    conf_rift = np.asarray(conf_rift, dtype=np.float32) * w_rift if len(conf_rift) else np.asarray(conf_rift, dtype=np.float32)
    conf_loftr = np.asarray(conf_loftr, dtype=np.float32) * w_loftr if len(conf_loftr) else np.asarray(conf_loftr, dtype=np.float32)

    parts1, parts2, confs, sources = [], [], [], []
    if len(pts1_rift) > 0:
        parts1.append(pts1_rift); parts2.append(pts2_rift); confs.append(conf_rift)
        sources.append(np.full(len(pts1_rift), "rift"))
    if len(pts1_loftr) > 0:
        parts1.append(pts1_loftr); parts2.append(pts2_loftr); confs.append(conf_loftr)
        sources.append(np.full(len(pts1_loftr), "loftr"))

    if not parts1:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty, np.zeros((0,), dtype=np.float32), np.array([], dtype=object)

    pts1 = np.vstack(parts1)
    pts2 = np.vstack(parts2)
    conf = np.concatenate(confs)
    source = np.concatenate(sources)

    offset = np.array([x_off, y_off], dtype=np.float32)  # pts are (x, y)
    pts1_global = pts1 + offset
    pts2_global = pts2 + offset

    return pts1_global, pts2_global, conf, source


def enforce_uniform_distribution(pts1, pts2, conf, img_shape, grid_size=10, max_per_cell=15):
    """Cap matches per spatial grid cell so dense texture regions (crater
    rims) can't dominate the final match set at the expense of flat terrain."""
    if len(pts1) == 0:
        return pts1, pts2, conf

    h, w = img_shape[:2]
    cell_h, cell_w = h / grid_size, w / grid_size

    cell_x = (pts1[:, 0] // cell_w).astype(int).clip(0, grid_size - 1)
    cell_y = (pts1[:, 1] // cell_h).astype(int).clip(0, grid_size - 1)
    cell_id = cell_y * grid_size + cell_x

    keep_idx = []
    empty_cells = 0
    for c in range(grid_size * grid_size):
        idx_in_cell = np.where(cell_id == c)[0]
        if len(idx_in_cell) == 0:
            empty_cells += 1
            continue
        top = idx_in_cell[np.argsort(-conf[idx_in_cell])[:max_per_cell]]
        keep_idx.extend(top.tolist())

    print(f"[fusion] Grid coverage: {grid_size * grid_size - empty_cells}/{grid_size * grid_size} "
          f"cells have matches ({empty_cells} empty)")

    keep_idx = np.array(keep_idx)
    return pts1[keep_idx], pts2[keep_idx], conf[keep_idx]
