"""
Stage 1: Metadata Parser
------------------------
Extracts image corners (lat/lon) and physical resolution (m/px) from either
an ISRO PDS3/XML label (TMC, OHRC) or a NASA LRO GeoTIFF.

This is a thin, dispatch-based merge of the two corner-extraction functions
that lived in image_procesiing.ipynb (`get_corners_and_resolution` and
`get_lro_corners_and_resolution`) so callers don't need to know which
product type they're handed.
"""
import xml.etree.ElementTree as ET
import rasterio
from rasterio.warp import transform_bounds

LUNAR_LATLON_CRS = "+proj=latlong +R=1737400 +no_defs"


def find_by_local_name(root, local_name):
    """XML namespaces vary by product; match on the tag's local name only."""
    for elem in root.iter():
        tag = elem.tag.split('}')[-1]
        if tag == local_name:
            return elem.text
    return None


def get_corners_and_resolution_xml(xml_path):
    """ISRO PDS3/XML label (Chandrayaan TMC/OHRC)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    corners = {
        'ul': (float(find_by_local_name(root, "upper_left_latitude")), float(find_by_local_name(root, "upper_left_longitude"))),
        'ur': (float(find_by_local_name(root, "upper_right_latitude")), float(find_by_local_name(root, "upper_right_longitude"))),
        'll': (float(find_by_local_name(root, "lower_left_latitude")), float(find_by_local_name(root, "lower_left_longitude"))),
        'lr': (float(find_by_local_name(root, "lower_right_latitude")), float(find_by_local_name(root, "lower_right_longitude"))),
    }
    resolution = float(find_by_local_name(root, "pixel_resolution"))
    return corners, resolution


def get_corners_and_resolution_tif(tif_path):
    """NASA LRO GeoTIFF (already georeferenced)."""
    with rasterio.open(tif_path) as src:
        resolution = src.res[0]
        left, bottom, right, top = src.bounds
        min_lon, min_lat, max_lon, max_lat = transform_bounds(
            src.crs, LUNAR_LATLON_CRS, left, bottom, right, top
        )
        corners = {
            'ul': (max_lat, min_lon),
            'ur': (max_lat, max_lon),
            'll': (min_lat, min_lon),
            'lr': (min_lat, max_lon),
        }
    return corners, resolution


def parse_metadata(path):
    """
    Dispatches on file extension.
    Returns: (corners dict, resolution m/px, product_type in {'xml', 'tif'})
    """
    path_str = str(path)
    if path_str.lower().endswith(".xml"):
        corners, res = get_corners_and_resolution_xml(path_str)
        return corners, res, "xml"
    elif path_str.lower().endswith((".tif", ".tiff")):
        corners, res = get_corners_and_resolution_tif(path_str)
        return corners, res, "tif"
    else:
        raise ValueError(
            f"Unrecognized product type for '{path_str}'. Expected a PDS3 "
            f".xml label (ISRO) or a Geo.tif (NASA LRO)."
        )
