"""How example/input was made from the paper's working data (not needed to run the example).

Usage: python example/prepare_input.py <folder with the paper data>
All rasters end up on one 1 m grid nested in the 10 m Sentinel-2 pixels (EPSG:3794).
"""
import os
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy import stats

src = sys.argv[1]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
os.makedirs(out, exist_ok=True)
W, H = 750, 1000
tf = from_origin(548749.5, 158500.65625, 1.0, 1.0)
CRS = "EPSG:3794"


def save(name, arr, dtype, nodata=None):
    arr = arr[None] if arr.ndim == 2 else arr
    with rasterio.open(os.path.join(out, name), "w", driver="GTiff", width=W, height=H, count=len(arr),
                       dtype=dtype, crs=CRS, transform=tf, nodata=nodata, compress="deflate") as dst:
        dst.write(arr.astype(dtype))


def to_grid(path, band, resampling):
    dst = np.zeros((H, W), np.float32)
    with rasterio.open(path) as s:
        reproject(rasterio.band(s, band), dst, src_crs=CRS, dst_transform=tf, dst_crs=CRS, resampling=resampling)
    return dst


def blocks(a):  # (10H, 10W) -> (H, W, 100) blocks of 10 x 10 pixels
    return a[:H * 10, :W * 10].reshape(H, 10, W, 10).transpose(0, 2, 1, 3).reshape(H, W, 100)


# true orthophoto RGB at 0.1 m (aligned to the LiDAR grid) -> 1 m mean, NIR at 0.2 m -> 1 m average
with rasterio.open(os.path.join(src, "corrected_orthophoto.tif")) as s:
    rgb = [blocks(s.read(b)).mean(-1) for b in (1, 2, 3)]
nir = to_grid(os.path.join(src, "2022_ir.tif"), 1, Resampling.average)
save("ortho.tif", np.clip(np.rint(np.stack(rgb + [nir])), 1, 255), "uint8", 0)

# SAM segment labels at 0.1 m (Table 2 parameters, before cDSM filtering) -> 1 m majority
with rasterio.open(os.path.join(src, "orthophoto_segmented_no_ndsm.tif_3000_2_0.85_0.9_64_1_2.tif"
                                     "_segment_indices.tif")) as s:
    seg = blocks(s.read(1))
save("segments.tif", stats.mode(seg, axis=-1, keepdims=False).mode, "int32", 0)

# cDSM classes from the classified LiDAR, 1 m
save("cdsm.tif", np.rint(to_grid(os.path.join(src, "corrected_mb_class.tif"), 1, Resampling.nearest)), "uint8", 0)

# Sentinel-2 L2A broadband albedo (Eq. 1) of 7 April 2022, 10 m, already on the grid
with rasterio.open(os.path.join(src, "albedo_final_2022.tif")) as s, \
        rasterio.open(os.path.join(out, "albedo_s2.tif"), "w", driver="GTiff", width=75, height=100, count=1,
                      dtype="float32", crs=CRS, transform=from_origin(548749.5, 158500.65625, 10, 10)) as dst:
    dst.write(s.read(1).astype("float32"), 1)

# annotated albedo range [min, max] and material class of every pixel (Table 5), for validation only
gt = os.path.join(src, "newgroundtruth")
with rasterio.open(os.path.join(gt, "ground_truth_albedo_minmax.tif")) as s:
    save("ground_truth_minmax.tif", s.read()[:, :H, :W], "float32", -1)
with rasterio.open(os.path.join(gt, "ground_truth_classes_indices.tif")) as s:
    save("ground_truth_classes.tif", s.read(1)[:H, :W], "uint8", 0)

# the paper's super-resolution baselines (Table 6): Sentinel-2 bands of a larger window (850 x 1100 m) upscaled
# to 1 m by ESRGAN and BSRGAN, broadband albedo by Eq. 1. The window starts 49.5 m west and 49.3 m north of the
# study area, so the rasters are placed on the example grid by their georeference (nearest pixel)
for name, f in (("ESRGAN", "center_crop_upscaled.tif_albedo.tif"), ("BSRGAN", "center_crop_upscaled_2.tif_albedo.tif")):
    save(f"baseline_{name}.tif", to_grid(os.path.join(src, "super_res", f), 1, Resampling.nearest), "float32")
