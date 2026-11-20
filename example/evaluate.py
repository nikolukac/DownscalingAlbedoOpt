"""Validate example/output/albedo.tif against the annotated albedo ranges and materials, and draw a figure.

Baselines: OpenCV and GDAL interpolation of the 10 m Sentinel-2 albedo to 1 m and every
example/input/baseline_*.tif. All rasters are compared pixel by pixel on the same grid.
"""
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage

here = os.path.dirname(os.path.abspath(__file__))
inp, out = os.path.join(here, "input"), os.path.join(here, "output")
MATERIALS = {1: "asphalt", 2: "soil", 3: "grass", 5: "trees", 6: "concrete", 7: "clay", 8: "water",
             9: "metal", 10: "wood"}


def read(name, band=1):
    with rasterio.open(os.path.join(here, name)) as s:
        return s.read(band).astype(np.float64)


def range_error(est, lo, hi):
    """Eqs. 17-18: distance of est to the annotated range [lo, hi], relative to the bound it is measured to."""
    below, above = est < lo, est >= hi
    err = np.where(below, lo - est, np.where(above, est - hi, 0.0))
    return err, err / np.where(below, lo, hi)


def edge_f1(est, cls):
    """F1 of the map's edges against the material boundaries of the ground truth (1 px tolerance,
    best gradient threshold per map)."""
    gt = (cls != np.roll(cls, 1, 0)) | (cls != np.roll(cls, 1, 1))
    gt[0], gt[:, 0] = False, False
    near_gt = ndimage.binary_dilation(gt)
    est = np.where(np.isfinite(est), est, np.nanmean(est))
    grad = np.hypot(ndimage.sobel(est, 0), ndimage.sobel(est, 1)) / 8
    best = 0.0
    for t in np.linspace(0.002, 0.05, 25):
        e = grad > t
        if e.any():
            p, r = (e & near_gt).sum() / e.sum(), (gt & ndimage.binary_dilation(e)).sum() / gt.sum()
            best = max(best, 2 * p * r / (p + r + 1e-12))
    return best


def ordering(est, cls, pairs, block):
    """Share of material pairs (a darker than b by their ranges) whose mean albedo keeps that order
    inside mixed coarse pixels (no material covers 90%). A tie counts as half."""
    ok = tot = 0
    for i in range(0, cls.shape[0] - block + 1, block):
        for j in range(0, cls.shape[1] - block + 1, block):
            c, v = cls[i:i + block, j:j + block], est[i:i + block, j:j + block]
            if np.bincount(c.ravel()).max() >= 0.9 * c.size:
                continue
            for a, b in pairs:
                ma, mb = c == a, c == b
                if ma.sum() >= 5 and mb.sum() >= 5:
                    va, vb = v[ma].mean(), v[mb].mean()
                    tot += 1
                    ok += 0.5 if np.isclose(va, vb, rtol=0, atol=1e-9) else float(va < vb)
    return ok / max(tot, 1)


est = read("output/albedo.tif")
lo, hi = read("input/ground_truth_minmax.tif", 1), read("input/ground_truth_minmax.tif", 2)
cls = read("input/ground_truth_classes.tif").astype(int)
s2 = read("input/albedo_s2.tif").astype(np.float32)
methods = {"DE, bicubic start": est}
for name, flag in (("nearest", cv2.INTER_NEAREST), ("bicubic", cv2.INTER_CUBIC), ("Lanczos-4", cv2.INTER_LANCZOS4)):
    methods[f"OpenCV {name}"] = cv2.resize(s2, est.shape[::-1], interpolation=flag).astype(np.float64)
with rasterio.open(os.path.join(inp, "albedo_s2.tif")) as s, rasterio.open(os.path.join(out, "albedo.tif")) as d:
    cubic = np.full(est.shape, np.nan)
    reproject(s.read(1), cubic, src_transform=s.transform, src_crs=s.crs, dst_transform=d.transform,
              dst_crs=d.crs, resampling=Resampling.cubic)
    block = round(s.transform.a / d.transform.a)
methods["GDAL cubic"] = cubic  # the interpolation albedo_de.py starts from
for f in sorted(glob.glob(os.path.join(inp, "baseline_*.tif"))):
    methods[os.path.basename(f)[9:-4]] = read(f)

span = {c: (lo[cls == c].mean(), hi[cls == c].mean()) for c in MATERIALS}
pairs = [(a, b) for a in MATERIALS for b in MATERIALS if span[a][1] <= span[b][0] + 1e-6]
lines = ["Against the annotated ground truth, all 1 m pixels.",
         "MAE, RMSE, MAPE: distance to the annotated albedo range (Eqs. 17-18).",
         "Hit: pixels inside the range. Hit/mat: hit rate averaged over the materials.",
         "Edge F1: map edges against material boundaries. Order: material pairs in the right order in mixed "
         "coarse pixels.", "",
         f"{'method':<20}{'MAE':>8}{'RMSE':>8}{'MAPE%':>8}{'Hit%':>7}{'Hit/mat%':>10}{'Edge F1':>9}{'Order%':>8}"]
for name, a in methods.items():
    err, rel = range_error(a, lo, hi)
    hit = (a >= lo) & (a < hi)
    lines.append(f"{name:<20}{err.mean():>8.4f}{np.sqrt((err ** 2).mean()):>8.4f}{100 * rel.mean():>8.2f}"
                 f"{100 * hit.mean():>7.1f}{100 * np.mean([hit[cls == c].mean() for c in MATERIALS]):>10.1f}"
                 f"{edge_f1(a, cls):>9.3f}{100 * ordering(a, cls, pairs, block):>8.1f}")
print("\n".join(lines))
with open(os.path.join(out, "evaluation.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")

ortho = np.moveaxis(rasterio.open(os.path.join(inp, "ortho.tif")).read()[:3], 0, -1)
fig, ax = plt.subplots(1, 5, figsize=(21, 5.4))
ax[0].imshow(ortho)
ax[0].set_title("True orthophoto (RGB, 1 m)")
panels = ((s2, "Sentinel-2 albedo (10 m)"), (methods["OpenCV bicubic"], "Bicubic (1 m)"),
          (est, "DE downscaling (1 m)"))
for a, (img, title) in zip(ax[1:4], panels):
    im = a.imshow(img, cmap="gray", vmin=0, vmax=0.5)
    a.set_title(title)
fig.colorbar(im, ax=ax[1:4], fraction=0.02, pad=0.01, label="albedo")
im = ax[4].imshow(range_error(est, lo, hi)[0], cmap="Reds", vmin=0, vmax=0.1)
ax[4].set_title("DE error against ground truth")
fig.colorbar(im, ax=ax[4], fraction=0.05, pad=0.02, label="absolute error")
for a in ax:
    a.set_axis_off()
fig.savefig(os.path.join(out, "example.png"), dpi=100, bbox_inches="tight")
