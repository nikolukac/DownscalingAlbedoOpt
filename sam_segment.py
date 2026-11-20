"""Optional step: segment a true orthophoto with the Segment Anything Model (SAM), Sec. 2.1.2.

Writes an int32 label raster (0 = no segment) for albedo_de.py --segments. Needs
`pip install torch segment-anything` and a SAM checkpoint (https://github.com/facebookresearch/segment-anything).
"""
import argparse

import numpy as np
import rasterio
import torch
from scipy import ndimage, stats
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("image", help="orthophoto GeoTIFF, the first three bands are used")
    p.add_argument("checkpoint", help="SAM checkpoint, e.g. sam_vit_h_4b8939.pth")
    p.add_argument("out", help="output label GeoTIFF")
    p.add_argument("--model", default="vit_h", help="vit_h (default), vit_l or vit_b")
    p.add_argument("--tile", type=int, default=3000, help="tile size in pixels, tiles overlap by half (default 3000)")
    p.add_argument("--factor", type=int, default=1, help="write labels downsampled by this factor (majority), "
                   "e.g. 10 for a 0.1 m image and a 1 m grid")
    a = p.parse_args()

    sam = sam_model_registry[a.model](checkpoint=a.checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")
    gen = SamAutomaticMaskGenerator(sam, points_per_batch=64, pred_iou_thresh=0.85, stability_score_thresh=0.90,
                                    crop_n_layers=1, crop_n_points_downscale_factor=2)  # Table 2
    with rasterio.open(a.image) as src:
        img, prof = src.read([1, 2, 3]), src.profile
    if img.dtype != np.uint8:
        img = np.clip(img / (np.iinfo(img.dtype).max / 255 if img.dtype.kind in "ui" else 1 / 255), 0, 255)
    img = np.ascontiguousarray(np.moveaxis(img.astype(np.uint8), 0, -1))
    h, w = img.shape[:2]

    step, masks = a.tile // 2, []
    for r in range(0, max(h - a.tile, 0) + step, step):
        for c in range(0, max(w - a.tile, 0) + step, step):
            tile = img[r:r + a.tile, c:c + a.tile]
            th, tw = tile.shape[:2]
            print(f"tile {r}, {c}")
            for m in gen.generate(tile):
                s = m["segmentation"]
                # a mask cut by an inner tile edge is seen whole by the overlapping tile
                if (r and s[0].any()) or (c and s[:, 0].any()) or (r + th < h and s[-1].any()) or \
                        (c + tw < w and s[:, -1].any()):
                    continue
                x, y, bw, bh = m["bbox"]
                masks.append((m["area"], r + y, c + x, s[y:y + bh + 1, x:x + bw + 1]))

    # paint large masks first so that smaller ones (e.g. chimneys on roofs) stay on top
    labels, area = np.zeros((h, w), np.int32), [0]
    for _, r, c, s in sorted(masks, key=lambda m: -m[0]):
        view = labels[r:r + s.shape[0], c:c + s.shape[1]]
        s = s[:view.shape[0], :view.shape[1]]
        under = view[s]
        top = np.bincount(under).argmax()
        inter = (under == top).sum() if top else 0
        if inter > 0.9 * (s.sum() + area[top] - inter):  # the same segment from an overlapping tile
            continue
        area.append(s.sum())
        view[s] = len(area) - 1
    # pixels outside every mask (often roads) become segments too, one per connected region
    gaps, n = ndimage.label(labels == 0)
    labels[gaps > 0] = gaps[gaps > 0] + len(area) - 1

    tf = prof["transform"]
    if a.factor > 1:
        f, h, w = a.factor, h // a.factor, w // a.factor
        blocks = labels[:h * f, :w * f].reshape(h, f, w, f).transpose(0, 2, 1, 3).reshape(h, w, f * f)
        labels, tf = stats.mode(blocks, axis=-1, keepdims=False).mode, tf * tf.scale(f)
    with rasterio.open(a.out, "w", driver="GTiff", height=h, width=w, count=1, dtype="int32", crs=prof["crs"],
                       transform=tf, nodata=0, compress="deflate") as dst:
        dst.write(labels.astype(np.int32), 1)
    print(f"{len(area) - 1} SAM segments and {n} gap segments written to {a.out}")


if __name__ == "__main__":
    main()
