"""Optimization-based downscaling of satellite broadband albedo with Differential Evolution.

Improved version of the method of N. Lukač, D. Mongus, M. Bizjak, "Optimization-Based Downscaling of
Satellite-Derived Isotropic Broadband Albedo to High Resolution", Remote Sensing 17(8), 1366 (2025),
https://doi.org/10.3390/rs17081366: DE fits an albedo offset per material on top of a bicubic start, in
overlapping windows. Equation numbers in the docstrings refer to the paper.
"""
import argparse
import csv
import os
import time

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage, sparse
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

# Eq. 1: Sentinel-2 L2A narrow-to-broadband weights for B2, B3, B4, B8, B11, B12 (Bonafoni and Sekertekin)
S2_NTB = np.array([0.2266, 0.1236, 0.1573, 0.3417, 0.1170, 0.0338])
MUTATIONS = ("rand/1", "best/1", "current-to-best/1", "rand/2", "best/2")
OFFSET = 0.3  # largest albedo offset per segment on top of the bicubic base


def ntb_albedo(s2):
    """Eq. 1: isotropic broadband albedo from a (6, H, W) Sentinel-2 L2A reflectance stack in [0, 1]."""
    return np.tensordot(S2_NTB, s2, axes=1)


def filter_segments(labels, cdsm, pixel_area, t1=0.75, t2=2.5, ground=2):
    """Eq. 3: remove small segments (area below t2 m2) with more than a t1 share of ground pixels in the cDSM."""
    ids, inv, n = np.unique(labels, return_inverse=True, return_counts=True)
    share = np.bincount(inv.ravel(), weights=(cdsm == ground).ravel()) / n
    drop = (ids > 0) & (share > t1) & (n * pixel_area < t2)
    return np.where(drop[inv].reshape(labels.shape), 0, labels)


def merge_segments(labels, x, k, seed=0):
    """K-means++ over the mean feature vectors of the L segments (Sec. 2.1.2), giving K noncontiguous segments.

    Returns the segment index of every pixel (-1 where there is none) and the K mean feature vectors."""
    m = (labels > 0) & np.isfinite(x).all(0)
    ids, inv = np.unique(labels[m], return_inverse=True)
    feats = np.stack([np.bincount(inv, weights=band[m]) for band in x], 1) / np.bincount(inv)[:, None]
    km = KMeans(min(k, len(ids)), n_init=10, random_state=seed).fit(feats)
    seg = np.full(labels.shape, -1)
    seg[m] = km.labels_[inv]
    xk = np.stack([np.bincount(seg[m], weights=band[m]) for band in x], 1) / np.bincount(seg[m])[:, None]
    return seg, xk


def coarse_coords(shape, fine_tf, coarse_tf):
    """Position of every fine pixel centre in coarse pixel units (column, row)."""
    col, row = np.meshgrid(np.arange(shape[1]) + 0.5, np.arange(shape[0]) + 0.5)
    return ~coarse_tf * (fine_tf * (col, row))


def link(seg, k, cc, rr, alpha):
    """Eqs. 4-5: share n_ki / sum_k n_ki of segment k in coarse pixel i, as a sparse matrix.

    Only coarse pixels with a valid albedo that contain segments are kept. Returns the matrix,
    their albedo and their flat indices in the coarse raster."""
    cc, rr = np.floor(cc).astype(int), np.floor(rr).astype(int)
    hc, wc = alpha.shape
    ok = (seg >= 0) & (rr >= 0) & (rr < hc) & (cc >= 0) & (cc < wc)
    i, kk = rr[ok] * wc + cc[ok], seg[ok]
    good = np.isfinite(alpha.ravel()[i])
    used, row = np.unique(i[good], return_inverse=True)
    n = sparse.csr_matrix((np.ones(good.sum()), (row, kk[good])), shape=(len(used), k))
    n = sparse.diags(1 / np.asarray(n.sum(1)).ravel()) @ n
    return n.tocsr(), alpha.ravel()[used], used


def bicubic(alpha, coarse_tf, shape, fine_tf, crs):
    """Bicubic interpolation of the coarse albedo onto the fine grid."""
    out = np.full(shape, np.nan)
    reproject(alpha, out, src_transform=coarse_tf, src_crs=crs, dst_transform=fine_tf, dst_crs=crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.cubic)
    return out


def segment_mean(img, labels):
    """Mean of img within every segment (label > 0); pixels without a segment keep their value."""
    lab, v = labels.ravel(), img.ravel()
    ok = np.isfinite(v)
    s = np.bincount(lab[ok], weights=v[ok], minlength=lab.max() + 1)
    s /= np.maximum(np.bincount(lab[ok], minlength=lab.max() + 1), 1)
    return np.where(labels > 0, s[labels], img)


def objective(pop, n, alpha):
    """E (Eq. 10) of every individual (M, K): sum over coarse pixels of |alpha_i - estimated alpha_i|."""
    return np.abs(n @ pop.T - alpha[:, None]).sum(0)


def de(fitness, pop, g=5000, mutation="current-to-best/1", crossover="bin",
       tau=0.1, delta=1e-5, patience=10, seed=0, log=None):
    """Self-adaptive Differential Evolution (Sec. 2.2) of a population (M, B, K, C) with values in [0, 1].

    The B problems evolve independently, each with its own best individual; fitness(pop) returns (M, B).
    Stops after g generations, or (after at least 100) when the summed best objective improved by less
    than delta over the last `patience` generations (delta=0 disables this). Returns the best individual of every problem
    (B, K, C) and the summed best objective of every generation."""
    rng = np.random.default_rng(seed)
    pop = pop.copy()
    m, b, k = pop.shape[:3]
    fit = fitness(pop)
    f, cr = np.full((m, b), 0.8), np.full((m, b), 0.9)
    hist = []
    for gen in range(g):
        # Eqs. 7-8: every individual carries its own F and CR
        fn = np.where(rng.random((m, b)) < tau, 0.1 + 0.9 * rng.random((m, b)), f)
        crn = np.where(rng.random((m, b)) < tau, rng.random((m, b)), cr)
        # five distinct random partners per individual, never the individual itself
        key = rng.random((m, m))
        np.fill_diagonal(key, 2)
        r = np.argsort(key, 1)[:, :5]
        q = lambda j: pop[r[:, j]]
        best, F = pop[np.argmin(fit, 0), np.arange(b)], fn[..., None, None]
        if mutation == "rand/1":
            v = q(0) + F * (q(1) - q(2))
        elif mutation == "best/1":
            v = best + F * (q(0) - q(1))
        elif mutation == "current-to-best/1":
            v = pop + F * (best - pop) + F * (q(0) - q(1))
        elif mutation == "rand/2":
            v = q(0) + F * (q(1) - q(2)) + F * (q(3) - q(4))
        else:  # best/2
            v = best + F * (q(0) - q(1)) + F * (q(2) - q(3))
        if crossover == "bin":
            take = rng.random((m, b, k)) < crn[..., None]
        else:  # exponential: a run of consecutive segments from a random start
            start = rng.integers(k, size=(m, b))
            run = 1 + np.cumprod(rng.random((m, b, k - 1)) < crn[..., None], -1).sum(-1)
            take = (np.arange(k) - start[..., None]) % k < run[..., None]
        u = np.clip(np.where(take[..., None], v, pop), 0, 1)
        fu = fitness(u)
        win = fu < fit
        pop[win], fit[win], f[win], cr[win] = u[win], fu[win], fn[win], crn[win]
        hist.append(fit.min(0).sum())
        if log:
            log(gen, hist[-1], f.mean(), cr.mean())
        if delta > 0 and gen >= max(patience, 100) and hist[-patience - 1] - hist[-1] < delta:
            break
    return pop[np.argmin(fit, 0), np.arange(b)], hist


def local_albedo(ak, n, alpha, used, shape, radius, lam, m=50, g=600, seed=0, **kw):
    """Local fit (an extension of the paper): DE fits the K segment offsets in overlapping windows of the
    coarse grid, so that one material may differ across the scene.

    Window centres lie every `radius` coarse pixels. A window reaches 2 * radius with Gaussian weights
    (sigma = radius) and adds the L1 penalty lam * (weighted share of the segment + 0.01 * weight sum)
    on the distance from the global values ak. Returns the local values on the grid of window centres
    (rows, cols, K)."""
    nd = n.toarray()
    r_i, c_i = np.divmod(used, shape[1])
    grid = np.tile(ak, (len(range(0, shape[0], radius)), len(range(0, shape[1], radius)), 1))
    cells, blocks, idx, wts = [], [], [], []
    for gr, r0 in enumerate(range(0, shape[0], radius)):
        for gc, c0 in enumerate(range(0, shape[1], radius)):
            d2 = ((r_i - r0) ** 2 + (c_i - c0) ** 2) / radius ** 2
            i = np.nonzero(d2 <= 4)[0]
            if len(i):
                cells.append((gr, gc))
                idx.append(i)
                wts.append(np.exp(-d2[i] / 2))
                blocks.append(sparse.csr_matrix(nd[i]))
    nb = sparse.block_diag(blocks, format="csr")
    i, w = np.concatenate(idx), np.concatenate(wts)
    starts = np.cumsum([0] + [len(j) for j in idx[:-1]])
    pen = lam * np.stack([wj @ nd[j] + 0.01 * wj.sum() for j, wj in zip(idx, wts)])

    def fitness(pop):
        p = pop[..., 0]
        pred = nb @ p.reshape(len(p), -1).T
        data = np.add.reduceat(np.abs(pred - alpha[i][:, None]) * w[:, None], starts)
        return data.T + (np.abs(p - ak) * pen).sum(-1)

    rng = np.random.default_rng(seed)
    pop = np.clip(ak[:, None] + rng.uniform(-0.02, 0.02, (m, len(blocks), len(ak), 1)), 0, 1)
    pop[0] = ak[:, None]
    best, _ = de(fitness, pop, g, delta=0, seed=seed, **kw)
    for (gr, gc), v in zip(cells, best[..., 0]):
        grid[gr, gc] = v
    return grid


def render(seg, x, xk, a, grid=None, cc=None, rr=None, radius=1):
    """Eqs. 14-16: albedo of every fine pixel from its segment.

    Pixels without a segment (gaps, removed segments) take the segment with the nearest mean feature
    vector (Eq. 15). With local albedos (grid), a pixel's value is interpolated bilinearly between the
    window centres around it. Returns the albedo and the segment index of every pixel."""
    seg = seg.copy()
    miss = (seg < 0) & np.isfinite(x).all(0)
    if miss.any():
        seg[miss] = pairwise_distances_argmin(x[:, miss].T, xk)
    out = np.where(seg >= 0, a[seg], np.nan)
    if grid is not None:
        for k in range(len(a)):
            mk = seg == k
            pos = [(rr[mk] - 0.5) / radius, (cc[mk] - 0.5) / radius]
            out[mk] = ndimage.map_coordinates(grid[..., k], pos, order=1, mode="nearest")
    return out, seg


def aggregate(albedo, cc, rr, used, wc):
    """Mean of the fine albedo inside every used coarse pixel."""
    i = (np.floor(rr) * wc + np.floor(cc)).astype(int).ravel()
    ok = np.isfinite(albedo.ravel()) & (rr.ravel() >= 0) & (cc.ravel() >= 0) & (cc.ravel() < wc)
    s = np.bincount(i[ok], weights=albedo.ravel()[ok], minlength=used.max() + 1)
    return s[used] / np.maximum(np.bincount(i[ok], minlength=used.max() + 1)[used], 1)


def metrics(est, ref):
    """MAE, RMSE and MAPE [%] of est against ref (MAPE over ref > 0)."""
    d, p = est - ref, ref > 0
    return {"MAE": np.mean(np.abs(d)), "RMSE": np.sqrt(np.mean(d ** 2)),
            "MAPE": 100 * np.mean(np.abs(d[p]) / ref[p])}


def read(path):
    """Raster as float64 (bands, H, W) with nodata as NaN, and its profile."""
    with rasterio.open(path) as src:
        a, prof = src.read(), src.profile
    out = a.astype(np.float64)
    if prof.get("nodata") is not None:
        out[a == prof["nodata"]] = np.nan
    return out, prof


def write(path, arr, like, dtype, nodata):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=dtype, crs=like["crs"], transform=like["transform"], nodata=nodata,
                       compress="deflate") as dst:
        dst.write(arr.astype(dtype), 1)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ortho", required=True, help="high-resolution image (e.g. R, G, B, NIR) on the target grid")
    p.add_argument("--segments", required=True, help="segment labels on the same grid, 0 = no segment")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--albedo", help="coarse broadband albedo raster (first band)")
    src.add_argument("--s2", help="Sentinel-2 L2A stack B2, B3, B4, B8, B11, B12 as reflectance 0-1 (Eq. 1)")
    p.add_argument("--cdsm", help="cDSM class raster on the ortho grid; enables the filter of Eq. 3")
    p.add_argument("--ground-class", type=int, default=2, help="ground class code in the cDSM (default 2)")
    p.add_argument("--t1", type=float, default=0.75, help="ground share threshold T1 (default 0.75)")
    p.add_argument("--t2", type=float, default=2.5, help="area threshold T2 in m2 (default 2.5)")
    p.add_argument("-k", type=int, default=100, help="number of K-means segments K (default 100)")
    p.add_argument("--mix", type=float, default=0.5, help="share of the plain bicubic map in the start, the "
                   "rest is its mean within every input segment (default 0.5)")
    p.add_argument("--mutation", choices=MUTATIONS, default="current-to-best/1")
    p.add_argument("--crossover", choices=("bin", "exp"), default="bin")
    p.add_argument("--radius", type=float, default=75, help="local fit: window spacing in m (default 75, "
                   "0 = one offset per segment for the whole scene)")
    p.add_argument("--lam", type=float, default=0.2, help="local fit: L1 penalty that keeps the offsets "
                   "small, 0 to 1 (default 0.2)")
    p.add_argument("--pop", type=int, default=100, help="with --radius 0: population size M (default 100)")
    p.add_argument("--gen", type=int, default=5000, help="with --radius 0: maximum generations G (default 5000)")
    p.add_argument("--delta", type=float, default=1e-5, help="with --radius 0: stop when E improves less than "
                   "this in 10 generations (default 1e-5, 0 = run all G)")
    p.add_argument("--local-gen", type=int, default=600, help="local fit: DE generations (default 600)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="output", help="output folder")
    a = p.parse_args()
    t0 = time.time()

    x, prof = read(a.ortho)
    if np.issubdtype(np.dtype(prof["dtype"]), np.integer):
        x /= np.iinfo(prof["dtype"]).max
    with rasterio.open(a.segments) as s:
        labels = s.read(1)
    if labels.shape != x.shape[1:]:
        raise SystemExit("--segments must be on the same grid as --ortho")
    if a.cdsm:
        with rasterio.open(a.cdsm) as s:
            cdsm = s.read(1)
        before = len(np.unique(labels)) - 1
        tf = prof["transform"]
        labels = filter_segments(labels, cdsm, abs(tf.a * tf.e), a.t1, a.t2, a.ground_class)
        print(f"cDSM filter: removed {before - len(np.unique(labels)) + 1} of {before} segments")
    seg, xk = merge_segments(labels, x, a.k, a.seed)
    k = len(xk)
    _, seg_all = render(seg, x, xk, np.zeros(k))

    coarse, cprof = read(a.albedo or a.s2)
    if cprof["crs"] != prof["crs"]:
        raise SystemExit("the coarse raster and --ortho must share a CRS")
    alpha = coarse[0] if a.albedo else ntb_albedo(coarse)
    cc, rr = coarse_coords(seg.shape, prof["transform"], cprof["transform"])
    # start from bicubic interpolation, partly levelled within segments; DE then finds an offset
    # in [-OFFSET, OFFSET] per segment so that every coarse mean matches the satellite again
    base = bicubic(alpha, cprof["transform"], seg.shape, prof["transform"], prof["crs"])
    base = a.mix * base + (1 - a.mix) * segment_mean(base, labels)
    n, alpha_i, used = link(seg_all, k, cc, rr, alpha)
    scale = 2 * OFFSET
    target = (alpha_i - aggregate(base, cc, rr, used, alpha.shape[1])) / scale + 0.5  # offsets mapped to [0, 1]
    print(f"{k} segments, {len(alpha_i)} coarse pixels, DE/{a.mutation}/{a.crossover}")

    os.makedirs(a.out, exist_ok=True)
    ak = np.full(k, 0.5)  # zero offsets: the local fit starts from the start map itself
    if a.radius <= 0:
        # one offset per segment for the whole scene
        rng = np.random.default_rng(a.seed)
        pop = np.clip(rng.normal(0.5, 0.02, (a.pop, 1, k, 1)), 0, 1)
        pop[0] = 0.5
        with open(os.path.join(a.out, "convergence.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["generation", "E", "MAE", "mean_F", "mean_CR"])

            def log(gen, e, f, cr):
                e *= scale
                w.writerow([gen + 1, f"{e:.6f}", f"{e / len(alpha_i):.6f}", f"{f:.4f}", f"{cr:.4f}"])
                if (gen + 1) % 100 == 0:
                    print(f"  generation {gen + 1}: E = {e:.4f}, MAE = {e / len(alpha_i):.5f}")

            best, hist = de(lambda q: objective(q[:, 0, :, 0], n, target)[:, None], pop, a.gen,
                            a.mutation, a.crossover, delta=a.delta, seed=a.seed, log=log)
        ak = best[0, :, 0]
        print(f"global DE: {len(hist)} generations")

    grid, r = None, 1
    if a.radius > 0:
        r = max(1, round(a.radius / abs(cprof["transform"].a)))
        grid = local_albedo(ak, n, target, used, alpha.shape, r, a.lam, g=a.local_gen, seed=a.seed,
                            mutation=a.mutation, crossover=a.crossover)
        print(f"local fit: {grid.shape[0]} x {grid.shape[1]} windows every {r} coarse pixels")
    v, seg_all = render(seg, x, xk, ak, grid, cc, rr, r)
    albedo = np.clip(base + scale * (v - 0.5), 0, 1)
    s = metrics(aggregate(albedo, cc, rr, used, alpha.shape[1]), alpha_i)
    print(f"against the coarse albedo: MAE {s['MAE']:.4f}, RMSE {s['RMSE']:.4f}, MAPE {s['MAPE']:.2f}%")

    write(os.path.join(a.out, "albedo.tif"), albedo, prof, "float32", np.nan)
    write(os.path.join(a.out, "segments_k.tif"), seg_all, prof, "int16", -1)
    with open(os.path.join(a.out, "segments_k.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["segment", "pixels"] + [f"mean_band{b + 1}" for b in range(len(x))] + ["offset"])
        counts = np.bincount(seg_all[seg_all >= 0], minlength=k)
        mean_v = np.bincount(seg_all[seg_all >= 0], weights=v[seg_all >= 0], minlength=k) / np.maximum(counts, 1)
        offset = scale * (mean_v - 0.5)  # mean albedo offset of the segment's pixels
        for i in range(k):
            w.writerow([i, counts[i]] + [f"{u:.5f}" for u in (*xk[i], offset[i])])
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
