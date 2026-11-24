"""Self-contained radiomics extractor (NumPy + SciPy + SimpleITK, no PyRadiomics).

30 features per tumor:
    shape (8):        voxel_count, volume_mm3, surface_area_mm2, surface_volume_ratio,
                      sphericity, elongation, flatness, max_diameter_mm
    first-order (16): mean, median, min, max, range, p10, p90, iqr, std, variance,
                      skewness, kurtosis, energy, entropy, mad, rms
    GLCM (6):         contrast, dissimilarity, homogeneity, asm (energy), entropy, correlation
                      (32 gray levels, 13 3D directions, symmetric, averaged over directions)

    python -m pancvlm.radiomics --images /data/Task07_Pancreas/imagesTr \
        --labels /data/Task07_Pancreas/labelsTr --out radiomics/radiomics_features.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np
from scipy import ndimage, stats
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist

from .common import MSD_TUMOR, list_cases, read_array

SHAPE = ["voxel_count", "volume_mm3", "surface_area_mm2", "surface_volume_ratio",
         "sphericity", "elongation", "flatness", "max_diameter_mm"]
FIRST_ORDER = ["mean", "median", "min", "max", "range", "p10", "p90", "iqr", "std", "variance",
               "skewness", "kurtosis", "energy", "entropy", "mad", "rms"]
GLCM = ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_asm",
        "glcm_entropy", "glcm_correlation"]
FEATURES = [f"shape_{n}" for n in SHAPE] + [f"fo_{n}" for n in FIRST_ORDER] + GLCM
assert len(FEATURES) == 30


# ----------------------------------------------------------------------------- shape
def surface_area_faces(mask: np.ndarray, spacing_zyx) -> float:
    """Count exposed voxel faces, weighted by face area. Overestimates smooth surfaces
    (staircase effect) relative to marching cubes, but is exact for the voxel object."""
    m = np.pad(mask.astype(np.int8), 1)
    sz, sy, sx = spacing_zyx
    face_area = {0: sy * sx, 1: sz * sx, 2: sz * sy}
    return float(sum(np.abs(np.diff(m, axis=ax)).sum() * face_area[ax] for ax in range(3)))


def shape_features(mask: np.ndarray, spacing_zyx) -> dict[str, float]:
    sp = np.asarray(spacing_zyx, dtype=float)
    n = int(mask.sum())
    V = n * float(np.prod(sp))
    A = surface_area_faces(mask, sp)
    pts = np.argwhere(mask) * sp
    if n >= 3:
        ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
        ev = np.clip(ev, 0, None)
        elong = math.sqrt(ev[1] / ev[0]) if ev[0] > 0 else 0.0
        flat = math.sqrt(ev[2] / ev[0]) if ev[0] > 0 else 0.0
    else:
        elong = flat = 0.0
    # max diameter: farthest pair among surface voxels, reduced to convex-hull vertices
    surface = mask & ~ndimage.binary_erosion(mask)
    sp_pts = np.argwhere(surface) * sp
    try:
        sp_pts = sp_pts[ConvexHull(sp_pts).vertices] if len(sp_pts) > 4 else sp_pts
    except QhullError:  # flat/degenerate objects
        pass
    dmax = float(pdist(sp_pts).max()) if len(sp_pts) > 1 else 0.0
    return {
        "voxel_count": n,
        "volume_mm3": V,
        "surface_area_mm2": A,
        "surface_volume_ratio": A / V if V else 0.0,
        "sphericity": (math.pi ** (1 / 3)) * (6 * V) ** (2 / 3) / A if A else 0.0,
        "elongation": elong,
        "flatness": flat,
        "max_diameter_mm": dmax,
    }


# ----------------------------------------------------------------------------- first order
def first_order_features(x: np.ndarray, bins: int = 64) -> dict[str, float]:
    x = x.astype(np.float64)
    p10, p25, p50, p75, p90 = np.percentile(x, [10, 25, 50, 75, 90])
    hist, _ = np.histogram(x, bins=bins)
    p = hist[hist > 0] / hist.sum()
    return {
        "mean": x.mean(), "median": p50, "min": x.min(), "max": x.max(), "range": np.ptp(x),
        "p10": p10, "p90": p90, "iqr": p75 - p25, "std": x.std(), "variance": x.var(),
        "skewness": float(stats.skew(x)) if x.std() > 0 else 0.0,
        "kurtosis": float(stats.kurtosis(x, fisher=False)) if x.std() > 0 else 0.0,
        "energy": float((x ** 2).sum()), "entropy": float(-(p * np.log2(p)).sum()),
        "mad": float(np.abs(x - x.mean()).mean()), "rms": float(np.sqrt((x ** 2).mean())),
    }


# ----------------------------------------------------------------------------- GLCM
def directions_13() -> list[tuple[int, int, int]]:
    """The 13 unique 3D neighbour offsets (the other 13 are their negatives)."""
    out = []
    for d in itertools.product((-1, 0, 1), repeat=3):
        if d == (0, 0, 0):
            continue
        if tuple(-v for v in d) not in out:
            out.append(d)
    return out


def quantize(values: np.ndarray, levels: int) -> np.ndarray:
    lo, hi = values.min(), values.max()
    if hi <= lo:
        return np.zeros_like(values, dtype=np.int64)
    q = np.floor((values - lo) / (hi - lo) * levels).astype(np.int64)
    return np.clip(q, 0, levels - 1)


def glcm_matrix(q: np.ndarray, mask: np.ndarray, offset, levels: int) -> np.ndarray:
    """Symmetric normalized GLCM for one offset; q holds levels inside mask (-1 elsewhere)."""
    dz, dy, dx = offset
    Z, Y, X = q.shape

    def rng(d, n):
        return slice(max(0, -d), n - max(0, d)), slice(max(0, d), n - max(0, -d))

    (az, bz), (ay, by), (ax, bx) = rng(dz, Z), rng(dy, Y), rng(dx, X)
    a, b = q[az, ay, ax], q[bz, by, bx]
    ok = mask[az, ay, ax] & mask[bz, by, bx]
    i, j = a[ok], b[ok]
    m = np.bincount(i * levels + j, minlength=levels * levels).reshape(levels, levels).astype(float)
    m = m + m.T
    s = m.sum()
    return m / s if s else m


def glcm_features(image: np.ndarray, mask: np.ndarray, levels: int = 32) -> dict[str, float]:
    q = np.full(image.shape, -1, dtype=np.int64)
    q[mask] = quantize(image[mask].astype(float), levels)
    I, J = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    feats = {k: [] for k in GLCM}
    for off in directions_13():
        P = glcm_matrix(q, mask, off, levels)
        if P.sum() == 0:
            continue
        mu_i, mu_j = (I * P).sum(), (J * P).sum()
        sd_i = math.sqrt(((I - mu_i) ** 2 * P).sum())
        sd_j = math.sqrt(((J - mu_j) ** 2 * P).sum())
        nz = P[P > 0]
        feats["glcm_contrast"].append(((I - J) ** 2 * P).sum())
        feats["glcm_dissimilarity"].append((np.abs(I - J) * P).sum())
        feats["glcm_homogeneity"].append((P / (1 + np.abs(I - J))).sum())
        feats["glcm_asm"].append((P ** 2).sum())
        feats["glcm_entropy"].append(-(nz * np.log2(nz)).sum())
        feats["glcm_correlation"].append(
            ((I - mu_i) * (J - mu_j) * P).sum() / (sd_i * sd_j) if sd_i * sd_j > 0 else 1.0)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in feats.items()}


def extract(image: np.ndarray, mask: np.ndarray, spacing_zyx, levels: int = 32) -> dict[str, float]:
    mask = mask.astype(bool)
    if not mask.any():
        raise ValueError("empty mask")
    # crop to the mask's bounding box (+1) for speed
    sl = ndimage.find_objects(mask.astype(np.uint8))[0]
    sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
    img, m = image[sl], mask[sl]
    out = {f"shape_{k}": v for k, v in shape_features(m, spacing_zyx).items()}
    out.update({f"fo_{k}": v for k, v in first_order_features(img[m]).items()})
    out.update(glcm_features(img, m, levels))
    return {k: float(out[k]) for k in FEATURES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--image-suffix", default="", help='e.g. "_0000" for nnU-Net raw folders')
    ap.add_argument("--levels", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("radiomics/radiomics_features.csv"))
    a = ap.parse_args()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", *FEATURES])
        for cid in list_cases(a.labels):
            lab, spacing, _ = read_array(a.labels / f"{cid}.nii.gz")
            img, _, _ = read_array(a.images / f"{cid}{a.image_suffix}.nii.gz")
            assert img.shape == lab.shape, (cid, img.shape, lab.shape)
            tumor = lab == MSD_TUMOR
            if not tumor.any():
                print(f"{cid}: no tumor, skipped")
                continue
            feats = extract(img, tumor, spacing, a.levels)
            w.writerow([cid, *(f"{feats[k]:.6g}" for k in FEATURES)])
            print(cid, f"V={feats['shape_volume_mm3']:.0f}mm3", f"H={feats['glcm_entropy']:.2f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
