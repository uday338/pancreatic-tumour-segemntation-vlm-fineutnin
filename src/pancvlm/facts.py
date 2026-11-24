"""Per-patient facts from the tumor mask and (predicted) organ masks.

Tumor source: MSD ground-truth labels (default) or nnU-Net predictions (--tumor-dir).
Organ source: AMOS-model predictions on the MSD scans (nnUNetv2_predict -d <amos id>), which
are model outputs, not ground truth, so adjacency facts are noisy.

    python -m pancvlm.facts --labels /data/Task07_Pancreas/labelsTr \
        --organs preds/amos_on_msd --out results/facts.json --contact-mm 3

Fields per case:
    tumor_voxels, tumor_volume_mm3, centroid_zyx (voxel), centroid_mm (zyx, from array origin),
    bbox_zyx, bbox_extent_mm, tumor_bbox_diagonal_mm (size proxy, NOT RECIST),
    n_components, component_voxels, tumor_slices (axial indices with tumor),
    tumor_in_pred_pancreas_fraction, touching_organs, pancreas_region (null: never computed)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

from .common import AMOS_LABELS, AMOS_PANCREAS, MSD_TUMOR, list_cases, read_array, save_json

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)


def connected_components(mask: np.ndarray, min_voxels: int = 0) -> tuple[int, list[int]]:
    """Number of 26-connected components, ignoring those smaller than min_voxels."""
    lab, n = ndimage.label(mask, structure=CONNECTIVITY_26)
    if n == 0:
        return 0, []
    sizes = np.bincount(lab.ravel())[1:]
    sizes = sorted((int(s) for s in sizes if s >= min_voxels), reverse=True)
    return len(sizes), sizes


def bbox_diagonal_mm(mask: np.ndarray, spacing_zyx) -> tuple[float, list[list[int]], list[float]]:
    coords = np.argwhere(mask)
    lo, hi = coords.min(0), coords.max(0)
    extent = (hi - lo + 1) * np.asarray(spacing_zyx)
    return float(np.sqrt((extent ** 2).sum())), [[int(a), int(b)] for a, b in zip(lo, hi)], extent.tolist()


def dilate_mm(mask: np.ndarray, spacing_zyx, radius_mm: float) -> tuple[np.ndarray, tuple[slice, ...]]:
    """Physically isotropic dilation via a distance transform, computed on a crop around the mask.
    Returns (dilated crop, crop slices)."""
    coords = np.argwhere(mask)
    pad = np.ceil(radius_mm / np.asarray(spacing_zyx)).astype(int) + 1
    lo = np.maximum(coords.min(0) - pad, 0)
    hi = np.minimum(coords.max(0) + pad + 1, mask.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    crop = mask[sl]
    dist = ndimage.distance_transform_edt(~crop, sampling=spacing_zyx)
    return dist <= radius_mm, sl


def touching_organs(tumor: np.ndarray, organ_map: np.ndarray, spacing_zyx, radius_mm: float,
                    organs: dict[int, str] = AMOS_LABELS, exclude=(AMOS_PANCREAS,)) -> list[str]:
    if tumor.shape != organ_map.shape:
        raise ValueError(f"tumor {tumor.shape} and organ map {organ_map.shape} differ: axis-order bug?")
    dil, sl = dilate_mm(tumor, spacing_zyx, radius_mm)
    om = organ_map[sl]
    present = set(np.unique(om[dil]).tolist())
    return [name for lbl, name in organs.items() if lbl in present and lbl not in exclude]


def case_facts(tumor: np.ndarray, spacing_zyx, organ_map: np.ndarray | None, contact_mm: float,
               min_component_voxels: int) -> dict:
    tumor = tumor.astype(bool)
    vox_vol = float(np.prod(spacing_zyx))
    f: dict = {"spacing_zyx": list(spacing_zyx), "tumor_voxels": int(tumor.sum()), "pancreas_region": None}
    f["tumor_volume_mm3"] = f["tumor_voxels"] * vox_vol
    if f["tumor_voxels"] == 0:
        f.update(n_components=0, component_voxels=[], touching_organs=[], tumor_slices=[],
                 tumor_bbox_diagonal_mm=0.0, bbox_zyx=None, bbox_extent_mm=None, centroid_zyx=None)
        return f
    c = np.argwhere(tumor).mean(0)
    f["centroid_zyx"] = c.round(2).tolist()
    f["centroid_mm"] = (c * np.asarray(spacing_zyx)).round(2).tolist()
    f["tumor_bbox_diagonal_mm"], f["bbox_zyx"], f["bbox_extent_mm"] = bbox_diagonal_mm(tumor, spacing_zyx)
    f["n_components"], f["component_voxels"] = connected_components(tumor, min_component_voxels)
    f["tumor_slices"] = np.flatnonzero(tumor.any(axis=(1, 2))).tolist()
    if organ_map is not None:
        f["touching_organs"] = touching_organs(tumor, organ_map, spacing_zyx, contact_mm)
        pan = organ_map == AMOS_PANCREAS
        f["tumor_in_pred_pancreas_fraction"] = float((tumor & pan).sum() / tumor.sum())
    else:
        f["touching_organs"] = None
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True, help="MSD labelsTr (tumor = label 2)")
    ap.add_argument("--tumor-dir", type=Path, help="use nnU-Net tumor predictions instead of labels")
    ap.add_argument("--organs", type=Path, help="AMOS-model organ predictions on the MSD scans")
    ap.add_argument("--contact-mm", type=float, default=3.0, help="dilation radius for contact (mm)")
    ap.add_argument("--min-component-voxels", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("results/facts.json"))
    a = ap.parse_args()

    src = a.tumor_dir or a.labels
    facts = {}
    for cid in list_cases(a.labels):
        lab, spacing, _ = read_array(src / f"{cid}.nii.gz")
        organ_map = None
        if a.organs is not None:
            organ_map, osp, _ = read_array(a.organs / f"{cid}.nii.gz")
            if not np.allclose(osp, spacing, atol=1e-3):
                raise ValueError(f"{cid}: organ spacing {osp} != label spacing {spacing}")
        facts[cid] = case_facts(lab == MSD_TUMOR, spacing, organ_map, a.contact_mm, a.min_component_voxels)
        print(cid, facts[cid]["tumor_volume_mm3"], facts[cid]["n_components"], facts[cid]["touching_organs"])
    meta = {"tumor_source": str(src), "organ_source": str(a.organs), "contact_mm": a.contact_mm,
            "min_component_voxels": a.min_component_voxels, "connectivity": 26}
    save_json({"meta": meta, "cases": facts}, a.out)
    print(f"{len(facts)} cases -> {a.out}")


if __name__ == "__main__":
    main()
