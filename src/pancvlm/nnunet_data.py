"""Build nnU-Net v2 raw datasets for the three matched experiments and the AMOS organ model.

    baseline  MSD Task07 as-is: 1 channel (CT), labels 0/1/2
    mask      Dataset003_AnatPancreas: 2 channels [CT, binary pancreas mask], mask channel = "noNorm"
    roi       Dataset006_ROIcrop: CT + labels cropped to the pancreas bounding box plus a margin
    amos      AMOS22 CT scans with 15 organ labels
    splits    copy splits_final.json from the baseline to the other experiments and verify case ids

Usage:
    python -m pancvlm.nnunet_data baseline --msd /data/Task07_Pancreas --id 1
    python -m pancvlm.nnunet_data mask     --msd /data/Task07_Pancreas --id 3
    python -m pancvlm.nnunet_data roi      --msd /data/Task07_Pancreas --id 6 --margin-mm 20
    python -m pancvlm.nnunet_data amos     --amos /data/amos22 --id 10
    python -m pancvlm.nnunet_data splits   --src 1 --dst 3 6
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

from .common import (
    AMOS_LABELS,
    MSD_PANCREAS,
    list_cases,
    load_json,
    read_array,
    read_image,
    save_json,
    sitk,
    write_like,
)

MSD_LABELS = {"background": 0, "pancreas": 1, "tumor": 2}


def _env_dir(var: str) -> Path:
    val = os.environ.get(var)
    if not val:
        raise SystemExit(f"Environment variable {var} is not set (see configs/paths.env.example)")
    return Path(val)


def dataset_dir(dataset_id: int, name: str) -> Path:
    return _env_dir("nnUNet_raw") / f"Dataset{dataset_id:03d}_{name}"


def find_dataset_folder(root: Path, dataset_id: int) -> Path:
    matches = sorted(root.glob(f"Dataset{dataset_id:03d}_*"))
    if len(matches) != 1:
        raise SystemExit(f"Expected one Dataset{dataset_id:03d}_* in {root}, found {matches}")
    return matches[0]


def write_dataset_json(out: Path, channel_names: dict[str, str], labels: dict[str, int], n: int) -> None:
    save_json(
        {
            "channel_names": channel_names,
            "labels": labels,
            "numTraining": n,
            "file_ending": ".nii.gz",
        },
        out / "dataset.json",
    )


def _msd_cases(msd: Path) -> list[str]:
    return list_cases(msd / "labelsTr")


# ----------------------------------------------------------------------------- baseline
def build_baseline(msd: Path, dataset_id: int, name: str = "PancreasBaseline") -> Path:
    out = dataset_dir(dataset_id, name)
    (out / "imagesTr").mkdir(parents=True, exist_ok=True)
    (out / "labelsTr").mkdir(parents=True, exist_ok=True)
    cases = _msd_cases(msd)
    for cid in cases:
        shutil.copy(msd / "imagesTr" / f"{cid}.nii.gz", out / "imagesTr" / f"{cid}_0000.nii.gz")
        shutil.copy(msd / "labelsTr" / f"{cid}.nii.gz", out / "labelsTr" / f"{cid}.nii.gz")
    write_dataset_json(out, {"0": "CT"}, MSD_LABELS, len(cases))
    print(f"baseline: {len(cases)} cases -> {out}")
    return out


# ----------------------------------------------------------------------------- mask channel
def pancreas_mask_from_label(label: np.ndarray) -> np.ndarray:
    """Binary pancreas mask INCLUDING the tumor (label >= 1).

    Using label == 1 alone would leave a tumor-shaped hole in the mask, which hands the
    network the answer. Even label >= 1 is "oracle" information that does not exist for a
    new patient; use --mask-dir with predicted masks for a deployable setting.
    """
    return (label >= MSD_PANCREAS).astype(np.uint8)


def build_mask_dataset(
    msd: Path,
    dataset_id: int,
    name: str = "AnatPancreas",
    mask_dir: Path | None = None,
    mask_channel_name: str = "noNorm",
) -> Path:
    """Two-channel dataset. The mask channel MUST be "noNorm": any other name makes nnU-Net
    z-score the binary mask per image, so 'pancreas' maps to (1-f)/sqrt(f(1-f)), which depends
    on the pancreas fraction f of each scan (the bug described in guide section 7.2)."""
    if mask_channel_name != "noNorm":
        print(f"WARNING: mask channel named {mask_channel_name!r} will be z-scored by nnU-Net")
    out = dataset_dir(dataset_id, name)
    (out / "imagesTr").mkdir(parents=True, exist_ok=True)
    (out / "labelsTr").mkdir(parents=True, exist_ok=True)
    cases = _msd_cases(msd)
    for cid in cases:
        ct_path = msd / "imagesTr" / f"{cid}.nii.gz"
        lab, _, lab_img = read_array(msd / "labelsTr" / f"{cid}.nii.gz")
        if mask_dir is not None:
            pred, _, _ = read_array(mask_dir / f"{cid}.nii.gz")
            if pred.shape != lab.shape:
                raise ValueError(f"{cid}: predicted mask {pred.shape} != label {lab.shape}")
            mask = (pred >= 1).astype(np.uint8)  # works for a pancreas+tumor prediction map
        else:
            mask = pancreas_mask_from_label(lab)
        shutil.copy(ct_path, out / "imagesTr" / f"{cid}_0000.nii.gz")
        write_like(mask, lab_img, out / "imagesTr" / f"{cid}_0001.nii.gz")
        shutil.copy(msd / "labelsTr" / f"{cid}.nii.gz", out / "labelsTr" / f"{cid}.nii.gz")
    write_dataset_json(out, {"0": "CT", "1": mask_channel_name}, MSD_LABELS, len(cases))
    src = "predicted" if mask_dir else "ground-truth (oracle)"
    print(f"mask: {len(cases)} cases, {src} pancreas masks -> {out}")
    return out


# ----------------------------------------------------------------------------- ROI crop
def roi_bounds(mask: np.ndarray, spacing_zyx, margin_mm: float) -> tuple[slice, slice, slice]:
    """Bounding box of a binary mask (z,y,x) expanded by margin_mm on every side."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask, cannot crop")
    lo, hi = coords.min(0), coords.max(0) + 1
    pad = np.ceil(margin_mm / np.asarray(spacing_zyx)).astype(int)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]


def crop_image(img: "sitk.Image", bounds_zyx) -> "sitk.Image":
    """Crop with sitk so origin/direction stay correct (sitk index order is x,y,z)."""
    z, y, x = bounds_zyx
    return sitk.RegionOfInterest(
        img,
        size=[x.stop - x.start, y.stop - y.start, z.stop - z.start],
        index=[x.start, y.start, z.start],
    )


def build_roi_dataset(
    msd: Path,
    dataset_id: int,
    name: str = "ROIcrop",
    margin_mm: float = 20.0,
    mask_dir: Path | None = None,
) -> Path:
    """Keeps the original case names so splits_final.json from the baseline still applies."""
    out = dataset_dir(dataset_id, name)
    (out / "imagesTr").mkdir(parents=True, exist_ok=True)
    (out / "labelsTr").mkdir(parents=True, exist_ok=True)
    cases = _msd_cases(msd)
    log = {}
    for cid in cases:
        ct_img = read_image(msd / "imagesTr" / f"{cid}.nii.gz")
        lab, spacing, lab_img = read_array(msd / "labelsTr" / f"{cid}.nii.gz")
        if mask_dir is not None:
            ref, _, _ = read_array(mask_dir / f"{cid}.nii.gz")
            ref = ref >= 1
        else:
            ref = lab >= MSD_PANCREAS
        b = roi_bounds(ref, spacing, margin_mm)
        tumor_kept = int((lab[b] == 2).sum()) / max(int((lab == 2).sum()), 1)
        sitk.WriteImage(crop_image(ct_img, b), str(out / "imagesTr" / f"{cid}_0000.nii.gz"), True)
        sitk.WriteImage(crop_image(lab_img, b), str(out / "labelsTr" / f"{cid}.nii.gz"), True)
        log[cid] = {"bounds_zyx": [[s.start, s.stop] for s in b], "tumor_fraction_kept": tumor_kept}
        if tumor_kept < 1.0:
            print(f"WARNING {cid}: crop removed {100 * (1 - tumor_kept):.1f}% of tumor voxels")
    write_dataset_json(out, {"0": "CT"}, MSD_LABELS, len(cases))
    save_json({"margin_mm": margin_mm, "mask_source": str(mask_dir or "ground truth"), "cases": log},
              out / "roi_crop_log.json")
    print(f"roi: {len(cases)} cases, margin {margin_mm} mm -> {out}")
    return out


# ----------------------------------------------------------------------------- AMOS
def build_amos_dataset(amos: Path, dataset_id: int, name: str = "AMOSorgans", ct_only: bool = True) -> Path:
    """AMOS22 ids 0001-0500 are CT, 0501+ are MRI. Takes every labelled CT case found in
    labelsTr (and labelsVa if present)."""
    out = dataset_dir(dataset_id, name)
    (out / "imagesTr").mkdir(parents=True, exist_ok=True)
    (out / "labelsTr").mkdir(parents=True, exist_ok=True)
    n = 0
    for img_sub, lab_sub in (("imagesTr", "labelsTr"), ("imagesVa", "labelsVa")):
        if not (amos / lab_sub).exists():
            continue
        for cid in list_cases(amos / lab_sub):
            num = int(cid.split("_")[-1])
            if ct_only and num > 500:
                continue
            shutil.copy(amos / img_sub / f"{cid}.nii.gz", out / "imagesTr" / f"{cid}_0000.nii.gz")
            shutil.copy(amos / lab_sub / f"{cid}.nii.gz", out / "labelsTr" / f"{cid}.nii.gz")
            n += 1
    labels = {"background": 0, **{v: k for k, v in AMOS_LABELS.items()}}
    write_dataset_json(out, {"0": "CT"}, labels, n)
    print(f"amos: {n} CT cases -> {out}")
    return out


# ----------------------------------------------------------------------------- splits
def split_case_ids(splits: list[dict]) -> set[str]:
    return {c for fold in splits for c in fold["train"] + fold["val"]}


def copy_splits(src_id: int, dst_ids: list[int]) -> None:
    """Reuse the baseline's fold assignment so every experiment is a paired comparison.
    Run AFTER nnUNetv2_plan_and_preprocess for all datasets and BEFORE training."""
    pre = _env_dir("nnUNet_preprocessed")
    src_file = find_dataset_folder(pre, src_id) / "splits_final.json"
    if not src_file.exists():
        raise SystemExit(f"{src_file} missing: run nnUNetv2_train for any fold of {src_id} once, "
                         "or nnUNetv2_plan_and_preprocess, to create it")
    splits = load_json(src_file)
    src_cases = split_case_ids(splits)
    for d in dst_ids:
        dst_dir = find_dataset_folder(pre, d)
        raw_cases = set(list_cases(find_dataset_folder(_env_dir("nnUNet_raw"), d) / "labelsTr"))
        if raw_cases != src_cases:
            missing, extra = src_cases - raw_cases, raw_cases - src_cases
            raise SystemExit(f"Dataset {d}: case ids differ from source (missing {sorted(missing)[:5]}, "
                             f"extra {sorted(extra)[:5]}). nnU-Net would make fresh random splits.")
        target = dst_dir / "splits_final.json"
        if target.exists() and load_json(target) == splits:
            print(f"Dataset {d}: splits already identical")
            continue
        shutil.copy(src_file, target)
        print(f"Dataset {d}: copied {src_file} -> {target}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("baseline"); p.add_argument("--msd", type=Path, required=True)
    p.add_argument("--id", type=int, default=1); p.add_argument("--name", default="PancreasBaseline")

    p = sub.add_parser("mask"); p.add_argument("--msd", type=Path, required=True)
    p.add_argument("--id", type=int, default=3); p.add_argument("--name", default="AnatPancreas")
    p.add_argument("--mask-dir", type=Path, help="predicted pancreas masks (default: ground truth)")
    p.add_argument("--mask-channel-name", default="noNorm")

    p = sub.add_parser("roi"); p.add_argument("--msd", type=Path, required=True)
    p.add_argument("--id", type=int, default=6); p.add_argument("--name", default="ROIcrop")
    p.add_argument("--margin-mm", type=float, default=20.0)
    p.add_argument("--mask-dir", type=Path, help="predicted pancreas masks (default: ground truth)")

    p = sub.add_parser("amos"); p.add_argument("--amos", type=Path, required=True)
    p.add_argument("--id", type=int, default=10); p.add_argument("--name", default="AMOSorgans")

    p = sub.add_parser("splits"); p.add_argument("--src", type=int, required=True)
    p.add_argument("--dst", type=int, nargs="+", required=True)

    a = ap.parse_args()
    if a.cmd == "baseline":
        build_baseline(a.msd, a.id, a.name)
    elif a.cmd == "mask":
        build_mask_dataset(a.msd, a.id, a.name, a.mask_dir, a.mask_channel_name)
    elif a.cmd == "roi":
        build_roi_dataset(a.msd, a.id, a.name, a.margin_mm, a.mask_dir)
    elif a.cmd == "amos":
        build_amos_dataset(a.amos, a.id, a.name)
    elif a.cmd == "splits":
        copy_splits(a.src, a.dst)


if __name__ == "__main__":
    main()
