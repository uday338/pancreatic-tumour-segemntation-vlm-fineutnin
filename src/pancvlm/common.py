"""Shared constants and helpers.

Axis convention: everything in this package uses SimpleITK, so NumPy arrays are (z, y, x)
and spacing tuples are converted to (z, y, x) as well. Never mix with NiBabel arrays,
which are (x, y, z), without transposing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:  # SimpleITK is needed for any I/O, but pure-array helpers work without it.
    import SimpleITK as sitk
except ImportError:  # pragma: no cover
    sitk = None

# MSD Task07 label map
MSD_BACKGROUND, MSD_PANCREAS, MSD_TUMOR = 0, 1, 2

# AMOS22 label map (15 organs + background)
AMOS_LABELS: dict[int, str] = {
    1: "spleen",
    2: "right kidney",
    3: "left kidney",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior vena cava",
    10: "pancreas",
    11: "right adrenal gland",
    12: "left adrenal gland",
    13: "duodenum",
    14: "bladder",
    15: "prostate/uterus",
}
AMOS_PANCREAS = 10

# Abdominal soft-tissue window used for the VQA PNGs
WINDOW_LOW, WINDOW_HIGH = -100.0, 200.0


def read_image(path: str | os.PathLike) -> "sitk.Image":
    if sitk is None:
        raise ImportError("SimpleITK is required for image I/O")
    return sitk.ReadImage(str(path))


def read_array(path: str | os.PathLike) -> tuple[np.ndarray, tuple[float, float, float], "sitk.Image"]:
    """Return (array[z,y,x], spacing[z,y,x] in mm, the sitk image for geometry)."""
    img = read_image(path)
    arr = sitk.GetArrayFromImage(img)
    spacing_zyx = tuple(float(s) for s in reversed(img.GetSpacing()))
    return arr, spacing_zyx, img


def write_like(arr: np.ndarray, reference: "sitk.Image", path: str | os.PathLike) -> None:
    """Write an array (z,y,x) with the geometry of `reference`."""
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(reference)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(path), useCompression=True)


def window_to_uint8(hu: np.ndarray, low: float = WINDOW_LOW, high: float = WINDOW_HIGH) -> np.ndarray:
    """pixel = 255 * (clip(HU, low, high) - low) / (high - low)."""
    x = (np.clip(hu.astype(np.float32), low, high) - low) / (high - low)
    return np.round(255.0 * x).astype(np.uint8)


def case_id_from_filename(name: str, ending: str = ".nii.gz") -> str:
    if not name.endswith(ending):
        raise ValueError(f"{name} does not end with {ending}")
    return name[: -len(ending)]


def list_cases(folder: str | os.PathLike, ending: str = ".nii.gz") -> list[str]:
    """Case ids in a folder, skipping macOS '._' resource-fork files that ship with MSD."""
    return sorted(
        case_id_from_filename(p.name, ending)
        for p in Path(folder).iterdir()
        if p.name.endswith(ending) and not p.name.startswith(".")
    )


def save_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | os.PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice = 2|P∩G| / (|P|+|G|). Returns NaN when both are empty (nnU-Net convention)."""
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(p, g).sum() / denom)
