"""Build the VQA dataset: 2 axial slices per patient (tumor-positive, tumor-negative), windowed
to [-100, 200] HU, saved as 8-bit grayscale PNGs WITHOUT any tumor outline, plus 4 questions
per image whose answers come from facts.json. The split is by patient.

    python -m pancvlm.vqa --images /data/Task07_Pancreas/imagesTr \
        --labels /data/Task07_Pancreas/labelsTr --facts results/facts.json \
        --out vqa_data --test-from-splits $nnUNet_preprocessed/Dataset001_PancreasBaseline/splits_final.json

Slice rules (recorded in vqa_data/meta.json):
    positive: the axial slice with the largest tumor cross-section ("max_area", default) or the
              slice through the tumor centroid ("centroid")
    negative: a tumor-free slice that contains pancreas ("pancreas", default; this keeps the
              negative anatomically similar so presence cannot be answered from body region)
              or any tumor-free slice nearest the tumor ("any")
Size buckets: tertiles of tumor_bbox_diagonal_mm over TRAIN patients, unless --size-mm is given.

Outputs: vqa_data/images/*.png, vqa_data/{train,val,test}.jsonl, vqa_data/meta.json
Each JSONL record: {id, case_id, split, image, slice_index, positive, qtype, question, answer}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .common import MSD_PANCREAS, MSD_TUMOR, load_json, read_array, save_json, window_to_uint8
from .questions import QUESTIONS, answers_for


def pick_positive(tumor: np.ndarray, rule: str = "max_area") -> int:
    areas = tumor.sum(axis=(1, 2))
    if areas.max() == 0:
        raise ValueError("no tumor voxels")
    if rule == "max_area":
        return int(areas.argmax())
    if rule == "centroid":
        z = int(round(np.argwhere(tumor)[:, 0].mean()))
        if areas[z] == 0:  # centroid can fall in a gap for multi-component tumors
            z = int(np.flatnonzero(areas)[np.abs(np.flatnonzero(areas) - z).argmin()])
        return z
    raise ValueError(rule)


def pick_negative(label: np.ndarray, rule: str = "pancreas") -> int | None:
    tumor_z = label.__eq__(MSD_TUMOR).any(axis=(1, 2))
    pan_area = (label == MSD_PANCREAS).sum(axis=(1, 2))
    free = np.flatnonzero(~tumor_z)
    if free.size == 0:
        return None
    if rule == "pancreas":
        cand = free[pan_area[free] > 0]
        if cand.size:
            return int(cand[pan_area[cand].argmax()])
        rule = "any"  # fall back
    if rule == "any":
        tz = np.flatnonzero(tumor_z)
        dist = np.min(np.abs(free[:, None] - tz[None, :]), axis=1)
        return int(free[dist.argmin()])
    raise ValueError(rule)


def save_png(arr2d: np.ndarray, path: Path, size: int | None = None) -> None:
    from PIL import Image

    im = Image.fromarray(window_to_uint8(arr2d), mode="L")
    if size and im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def patient_split(cases: list[str], n_val: int, n_test: int, seed: int,
                  test_cases: list[str] | None = None) -> dict[str, str]:
    rng = random.Random(seed)
    cases = sorted(cases)
    if test_cases is not None:
        test = sorted(set(test_cases) & set(cases))
        rest = [c for c in cases if c not in set(test)]
    else:
        shuffled = cases[:]
        rng.shuffle(shuffled)
        test, rest = sorted(shuffled[:n_test]), shuffled[n_test:]
    rest = sorted(rest)
    rng.shuffle(rest)
    val = set(rest[:n_val])
    return {c: ("test" if c in test else "val" if c in val else "train") for c in cases}


def tertiles(values: list[float]) -> tuple[float, float]:
    a, b = np.percentile(values, [100 / 3, 200 / 3])
    return float(a), float(b)


def build(images: Path, labels: Path, facts_path: Path, out: Path, image_suffix: str = "",
          pos_rule: str = "max_area", neg_rule: str = "pancreas", n_val: int = 22, n_test: int = 57,
          seed: int = 42, test_cases: list[str] | None = None,
          size_mm: tuple[float, float] | None = None, png_size: int | None = 512) -> dict:
    facts = load_json(facts_path)["cases"]
    cases = sorted(c for c, f in facts.items() if f["tumor_voxels"] > 0)
    split = patient_split(cases, n_val, n_test, seed, test_cases)
    size_source = "fixed (--size-mm)" if size_mm is not None else "tertiles of train patients"
    if size_mm is None:
        size_mm = tertiles([facts[c]["tumor_bbox_diagonal_mm"] for c in cases if split[c] == "train"])

    records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    img_dir = out / "images"
    for cid in cases:
        lab, _, _ = read_array(labels / f"{cid}.nii.gz")
        ct, _, _ = read_array(images / f"{cid}{image_suffix}.nii.gz")
        assert ct.shape == lab.shape, (cid, ct.shape, lab.shape)
        zpos = pick_positive(lab == MSD_TUMOR, pos_rule)
        zneg = pick_negative(lab, neg_rule)
        for positive, z in ((True, zpos), (False, zneg)):
            if z is None:
                print(f"{cid}: no tumor-free slice, skipping negative")
                continue
            tag = "pos" if positive else "neg"
            png = img_dir / f"{cid}_{tag}.png"
            save_png(ct[z], png, png_size)
            ans = answers_for(positive, facts[cid], size_mm)
            for qt, q in QUESTIONS.items():
                records[split[cid]].append({
                    "id": f"{cid}_{tag}_{qt}", "case_id": cid, "split": split[cid],
                    "image": str(png.resolve()), "slice_index": int(z), "positive": positive,
                    "qtype": qt, "question": q, "answer": ans[qt],
                })
    for s, recs in records.items():
        with open(out / f"{s}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    meta = {
        "positive_rule": pos_rule, "negative_rule": neg_rule, "window_hu": [-100, 200],
        "png_size": png_size, "size_thresholds_mm": list(size_mm),
        "size_threshold_source": size_source,
        "test_source": "nnU-Net fold validation cases" if test_cases is not None else "random",
        "seed": seed, "facts": str(facts_path),
        "patients": {s: sorted({r["case_id"] for r in recs}) for s, recs in records.items()},
        "counts": {s: {"images": len({r["image"] for r in recs}), "qa": len(recs),
                       "patients": len({r["case_id"] for r in recs})} for s, recs in records.items()},
    }
    save_json(meta, out / "meta.json")
    for s, c in meta["counts"].items():
        print(f"{s:>5}: {c['patients']} patients, {c['images']} images, {c['qa']} QA pairs")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--facts", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("vqa_data"))
    ap.add_argument("--image-suffix", default="")
    ap.add_argument("--positive-rule", choices=["max_area", "centroid"], default="max_area")
    ap.add_argument("--negative-rule", choices=["pancreas", "any"], default="pancreas")
    ap.add_argument("--n-val", type=int, default=22)
    ap.add_argument("--n-test", type=int, default=57)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-from-splits", type=Path,
                    help="nnU-Net splits_final.json: use one fold's validation cases as the VQA test set")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--size-mm", type=float, nargs=2, help="fixed Small/Medium and Medium/Large cut-offs")
    ap.add_argument("--png-size", type=int, default=512)
    a = ap.parse_args()

    test_cases = None
    if a.test_from_splits:
        test_cases = load_json(a.test_from_splits)[a.fold]["val"]
    a.out.mkdir(parents=True, exist_ok=True)
    build(a.images, a.labels, a.facts, a.out, a.image_suffix, a.positive_rule, a.negative_rule,
          a.n_val, a.n_test, a.seed, test_cases, tuple(a.size_mm) if a.size_mm else None, a.png_size)


if __name__ == "__main__":
    main()
