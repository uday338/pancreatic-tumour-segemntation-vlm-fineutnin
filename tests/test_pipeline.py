import json

import numpy as np
import pytest

from conftest import SPACING_ZYX, ball
from pancvlm import facts, radiomics
from pancvlm.eval_vlm import majority_baseline, score


def test_components():
    m = np.zeros((10, 10, 10), bool)
    m[1:3, 1:3, 1:3] = True
    m[6:9, 6:9, 6:9] = True
    m[5, 5, 5] = True  # touches the second block by a corner -> 26-connected
    assert facts.connected_components(m)[0] == 2
    assert facts.connected_components(m, min_voxels=10)[0] == 1


def test_case_facts(synthetic_case):
    ct, label, organs = synthetic_case
    f = facts.case_facts(label == 2, SPACING_ZYX, organs, contact_mm=3.0, min_component_voxels=0)
    assert f["n_components"] == 1
    assert f["touching_organs"] == ["duodenum"]  # liver is far; pancreas excluded
    assert f["tumor_volume_mm3"] == pytest.approx(f["tumor_voxels"] * np.prod(SPACING_ZYX))
    assert f["tumor_bbox_diagonal_mm"] > 0
    with pytest.raises(ValueError):
        facts.touching_organs(label == 2, organs.transpose(2, 1, 0), SPACING_ZYX, 3.0)


def test_radiomics_sphere():
    shape = (40, 40, 40)
    m = ball(shape, (20, 20, 20), (12, 12, 12))
    img = np.random.default_rng(0).normal(40, 10, shape)
    f = radiomics.extract(img, m, (1.0, 1.0, 1.0))
    assert len(f) == 30
    assert f["shape_volume_mm3"] == pytest.approx(4 / 3 * np.pi * 12 ** 3, rel=0.05)
    assert 0.6 < f["shape_sphericity"] <= 1.0  # face-count area overestimates -> below 1
    assert f["shape_elongation"] == pytest.approx(1.0, abs=0.05)
    assert f["shape_max_diameter_mm"] == pytest.approx(24, abs=1.5)
    assert f["fo_mean"] == pytest.approx(40, abs=1)
    assert 0 <= f["glcm_homogeneity"] <= 1 and 0 < f["glcm_asm"] <= 1


def test_glcm_constant_region():
    m = np.ones((5, 5, 5), bool)
    f = radiomics.glcm_features(np.full((5, 5, 5), 7.0), m)
    assert f["glcm_contrast"] == 0 and f["glcm_asm"] == pytest.approx(1.0)


def test_end_to_end_nifti(tmp_path, synthetic_case, monkeypatch):
    sitk = pytest.importorskip("SimpleITK")
    from pancvlm import nnunet_data, vqa

    ct, label, organs = synthetic_case
    msd = tmp_path / "msd"
    for sub in ("imagesTr", "labelsTr"):
        (msd / sub).mkdir(parents=True)
    org_dir = tmp_path / "organs"; org_dir.mkdir()
    for i in range(1, 7):
        shift = i - 1  # vary tumor position a little
        lab = np.roll(label, shift, axis=2)
        for arr, path in ((ct if i % 2 else np.roll(ct, shift, 2), msd / "imagesTr" / f"pancreas_{i:03d}.nii.gz"),
                          (lab, msd / "labelsTr" / f"pancreas_{i:03d}.nii.gz"),
                          (np.roll(organs, shift, 2), org_dir / f"pancreas_{i:03d}.nii.gz")):
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing(tuple(reversed(SPACING_ZYX)))
            sitk.WriteImage(img, str(path))

    # nnU-Net raw datasets
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
    d3 = nnunet_data.build_mask_dataset(msd, 3)
    dj = json.loads((d3 / "dataset.json").read_text())
    assert dj["channel_names"] == {"0": "CT", "1": "noNorm"}
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(d3 / "imagesTr" / "pancreas_001_0001.nii.gz")))
    assert set(np.unique(mask)) == {0, 1} and mask[label == 2].all()  # tumor inside mask, no hole
    d6 = nnunet_data.build_roi_dataset(msd, 6, margin_mm=5)
    cropped = sitk.ReadImage(str(d6 / "labelsTr" / "pancreas_001.nii.gz"))
    assert sitk.GetArrayFromImage(cropped).shape[2] < label.shape[2]
    assert (sitk.GetArrayFromImage(cropped) == 2).sum() == (label == 2).sum()

    # facts -> VQA
    fx = {}
    for i in range(1, 7):
        cid = f"pancreas_{i:03d}"
        lab = sitk.GetArrayFromImage(sitk.ReadImage(str(msd / "labelsTr" / f"{cid}.nii.gz")))
        org = sitk.GetArrayFromImage(sitk.ReadImage(str(org_dir / f"{cid}.nii.gz")))
        fx[cid] = facts.case_facts(lab == 2, SPACING_ZYX, org, 3.0, 0)
    (tmp_path / "facts.json").write_text(json.dumps({"meta": {}, "cases": fx}))
    out = tmp_path / "vqa"; out.mkdir()
    meta = vqa.build(msd / "imagesTr", msd / "labelsTr", tmp_path / "facts.json", out,
                     n_val=1, n_test=2, png_size=64)
    assert meta["counts"]["test"]["patients"] == 2 and meta["counts"]["train"]["patients"] == 3
    train = [json.loads(l) for l in open(out / "train.jsonl")]
    test = [json.loads(l) for l in open(out / "test.jsonl")]
    assert len(train) == 3 * 2 * 4
    assert not ({r["case_id"] for r in train} & {r["case_id"] for r in test})  # patient-level split
    assert {r["qtype"] for r in train} == {"presence", "anatomy", "size", "count"}
    pos = [r for r in train if r["positive"] and r["qtype"] == "anatomy"]
    assert all(r["answer"] == "Duodenum" for r in pos)

    # majority baseline scoring runs and negatives are all correct
    rep = score(majority_baseline(train, test))
    assert rep["by_image"]["negative"]["accuracy"] == 1.0
