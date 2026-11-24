# Pancreatic tumour CT: from nnU-Net segmentation to a fine-tuned VLM

Code for a two-part study on pancreatic tumours in contrast CT (MSD Task07, 281 cases).

1. **Segmentation with anatomical priors.** Does explicit anatomical context help a 3D nnU-Net v2
   find pancreatic tumours? Three matched 5-fold experiments: CT only (baseline), CT plus a
   binary pancreas-mask channel, and CT cropped to a pancreas ROI.
2. **Vision-language.** Segmentation-derived facts (presence, size, count, adjacent organs from
   an AMOS22 organ model) become an automatically generated VQA dataset. Qwen2.5-VL-3B is then
   fine-tuned on it with LoRA.

Radiomics (30 hand-crafted features, PCA, k-means) characterises the tumours alongside.

```
MSD Task07 ──► nnU-Net v2 (baseline / +mask / ROI crop) ──► cv_summary (paired 5-fold Dice)
AMOS22 ─────► nnU-Net organ model ──► organ masks on MSD ─┐
MSD labels ───────────────────────────────────────────────┼─► facts.json ─► VQA (562 PNGs × 4 Qs)
                                       radiomics (30 feats)┘                    │
                                                                Qwen2.5-VL-3B + LoRA ─► eval
```

## Reference results (from the original run)

| | Value |
|---|---|
| Tumour Dice, CT-only baseline (5-fold mean) | 0.503 (folds 0.444 to 0.540) |
| + pancreas-mask channel | 0.507 (+0.004, 2 wins / 3 losses, 95% CI includes 0) |
| ROI crop | 0.471 (−0.032, lost 4 of 5 folds) |
| Pancreas Dice, baseline | 0.818 |
| AMOS organ model, foreground mean Dice (fold 0) | 0.875 |
| VQA: train / val / test patients | 202 / 22 / 57 (404 / 44 / 114 images) |
| LoRA trainable parameters (r = 16) | 7,372,800 of 3,761,995,776 (0.196%) |
| VLM overall accuracy (test) | 72.4% (330 / 456) |

The per-category VLM numbers from the original run (for example 90.8% "presence") were affected
by the `qtype()` bug described below. `eval_vlm` recomputes them correctly.

## Layout

```
src/pancvlm/
  common.py              I/O (SimpleITK, arrays are z,y,x), HU windowing, Dice, AMOS label map
  nnunet_data.py         build nnU-Net raw datasets: baseline, mask (noNorm), ROI crop, AMOS; copy splits
  cv_summary.py          per-fold tumour/pancreas Dice and paired stats vs baseline
  facts.py               per-patient facts: volume, bbox diagonal, components, touching organs
  radiomics.py           30-feature extractor (shape, first-order, 3D GLCM), no PyRadiomics needed
  radiomics_analysis.py  standardise, PCA, k-means, size-vs-texture correlation, median split
  questions.py           question templates, answers, scoring (pure Python, unit-tested)
  vqa.py                 slice selection, PNG export, QA generation, patient-level split
  vlm_data.py            Qwen chat messages, label masking, the fixed collator
  train_vlm.py           LoRA SFT with the HF Trainer
  diagnose_vlm.py        image-token / label-mask / loss-scale diagnostics
  eval_vlm.py            generation, scoring by stored qtype, pos/neg split, baselines
jobs/                    PBS scripts (A100 icelake nodes)
configs/paths.env.example
run_pipeline.sh          the full order of operations
tests/                   unit tests on synthetic volumes (no GPU, no data needed)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
cp configs/paths.env.example configs/paths.env   # edit paths, then:
source configs/paths.env
pytest                                           # 13 tests, CPU only
```

Data: download MSD **Task07_Pancreas** (medicaldecathlon.com) and **AMOS22** (amos22.grand-challenge.org).
The Qwen weights are pulled from Hugging Face (`Qwen/Qwen2.5-VL-3B-Instruct`) on first use; set `HF_HOME`.

## Running it

`run_pipeline.sh` lists every step in order. The main ones:

### 1. Segmentation experiments

```bash
python -m pancvlm.nnunet_data baseline --msd $MSD --id 1
python -m pancvlm.nnunet_data mask     --msd $MSD --id 3      # Dataset003_AnatPancreas
python -m pancvlm.nnunet_data roi      --msd $MSD --id 6 --margin-mm 20   # Dataset006_ROIcrop
for d in 1 3 6; do nnUNetv2_plan_and_preprocess -d $d --verify_dataset_integrity -c 3d_fullres; done
# after splits_final.json exists for dataset 1:
python -m pancvlm.nnunet_data splits --src 1 --dst 3 6
for d in 1 3 6; do for f in 0 1 2 3 4; do qsub -v DATASET=$d,FOLD=$f jobs/nnunet_train.pbs; done; done
python -m pancvlm.cv_summary --baseline 1 --compare 3 6
```

Points that matter for a fair comparison:

* **Matched folds.** All three experiments use the baseline's `splits_final.json`. The `splits`
  command refuses to copy it if the case ids differ, which would make nnU-Net silently create new
  random folds. The ROI crop keeps the original case names for this reason.
* **The mask channel is `"noNorm"`.** With any other channel name nnU-Net z-scores the binary mask
  per image, so "pancreas" becomes (1−f)/√(f(1−f)) and depends on how much of the scan the
  pancreas fills. Fixing this recovered about +0.024 Dice in the original run.
* **The mask includes the tumour** (`label >= 1`). A `label == 1` mask would leave a tumour-shaped
  hole, which gives the answer away. Ground-truth masks are still oracle information; pass
  `--mask-dir` with predicted pancreas masks for a deployable setting.
* `roi_crop_log.json` records the margin, the crop box and the fraction of tumour kept per case.

### 2. Organs, facts and radiomics

```bash
python -m pancvlm.nnunet_data amos --amos $AMOS --id 10
nnUNetv2_plan_and_preprocess -d 10 -c 3d_fullres && qsub -v DATASET=10,FOLD=0 jobs/nnunet_train.pbs
qsub jobs/nnunet_predict_organs.pbs                    # organ masks on the MSD scans
python -m pancvlm.facts --labels $MSD/labelsTr --organs preds/amos_on_msd --contact-mm 3
python -m pancvlm.radiomics --images $MSD/imagesTr --labels $MSD/labelsTr
python -m pancvlm.radiomics_analysis --features radiomics/radiomics_features.csv --k 3
```

* Adjacency: the tumour is dilated by `--contact-mm` (a physical radius via a distance transform,
  so anisotropic slice spacing is handled), then each predicted organ it overlaps is listed,
  excluding the pancreas. Organ masks are predictions under AMOS→MSD domain shift, so these labels
  are noisy.
* Size is `tumor_bbox_diagonal_mm`, a bounding-box proxy and **not** RECIST.
* Tumour count uses 26-connected components, ignoring fragments under `--min-component-voxels`.

### 3. VQA dataset

```bash
python -m pancvlm.vqa --images $MSD/imagesTr --labels $MSD/labelsTr --facts results/facts.json \
    --out vqa_data --test-from-splits $nnUNet_preprocessed/Dataset001_PancreasBaseline/splits_final.json --fold 0
```

Per patient, one tumour-positive slice (largest tumour cross-section) and one tumour-free slice
(by default one that still contains pancreas, so presence cannot be guessed from the body region).
Slices are windowed to [−100, 200] HU and saved as 512×512 grayscale PNGs with **no outline
drawn**. Four questions per image:

| qtype | Question | Positive | Negative |
|---|---|---|---|
| presence | Is a pancreatic tumor present in this image? | Yes | No |
| anatomy | Which organs are adjacent to the tumor? | e.g. "Stomach, duodenum" | Not applicable |
| size | What is the approximate size of the tumor? | Small / Medium / Large | Not applicable |
| count | How many distinct tumors are present? | One / Multiple | None |

Size buckets are tertiles of the training patients' bbox diagonals unless `--size-mm` is given.
The split is by patient, so both slices of a patient land in the same split. `meta.json` records
every rule and threshold used.

### 4. Fine-tune Qwen2.5-VL-3B with LoRA

```bash
qsub jobs/vlm_train.pbs                                   # v1: r16/α32, lr 1e-4, 3 epochs
qsub -v EXTRA="--lora-r 32 --lora-alpha 64 --lr 5e-4 --epochs 5",OUT=vlm_out_v2 jobs/vlm_train.pbs
```

Fixes from the original run are built in:

* Each image message carries its absolute path (`{"type": "image", "image": "/abs/path.png"}`)
  and images are loaded with `process_vision_info`. Without it the vision tower gets zero patches
  and fails with `shape '[0, 4, -1]' is invalid for input of size 1280`.
* The collator pads text tensors but **concatenates** `pixel_values` and `image_grid_thw` along
  dim 0.
* Labels are −100 everywhere except the assistant answer and `<|im_end|>`.

`diagnose_vlm` prints the image path numbers (512×512 → 504×504, `image_grid_thw = [1, 36, 36]`,
`pixel_values` 1296×1176, 324 `<|image_pad|>` tokens), shows which tokens carry loss, and computes
the per-token loss itself. Use it to settle whether the logged loss of about 7.06 was summed over
the 8 accumulation steps (7.06 / 8 ≈ 0.88).

`--lora-vision` also adapts the ViT's `qkv`/`proj`, and `--lora-mlp` adds the LLM MLP projections.
With the defaults only the LLM attention is adapted and the vision encoder stays frozen.

### 5. Evaluate

```bash
python -m pancvlm.eval_vlm --data vqa_data --adapter vlm_out/lora_adapter --out results/eval_v1
python -m pancvlm.eval_vlm --data vqa_data --zero-shot-hint --out results/eval_zeroshot
python -m pancvlm.eval_vlm --data vqa_data --baseline-only --out results/eval_majority
```

* Categories come from the `qtype` stored in each record. The old evaluator matched "present"
  before "how many", so every count question ("How many distinct tumors are present?") was scored
  as presence. `metrics.json` also reports the old grouping so the effect can be shown.
* Organ lists are compared as unordered sets (strict), and a mean Jaccard gives a partial-credit view.
* Accuracy is split by positive and negative images. Negative images have fixed answers, so a
  model that gets presence right already scores well there.
* Baselines: majority class per (qtype, true presence) from the training set, and zero-shot base
  Qwen. A "lenient" score maps free-text answers onto the label space for the zero-shot model.

## Limitations

* 0.503 tumour Dice is a research result, not a clinical tool.
* VQA labels are derived from segmentation and were never checked by a radiologist. Adjacency
  inherits organ-model errors (duodenum Dice 0.796) and domain shift.
* Size and count describe the 3D tumour, but the model sees one 2D slice.
* Five folds give five paired differences, enough to rule out large effects but not to establish
  small ones.
* There is no external validation. The hospital CT has no labels, so it supports only a
  qualitative demo.

## Next steps

Store and report per-category accuracy with the fixed evaluator, add the baselines, resolve the
loss-scale question before tuning v2, improve adjacency labels with a stronger organ and vessel
model (for example TotalSegmentator), add texture and head/body/tail questions, try multi-slice
input or LoRA on the vision encoder, and test on larger tumour datasets (PanTS, PANORAMA).

## Software versions used

Python 3.10.18, PyTorch 2.6.0+cu118, transformers 4.51.3, PEFT 0.20.0, Accelerate 1.14.0,
qwen-vl-utils 0.0.14, nnU-Net v2, NumPy 2.2.5, SciPy 1.15.3, pandas 2.3.2, scikit-learn 1.7.1,
SimpleITK 2.5.5.
