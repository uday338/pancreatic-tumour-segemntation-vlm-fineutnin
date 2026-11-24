#!/bin/bash
# End-to-end order of operations. Heavy steps are PBS jobs; run the rest on a compute node.
# Each step can be run on its own; read the comments before running blindly.
set -euo pipefail
source configs/paths.env

## 1. nnU-Net raw datasets (matched experiments + organ model)
python -m pancvlm.nnunet_data baseline --msd "$MSD" --id "$D_BASE"
python -m pancvlm.nnunet_data mask     --msd "$MSD" --id "$D_MASK"          # mask channel is noNorm
python -m pancvlm.nnunet_data roi      --msd "$MSD" --id "$D_ROI" --margin-mm 20
python -m pancvlm.nnunet_data amos     --amos "$AMOS" --id "$D_AMOS"

## 2. plan + preprocess
for d in $D_BASE $D_MASK $D_ROI $D_AMOS; do
  nnUNetv2_plan_and_preprocess -d "$d" --verify_dataset_integrity -c 3d_fullres
done

## 3. identical folds for all three experiments.
# splits_final.json is written when fold 0 of the baseline starts training. Either train fold 0 of the
# baseline first, or create it now with a short run, then copy it before training the others:
#   python -m pancvlm.nnunet_data splits --src $D_BASE --dst $D_MASK $D_ROI
# Then: diff the three splits_final.json files.

## 4. train 5 folds x 3 experiments (+ organ model), one PBS job per fold
#   for d in $D_BASE $D_MASK $D_ROI; do for f in 0 1 2 3 4; do qsub -v DATASET=$d,FOLD=$f jobs/nnunet_train.pbs; done; done
#   qsub -v DATASET=$D_AMOS,FOLD=0 jobs/nnunet_train.pbs   # (+ folds 1-4 for an ensemble)

## 5. paired comparison
python -m pancvlm.cv_summary --baseline "$D_BASE" --compare "$D_MASK" "$D_ROI" --out results/cv_summary.json

## 6. organs on MSD, facts, radiomics
#   qsub jobs/nnunet_predict_organs.pbs
python -m pancvlm.facts --labels "$MSD/labelsTr" --organs preds/amos_on_msd --contact-mm 3 --out results/facts.json
python -m pancvlm.radiomics --images "$MSD/imagesTr" --labels "$MSD/labelsTr" --out radiomics/radiomics_features.csv
python -m pancvlm.radiomics_analysis --features radiomics/radiomics_features.csv --k 3 --out radiomics/analysis

## 7. VQA dataset (test patients = baseline fold-0 validation patients)
python -m pancvlm.vqa --images "$MSD/imagesTr" --labels "$MSD/labelsTr" --facts results/facts.json \
  --out vqa_data --test-from-splits "$nnUNet_preprocessed"/Dataset00${D_BASE}_*/splits_final.json --fold 0

## 8. VLM: train + evaluate
#   qsub jobs/vlm_train.pbs && qsub jobs/vlm_eval.pbs
