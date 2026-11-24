"""pancvlm: pancreatic tumor CT analysis, from nnU-Net segmentation to a LoRA-tuned Qwen2.5-VL.

Stages (see README.md):
    nnunet_data        build nnU-Net v2 raw datasets (baseline, +mask channel, ROI crop, AMOS organs)
    cv_summary         paired 5-fold comparison of tumor Dice across experiments
    facts              per-patient facts from tumor + predicted organ masks (size, count, adjacency)
    radiomics          self-contained 30-feature extractor (shape, first-order, GLCM)
    radiomics_analysis standardize, PCA, k-means, correlations
    vqa                2D slices + automatically generated question/answer pairs, patient-level split
    train_vlm          LoRA supervised fine-tuning of Qwen2.5-VL-3B
    eval_vlm           generation + scoring by stored qtype, pos/neg split, baselines
"""

__version__ = "0.1.0"
