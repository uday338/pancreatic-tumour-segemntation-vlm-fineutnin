"""Diagnostics for the VLM pipeline (guide sections 14.2, 16.3 and 16.4).

1. Image path check: for a 512x512 PNG expect 504x504 after smart_resize,
   image_grid_thw = [1, 36, 36], pixel_values 1296 x 1176, and 324 <|image_pad|> tokens.
2. Label masking check: prints which tokens carry loss (should be only the answer + <|im_end|>).
3. Loss check: computes model(**batch).loss yourself on a few training batches, so you can
   compare it against the loss the Trainer logged (some transformers versions log the sum over
   gradient-accumulation steps, e.g. 7.06 / 8 = 0.88 per token).

    python -m pancvlm.diagnose_vlm --data vqa_data [--adapter vlm_out/lora_adapter] [--n-batches 8]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .train_vlm import DEFAULT_MODEL, load_model_and_processor
from .vlm_data import IGNORE_INDEX, VQADataset, make_collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("vqa_data"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--no-model", action="store_true", help="only check processor outputs")
    a = ap.parse_args()

    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(a.model)
    ds = VQADataset(a.data / "train.jsonl", processor)
    rec = ds.records[0]
    item = ds[0]
    w, h = Image.open(rec["image"]).size
    thw = item["image_grid_thw"][0].tolist()
    pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_pad = int((item["input_ids"] == pad_id).sum())
    print(f"image {w}x{h} -> processed {thw[1] * 14}x{thw[2] * 14}")
    print(f"image_grid_thw = {thw}; pixel_values {tuple(item['pixel_values'].shape)}; image-pad tokens = {n_pad}")
    print(f"expected image-pad tokens = t*h*w/4 = {thw[0] * thw[1] * thw[2] // 4}")
    kept = item["labels"] != IGNORE_INDEX
    print("loss tokens:", repr(processor.tokenizer.decode(item["input_ids"][kept])))
    print(f"loss on {int(kept.sum())} of {len(kept)} tokens")
    if a.no_model:
        return

    model, _ = load_model_and_processor(a.model)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.cuda().eval()
    print("GPU:", torch.cuda.get_device_name(0))
    collate = make_collate(processor.tokenizer.pad_token_id)
    losses = []
    with torch.no_grad():
        for i in range(min(a.n_batches, len(ds))):
            batch = {k: v.cuda() for k, v in collate([ds[i]]).items()}
            losses.append(model(**batch).loss.item())
    m = sum(losses) / len(losses)
    print(f"mean per-token loss over {len(losses)} examples: {m:.4f}  "
          f"(p(correct token) ~ {math.exp(-m):.3f})")


if __name__ == "__main__":
    main()
