"""LoRA supervised fine-tuning of Qwen2.5-VL-3B-Instruct on the pancreatic VQA dataset.

v1 (defaults):  r=16, alpha=32, dropout 0.05, lr 1e-4, 3 epochs, batch 1 x grad-accum 8, bf16
v2:             --lora-r 32 --lora-alpha 64 --lr 5e-4 --epochs 5 --output-dir vlm_out_v2

    python -m pancvlm.train_vlm --data vqa_data --output-dir vlm_out

target_modules q_proj/k_proj/v_proj/o_proj only match the language model's attention (the ViT
uses a fused `qkv` and `proj`), giving 36 layers x 204,800 = 7,372,800 trainable parameters at
r=16 (0.196%). Use --lora-vision to also adapt the vision encoder, or --lora-mlp for the LLM MLPs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .vlm_data import VQADataset, make_collate

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_model_and_processor(model_name: str, min_pixels: int | None = None, max_pixels: int | None = None):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    kw = {}
    if min_pixels:
        kw["min_pixels"] = min_pixels
    if max_pixels:
        kw["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_name, **kw)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    return model, processor


def lora_targets(vision: bool, mlp: bool) -> list[str] | str:
    if not vision:
        mods = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if mlp:
            mods += ["gate_proj", "up_proj", "down_proj"]
        return mods
    # regex over full module names: LLM attention (+MLP) and the ViT's fused qkv/proj
    llm = "q_proj|k_proj|v_proj|o_proj" + ("|gate_proj|up_proj|down_proj" if mlp else "")
    return rf".*(model\.layers\.\d+\.(self_attn|mlp)\.({llm})|visual\.blocks\.\d+\.attn\.(qkv|proj))$"


def count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("vqa_data"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", type=Path, default=Path("vlm_out"))
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-vision", action="store_true")
    ap.add_argument("--lora-mlp", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-to", default="none")
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(a.seed)
    model, processor = load_model_and_processor(a.model, max_pixels=a.max_pixels)
    model.config.use_cache = False
    cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                     target_modules=lora_targets(a.lora_vision, a.lora_mlp), bias="none",
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    if a.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable, total = count_params(model)
    print(f"trainable params: {trainable:,} / {total:,} = {100 * trainable / total:.3f}%")

    train_ds = VQADataset(a.data / "train.jsonl", processor)
    val_path = a.data / "val.jsonl"
    val_ds = VQADataset(val_path, processor) if val_path.exists() else None
    collate = make_collate(processor.tokenizer.pad_token_id)

    args = TrainingArguments(
        output_dir=str(a.output_dir),
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        num_train_epochs=a.epochs,
        warmup_ratio=a.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,   # keep pixel_values / image_grid_thw
        dataloader_num_workers=2,
        report_to=a.report_to,
        seed=a.seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collate)
    trainer.train()

    adapter_dir = a.output_dir / "lora_adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    with open(a.output_dir / "train_config.json", "w") as f:
        json.dump({**{k: str(v) for k, v in vars(a).items()},
                   "trainable_params": trainable, "total_params": total,
                   "log_history": trainer.state.log_history}, f, indent=2)
    print(f"adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
