"""Dataset and collator for Qwen2.5-VL supervised fine-tuning.

Two fixes from the original run are built in (guide chapter 16):
  1. Every image message carries its absolute path: {"type": "image", "image": "/abs/path.png"}.
     A bare {"type": "image"} makes the chat template emit <|image_pad|> tokens while
     process_vision_info loads nothing, and the vision tower crashes with
     "shape '[0, 4, -1]' is invalid for input of size 1280".
  2. The collator pads TEXT tensors (input_ids, attention_mask, labels) but CONCATENATES
     vision tensors along dim 0: pixel_values is (total patches across images) x 1176, and
     image_grid_thw has one [t, h, w] row per image (like a CSR buffer + index array).

Label masking: only the assistant answer tokens (e.g. "Yes<|im_end|>") contribute to the loss.
System/user/image-pad/padding positions are set to -100.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_messages(rec: dict, with_answer: bool = True, prompt_suffix: str = "") -> list[dict]:
    image = rec["image"]
    if not Path(image).is_absolute():
        raise ValueError(f"image path must be absolute: {image}")
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": rec["question"] + prompt_suffix},
    ]}]
    if with_answer:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": rec["answer"]}]})
    return msgs


class VQADataset(torch.utils.data.Dataset):
    def __init__(self, jsonl: str | Path, processor, prompt_suffix: str = ""):
        self.records = load_jsonl(jsonl)
        self.processor = processor
        self.prompt_suffix = prompt_suffix

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        from qwen_vl_utils import process_vision_info

        rec = self.records[i]
        full = build_messages(rec, True, self.prompt_suffix)
        prompt = build_messages(rec, False, self.prompt_suffix)
        full_text = self.processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        images, _ = process_vision_info(full)
        if not images:
            raise RuntimeError(f"no image loaded for {rec['id']}")
        enc = self.processor(text=[full_text], images=images, return_tensors="pt")
        penc = self.processor(text=[prompt_text], images=images, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        n_prompt = penc["input_ids"].shape[1]
        if not torch.equal(input_ids[:n_prompt], penc["input_ids"][0]):
            raise RuntimeError("prompt tokens are not a prefix of the full sequence; check the chat template")
        labels = input_ids.clone()
        labels[:n_prompt] = IGNORE_INDEX
        # Drop the trailing newline after <|im_end|> from the loss if the template adds one
        tail = self.processor.tokenizer("\n", add_special_tokens=False)["input_ids"]
        if len(tail) == 1 and input_ids[-1].item() == tail[0]:
            labels[-1] = IGNORE_INDEX
        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"][0],
            "labels": labels,
            "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"],
        }


def make_collate(pad_token_id: int):
    def collate(batch: list[dict]) -> dict:
        return {
            "input_ids": pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id),
            "attention_mask": pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
            "labels": pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=IGNORE_INDEX),
            "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),      # concat, not pad
            "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch], dim=0),  # one row per image
        }
    return collate
