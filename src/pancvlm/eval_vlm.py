"""Evaluate the VLM on the VQA test set, with the qtype fix and the baselines from guide 17.3.

    # fine-tuned model
    python -m pancvlm.eval_vlm --data vqa_data --adapter vlm_out/lora_adapter --out results/eval_v1
    # zero-shot base model (no LoRA), same test set
    python -m pancvlm.eval_vlm --data vqa_data --out results/eval_zeroshot --zero-shot-hint
    # majority-class baseline only (no GPU)
    python -m pancvlm.eval_vlm --data vqa_data --baseline-only --out results/eval_majority
    # re-score an existing predictions file
    python -m pancvlm.eval_vlm --rescore results/eval_v1/predictions.jsonl --out results/eval_v1

Reported:
  overall accuracy (strict exact match after normalization; organ lists compared as unordered sets)
  per-qtype accuracy using the qtype stored in each record (not parsed from the question)
  accuracy split by positive vs negative images, and qtype x positive/negative
  "lenient" accuracy, which maps free text onto the label space (for zero-shot outputs)
  mean organ Jaccard for anatomy questions (partial-credit view)
  what the old buggy qtype() grouping would have reported, for comparison
  majority-class baseline: most frequent train answer per (qtype, true presence)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import save_json
from .questions import is_correct, organ_jaccard, qtype_buggy

ZERO_SHOT_HINTS = {
    "presence": " Answer Yes or No.",
    "anatomy": " Answer with a comma-separated list of organs, or None.",
    "size": " Answer Small, Medium or Large.",
    "count": " Answer None, One or Multiple.",
}


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _acc(rows: list[dict], key: str = "correct") -> dict:
    n = len(rows)
    k = sum(r[key] for r in rows)
    return {"correct": k, "n": n, "accuracy": round(k / n, 4) if n else None}


def score(preds: list[dict]) -> dict:
    for r in preds:
        r["correct"] = is_correct(r["prediction"], r["answer"], r["qtype"])
        r["correct_lenient"] = is_correct(r["prediction"], r["answer"], r["qtype"], lenient=True)
    rep: dict = {"overall": _acc(preds), "overall_lenient": _acc(preds, "correct_lenient")}
    by_q, by_pn, by_qpn, by_buggy = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for r in preds:
        pn = "positive" if r["positive"] else "negative"
        by_q[r["qtype"]].append(r)
        by_pn[pn].append(r)
        by_qpn[f"{r['qtype']}/{pn}"].append(r)
        by_buggy[qtype_buggy(r["question"])].append(r)
    rep["by_qtype"] = {k: _acc(v) for k, v in sorted(by_q.items())}
    rep["by_qtype_lenient"] = {k: _acc(v, "correct_lenient") for k, v in sorted(by_q.items())}
    rep["by_image"] = {k: _acc(v) for k, v in sorted(by_pn.items())}
    rep["by_qtype_and_image"] = {k: _acc(v) for k, v in sorted(by_qpn.items())}
    rep["by_qtype_old_buggy_grouping"] = {k: _acc(v) for k, v in sorted(by_buggy.items())}
    anat = [r for r in preds if r["qtype"] == "anatomy" and r["positive"]]
    if anat:
        rep["anatomy_mean_jaccard_positive"] = round(
            sum(organ_jaccard(r["prediction"], r["answer"]) for r in anat) / len(anat), 4)
    return rep


def majority_baseline(train: list[dict], test: list[dict]) -> list[dict]:
    """Given the TRUE presence of each test image, answer the most frequent train answer."""
    counts: dict = defaultdict(Counter)
    for r in train:
        counts[(r["qtype"], r["positive"])][r["answer"]] += 1
    out = []
    for r in test:
        c = counts.get((r["qtype"], r["positive"]))
        out.append({**r, "prediction": c.most_common(1)[0][0] if c else ""})
    return out


def generate(test: list[dict], model_name: str, adapter: Path | None, max_new_tokens: int,
             zero_shot_hint: bool) -> list[dict]:
    import torch
    from qwen_vl_utils import process_vision_info

    from .train_vlm import load_model_and_processor
    from .vlm_data import build_messages

    model, processor = load_model_and_processor(model_name)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.cuda().eval()
    out = []
    for i, r in enumerate(test):
        hint = ZERO_SHOT_HINTS[r["qtype"]] if zero_shot_hint else ""
        msgs = build_messages(r, with_answer=False, prompt_suffix=hint)
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, _ = process_vision_info(msgs)
        enc = processor(text=[text], images=images, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        pred = processor.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        out.append({**r, "prediction": pred})
        if i % 50 == 0:
            print(f"[{i}/{len(test)}] {r['qtype']}: ref={r['answer']!r} pred={pred!r}")
    return out


def write(preds: list[dict], rep: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.jsonl", "w") as f:
        for r in preds:
            f.write(json.dumps(r) + "\n")
    save_json(rep, out / "metrics.json")
    o = rep["overall"]
    print(f"overall: {o['correct']}/{o['n']} = {100 * o['accuracy']:.1f}%")
    for k, v in rep["by_qtype"].items():
        print(f"  {k:>9}: {v['correct']}/{v['n']} = {100 * v['accuracy']:.1f}%")
    for k, v in rep["by_image"].items():
        print(f"  {k:>9} images: {100 * v['accuracy']:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("vqa_data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", type=Path, help="LoRA adapter dir; omit for zero-shot")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--zero-shot-hint", action="store_true", help="append answer-format hints to questions")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--rescore", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    if a.rescore:
        preds = load_jsonl(a.rescore)
    else:
        test = load_jsonl(a.data / f"{a.split}.jsonl")
        if a.baseline_only:
            preds = majority_baseline(load_jsonl(a.data / "train.jsonl"), test)
        else:
            preds = generate(test, a.model, a.adapter, a.max_new_tokens, a.zero_shot_hint)
    rep = score(preds)
    if not a.rescore and not a.baseline_only:
        base = majority_baseline(load_jsonl(a.data / "train.jsonl"), load_jsonl(a.data / f"{a.split}.jsonl"))
        rep["majority_baseline"] = score(base)["overall"]
    rep["setting"] = {"adapter": str(a.adapter), "zero_shot_hint": a.zero_shot_hint,
                      "baseline_only": a.baseline_only, "split": a.split}
    write(preds, rep, a.out)


if __name__ == "__main__":
    main()
