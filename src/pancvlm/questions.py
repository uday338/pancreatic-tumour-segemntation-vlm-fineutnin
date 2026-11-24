"""Question templates, answer generation and scoring. Pure Python (no torch), so it is
shared by the dataset builder, the trainer and the evaluator, and easy to unit-test.

The question type is stored as a field ("qtype") in every record when the dataset is built.
Evaluation reads that field instead of parsing the question text, which is what caused the
original qtype() bug: "How many distinct tumors are present?" matched "present" first and was
counted as a presence question.
"""

from __future__ import annotations

import re
import string

from .common import AMOS_LABELS, AMOS_PANCREAS

QUESTIONS: dict[str, str] = {
    "presence": "Is a pancreatic tumor present in this image?",
    "anatomy": "Which organs are adjacent to the tumor?",
    "size": "What is the approximate size of the tumor?",
    "count": "How many distinct tumors are present?",
}
QTYPES = list(QUESTIONS)
NOT_APPLICABLE = "Not applicable"
ORGAN_ORDER = [n for k, n in AMOS_LABELS.items() if k != AMOS_PANCREAS]


def qtype_from_question(q: str) -> str:
    """Fallback for records without a stored qtype. Most specific test first."""
    q = q.lower()
    if "how many" in q or "distinct" in q:
        return "count"
    if "present" in q:
        return "presence"
    if "adjacent" in q:
        return "anatomy"
    if "size" in q:
        return "size"
    raise ValueError(f"unknown question type: {q!r}")


def qtype_buggy(q: str) -> str:
    """The original, buggy ordering. Kept only to quantify the bug's effect in reports."""
    q = q.lower()
    if "present" in q:
        return "presence"
    if "adjacent" in q:
        return "anatomy"
    if "size" in q:
        return "size"
    if "how many" in q or "distinct" in q:
        return "count"
    raise ValueError(q)


def size_bucket(diag_mm: float, thresholds: tuple[float, float]) -> str:
    t1, t2 = thresholds
    return "Small" if diag_mm < t1 else ("Medium" if diag_mm < t2 else "Large")


def format_organs(organs: list[str]) -> str:
    if not organs:
        return "None"
    ordered = sorted(organs, key=lambda o: ORGAN_ORDER.index(o) if o in ORGAN_ORDER else 99)
    s = ", ".join(ordered)
    return s[0].upper() + s[1:]


def answers_for(positive: bool, fact: dict, size_thresholds: tuple[float, float]) -> dict[str, str]:
    if not positive:
        return {"presence": "No", "anatomy": NOT_APPLICABLE, "size": NOT_APPLICABLE, "count": "None"}
    organs = fact.get("touching_organs") or []
    return {
        "presence": "Yes",
        "anatomy": format_organs(organs),
        "size": size_bucket(fact["tumor_bbox_diagonal_mm"], size_thresholds),
        "count": "One" if fact["n_components"] <= 1 else "Multiple",
    }


# ----------------------------------------------------------------------------- scoring
_PUNCT = str.maketrans("", "", string.punctuation.replace("/", ""))


def normalize(s: str) -> str:
    s = s.strip().lower().replace("<|im_end|>", "")
    s = s.translate(_PUNCT)
    return re.sub(r"\s+", " ", s).strip()


def organ_set(s: str) -> frozenset[str]:
    n = normalize(s)
    if n in ("none", "no organs", ""):
        return frozenset()
    if n == normalize(NOT_APPLICABLE):
        return frozenset({"__na__"})
    aliases = {"ivc": "inferior vena cava", "vena cava": "inferior vena cava",
               "right adrenal": "right adrenal gland", "left adrenal": "left adrenal gland"}
    found = set()
    for name in ORGAN_ORDER:
        if name in n:
            found.add(name)
    for alias, name in aliases.items():
        if alias in n:
            found.add(name)
    return frozenset(found)


def canonicalize(pred: str, qtype: str) -> str:
    """Map free text (e.g. a zero-shot answer) onto the label space. Used for 'lenient' scoring."""
    n = normalize(pred)
    if "not applicable" in n or n in ("na", "n/a"):
        return normalize(NOT_APPLICABLE)
    if qtype == "presence":
        if n.startswith("yes"):
            return "yes"
        if n.startswith("no"):
            return "no"
        if re.search(r"\b(no|not|absent)\b", n):
            return "no"
        if re.search(r"\b(yes|present|visible)\b", n):
            return "yes"
    elif qtype == "size":
        for w in ("small", "medium", "large"):
            if w in n:
                return w
    elif qtype == "count":
        if re.search(r"\b(none|no|zero|0)\b", n):
            return "none"
        if re.search(r"\b(multiple|several|two|three|many|[2-9])\b", n):
            return "multiple"
        if re.search(r"\b(one|single|a|1)\b", n):
            return "one"
    return n


def is_correct(pred: str, ref: str, qtype: str, lenient: bool = False) -> bool:
    if qtype == "anatomy":
        return organ_set(pred) == organ_set(ref)  # unordered set comparison, no partial credit
    if lenient:
        return canonicalize(pred, qtype) == canonicalize(ref, qtype)
    return normalize(pred) == normalize(ref)


def organ_jaccard(pred: str, ref: str) -> float:
    a, b = organ_set(pred), organ_set(ref)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
