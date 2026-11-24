import math

import numpy as np

from pancvlm.common import dice, window_to_uint8
from pancvlm.cv_summary import paired_stats
from pancvlm.questions import (QUESTIONS, answers_for, canonicalize, format_organs, is_correct,
                               organ_set, qtype_buggy, qtype_from_question, size_bucket)


def test_dice():
    a = np.zeros(10, bool); a[:4] = True
    b = np.zeros(10, bool); b[2:6] = True
    assert dice(a, b) == 0.5
    assert math.isnan(dice(np.zeros(3), np.zeros(3)))


def test_window():
    out = window_to_uint8(np.array([-1000, -100, 50, 200, 3000]))
    assert out.tolist() == [0, 0, 128, 255, 255]


def test_qtype_fix_and_bug():
    assert qtype_from_question(QUESTIONS["count"]) == "count"
    assert qtype_buggy(QUESTIONS["count"]) == "presence"  # the original bug
    for qt, q in QUESTIONS.items():
        assert qtype_from_question(q) == qt


def test_paired_stats_matches_guide():
    base = [0.4437, 0.5400, 0.5120, 0.5097, 0.5093]
    mask = [0.5013, 0.5072, 0.5396, 0.4893, 0.4957]
    s = paired_stats(base, mask)
    assert abs(s["mean_diff"] - 0.004) < 0.001
    assert abs(s["sd_diff"] - 0.038) < 0.002
    assert s["wins"] == 2 and s["losses"] == 3 and s["ci_includes_zero"]
    roi = [0.4530, 0.4538, 0.5063, 0.4734, 0.4686]
    assert paired_stats(base, roi)["losses"] == 4


def test_answers():
    fact = {"touching_organs": ["stomach", "duodenum"], "tumor_bbox_diagonal_mm": 40, "n_components": 1}
    a = answers_for(True, fact, (30, 50))
    assert a == {"presence": "Yes", "anatomy": "Stomach, duodenum", "size": "Medium", "count": "One"}
    assert answers_for(False, fact, (30, 50))["count"] == "None"
    assert size_bucket(10, (30, 50)) == "Small" and size_bucket(60, (30, 50)) == "Large"
    assert format_organs([]) == "None"


def test_scoring():
    assert is_correct("yes.", "Yes", "presence")
    assert is_correct("duodenum, stomach", "Stomach, duodenum", "anatomy")
    assert not is_correct("duodenum", "Stomach, duodenum", "anatomy")
    assert is_correct("Not applicable", "Not applicable", "anatomy")
    assert not is_correct("None", "Not applicable", "anatomy")
    assert organ_set("the IVC and liver") == {"inferior vena cava", "liver"}
    assert canonicalize("Yes, there is a tumor.", "presence") == "yes"
    assert canonicalize("There is a single lesion", "count") == "one"
    assert is_correct("The tumor appears large.", "Large", "size", lenient=True)
    assert not is_correct("The tumor appears large.", "Large", "size", lenient=False)
