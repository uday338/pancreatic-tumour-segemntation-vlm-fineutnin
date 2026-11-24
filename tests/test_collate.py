import pytest

torch = pytest.importorskip("torch")
from pancvlm.vlm_data import IGNORE_INDEX, build_messages, make_collate  # noqa: E402


def test_collate_concats_vision_and_pads_text():
    b1 = {"input_ids": torch.arange(5), "attention_mask": torch.ones(5, dtype=torch.long),
          "labels": torch.tensor([-100, -100, -100, 3, 4]),
          "pixel_values": torch.zeros(1296, 1176), "image_grid_thw": torch.tensor([[1, 36, 36]])}
    b2 = {"input_ids": torch.arange(3), "attention_mask": torch.ones(3, dtype=torch.long),
          "labels": torch.tensor([-100, 1, 2]),
          "pixel_values": torch.zeros(64, 1176), "image_grid_thw": torch.tensor([[1, 8, 8]])}
    out = make_collate(pad_token_id=0)([b1, b2])
    assert out["input_ids"].shape == (2, 5)
    assert out["labels"][1, 3:].tolist() == [IGNORE_INDEX, IGNORE_INDEX]
    assert out["attention_mask"][1].tolist() == [1, 1, 1, 0, 0]
    assert out["pixel_values"].shape == (1296 + 64, 1176)
    assert out["image_grid_thw"].tolist() == [[1, 36, 36], [1, 8, 8]]


def test_messages_require_absolute_image_path():
    with pytest.raises(ValueError):
        build_messages({"image": "rel.png", "question": "q", "answer": "a"})
    m = build_messages({"image": "/abs/x.png", "question": "q", "answer": "a"})
    assert m[0]["content"][0] == {"type": "image", "image": "/abs/x.png"}
