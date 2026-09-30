from pathlib import Path

from musicinstruct.metrics import score_item
from musicinstruct.schema import BenchmarkItem, Plan
from musicinstruct.transforms import make_seed_midi, transpose


def _item(source: Path, gold: Path, result) -> BenchmarkItem:
    return BenchmarkItem(
        item_id="t1",
        composition_id="seed_00",
        gold_mode="unique",
        op_family="transpose",
        instruction="Transpose up 2 semitones.",
        midi_in=str(source),
        gold_midi=str(gold),
        plan=result.plan,
        must_change=result.must_change,
        must_preserve=result.must_preserve,
        edit_mask=result.edit_mask,
        split="test",
    )


def test_gold_scores_perfectly(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    gold = tmp_path / "gold.mid"
    result = transpose(source, gold, semitones=2)
    item = _item(source, gold, result)
    scores = score_item(item, str(gold), item.plan)
    assert scores.validity == 1.0
    assert scores.edit_success >= 0.99
    assert scores.preserve >= 0.99
    assert scores.joint == 1.0
    assert scores.plan_match == 1.0


def test_empty_must_change_scores_zero(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    gold = tmp_path / "gold.mid"
    result = transpose(source, gold, semitones=2)
    item = _item(source, gold, result)
    item = item.model_copy(update={"must_change": []})
    scores = score_item(item, str(gold), item.plan)
    assert scores.edit_success == 0.0
    assert scores.joint == 0.0


def test_wrong_plan_scores_zero(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    gold = tmp_path / "gold.mid"
    result = transpose(source, gold, semitones=2)
    item = _item(source, gold, result)
    scores = score_item(item, str(gold), Plan(op="transpose", params={"semitones": 5}))
    assert scores.plan_match == 0.0
