from pathlib import Path

from musicinstruct.predicates import evaluate_predicates
from musicinstruct.schema import EditMask, Predicate
from musicinstruct.transforms import make_seed_midi, transpose


def test_pitch_shifted_predicate_passes(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    gold = tmp_path / "gold.mid"
    result = transpose(source, gold, semitones=5)
    preds = evaluate_predicates(result.must_change, str(source), str(gold), result.edit_mask)
    assert preds[0].passed
    assert preds[0].score >= 0.9


def test_preserve_outside_mask_after_transpose(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    gold = tmp_path / "gold.mid"
    result = transpose(source, gold, semitones=2)
    preds = evaluate_predicates(result.must_preserve, str(source), str(gold), result.edit_mask)
    assert all(p.passed for p in preds)


def test_track_muted_fails_when_track_is_loud(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    preds = evaluate_predicates(
        [Predicate(name="track_muted", params={"tracks": [0], "max_velocity": 1})],
        str(source),
        str(source),
        EditMask(),
    )
    assert not preds[0].passed
