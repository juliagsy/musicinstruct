from pathlib import Path

import pytest

from musicinstruct.evaluation import load_predictions, score_records, write_predictions
from musicinstruct.generate import OutputDirectoryExistsError, generate_pilot_dataset
from musicinstruct.dataset import validate_dataset
from musicinstruct.plan_executor import normalize_plan, validate_plan_for_source
from musicinstruct.predicates import eval_predicate
from musicinstruct.schema import BenchmarkItem, EditMask, Plan, Predicate, Prediction
from musicinstruct.transforms import make_seed_midi


def test_generate_pilot_refuses_existing_output(tmp_path: Path) -> None:
    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=20)
    with pytest.raises(OutputDirectoryExistsError):
        generate_pilot_dataset(out, target=20)


def test_generate_pilot_force_overwrites(tmp_path: Path) -> None:
    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=20)
    items = generate_pilot_dataset(out, target=25, force=True)
    assert len(items) == 25


def test_malformed_predicate_scores_zero(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    result = eval_predicate(
        "pitch_shifted_by",
        {},
        str(source),
        str(source),
        EditMask(),
    )
    assert not result.passed
    assert result.score == 0.0
    assert "missing param" in result.detail


def test_invalid_track_index_scores_zero(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    result = eval_predicate(
        "program_is",
        {"track": 99, "program": 25},
        str(source),
        str(source),
        EditMask(),
    )
    assert not result.passed
    assert result.score == 0.0
    assert "out of range" in result.detail


def test_score_records_reports_missing_predictions(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    item = BenchmarkItem(
        item_id="x1",
        composition_id="seed_00",
        gold_mode="unique",
        op_family="transpose",
        instruction="Transpose up 1 semitone.",
        midi_in=str(source),
        gold_midi=str(source),
        split="test",
    )
    manifest = tmp_path / "gold.jsonl"
    manifest.write_text(item.model_dump_json() + "\n", encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    results = score_records(manifest, empty)
    assert results["missing_prediction_ids"] == ["x1"]
    assert results["prediction_coverage"] == 0.0


def test_score_records_strict_mode_raises(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    item = BenchmarkItem(
        item_id="x1",
        composition_id="seed_00",
        gold_mode="unique",
        op_family="transpose",
        instruction="Transpose up 1 semitone.",
        midi_in=str(source),
        gold_midi=str(source),
        split="test",
    )
    manifest = tmp_path / "gold.jsonl"
    manifest.write_text(item.model_dump_json() + "\n", encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="missing predictions"):
        score_records(manifest, empty, strict=True)


def test_load_predictions_reports_json_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_predictions(path)


def test_write_predictions_uses_relative_paths(tmp_path: Path) -> None:
    midi = tmp_path / "outputs" / "x.mid"
    midi.parent.mkdir(parents=True)
    midi.write_bytes(b"MThd")
    pred_path = tmp_path / "outputs" / "preds.jsonl"
    write_predictions(
        pred_path,
        [Prediction(item_id="x1", midi_path=str(midi.resolve()))],
    )
    line = pred_path.read_text(encoding="utf-8").strip()
    assert str(midi.resolve()) not in line
    assert "x.mid" in line


def test_validate_plan_rejects_out_of_range_track(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    plan = Plan(op="mute_tracks", params={"tracks": [99]})
    with pytest.raises(ValueError, match="out of range"):
        validate_plan_for_source(plan, source)


def test_self_test_does_not_write_dataset_sidecar(tmp_path: Path) -> None:
    from argparse import Namespace

    from musicinstruct.cli import cmd_self_test

    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=20)
    manifest = out / "pilot.jsonl"
    sidecar = out / "_gold_predictions.jsonl"
    assert cmd_self_test(Namespace(dataset=str(manifest), joint_threshold=0.9)) == 0
    assert not sidecar.exists()


def test_validate_dataset_rejects_unknown_predicate(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    item = BenchmarkItem(
        item_id="x1",
        composition_id="seed_00",
        gold_mode="unique",
        op_family="transpose",
        instruction="Transpose up 1 semitone.",
        midi_in=str(source),
        gold_midi=str(source),
        must_change=[Predicate(name="not_a_real_predicate", params={})],
        must_preserve=[Predicate(name="notes_unchanged_outside_mask", params={})],
        split="test",
    )
    manifest = tmp_path / "gold.jsonl"
    manifest.write_text(item.model_dump_json() + "\n", encoding="utf-8")
    report = validate_dataset(manifest)
    assert not report["valid"]
    assert any("unknown predicate" in err for err in report["errors"])


def test_edit_mask_rejects_inverted_time_range() -> None:
    with pytest.raises(ValueError, match="end_time must be >= start_time"):
        EditMask(start_time=4.0, end_time=1.0)


def test_normalize_velocity_defaults_to_all_tracks(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    item = BenchmarkItem(
        item_id="x",
        composition_id="seed_00",
        gold_mode="unique",
        op_family="velocity_scale",
        instruction="Scale velocities by 0.5.",
        midi_in=str(source),
    )
    plan = normalize_plan(Plan(op="velocity_scale", params={"factor": 0.5}), item)
    assert plan.params["tracks"] == [0, 1, 2]
