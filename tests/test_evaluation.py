from pathlib import Path

from musicinstruct.dataset import save_jsonl, validate_dataset
from musicinstruct.evaluation import gold_predictions, score_records, write_predictions
from musicinstruct.schema import Prediction
from musicinstruct.generate import TARGET_ITEMS, generate_pilot_dataset
from musicinstruct.schema import BenchmarkItem
from musicinstruct.transforms import make_seed_midi, transpose


def test_end_to_end_self_score(tmp_path: Path) -> None:
    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=300)
    manifest = out / "pilot.jsonl"
    report = validate_dataset(manifest)
    assert report["valid"]
    assert report["count"] == 300

    preds = out / "preds.jsonl"
    from musicinstruct.dataset import load_jsonl, resolve_item_paths

    items = [resolve_item_paths(i, out) for i in load_jsonl(manifest)]
    write_predictions(preds, gold_predictions(items))
    results = score_records(manifest, preds)
    assert results["overall"]["joint"] == 1.0
    assert results["overall"]["edit_success"] >= 0.999
    assert results["overall"]["preserve"] == 1.0
    assert results["prediction_coverage"] == 1.0
    assert results["missing_prediction_ids"] == []


def test_score_records_resolves_cwd_relative_midi_path(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=300)
    manifest = out / "pilot.jsonl"
    item = next(i for i in __import__("musicinstruct.dataset", fromlist=["load_jsonl"]).load_jsonl(manifest) if i.split == "test")
    from musicinstruct.dataset import load_jsonl, resolve_item_paths

    item = resolve_item_paths(item, out)
    preds_dir = tmp_path / "results"
    midi_dir = preds_dir / "midi"
    midi_dir.mkdir(parents=True)
    gold = Path(item.gold_midi)
    dest = midi_dir / f"{item.item_id}.mid"
    dest.write_bytes(gold.read_bytes())
    monkeypatch.chdir(tmp_path)
    write_predictions(
        preds_dir / "predictions.jsonl",
        [
            Prediction(
                item_id=item.item_id,
                midi_path=str(Path.cwd().relative_to(tmp_path) / "results" / "midi" / f"{item.item_id}.mid"),
                plan=item.plan,
            )
        ],
    )
    results = score_records(manifest, preds_dir / "predictions.jsonl", item_ids={item.item_id})
    assert results["items"][0]["validity"] == 1.0


def test_write_predictions_relativizes_external_midi_path(tmp_path: Path) -> None:
    midi = make_seed_midi(tmp_path / "data/source.mid", seed_index=0)
    preds = tmp_path / "results/predictions.jsonl"
    write_predictions(preds, [Prediction(item_id="x", midi_path=str(midi))])
    row = __import__("json").loads(preds.read_text())
    assert row["midi_path"] == "../data/source.mid"


def test_missing_prediction_gold_note_f1_counts_as_zero(tmp_path: Path) -> None:
    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=TARGET_ITEMS)
    manifest = out / "pilot.jsonl"
    from musicinstruct.dataset import load_jsonl, resolve_item_paths

    item = resolve_item_paths(
        next(i for i in load_jsonl(manifest) if i.split == "test"),
        out,
    )
    preds = tmp_path / "preds.jsonl"
    write_predictions(
        preds,
        [Prediction(item_id=item.item_id, midi_path=None, plan=None, metadata={})],
    )
    results = score_records(manifest, preds, split="test", item_ids={item.item_id})
    assert results["overall"]["gold_note_f1"] == 0.0


def test_missing_prediction_scores_zero(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    gold = tmp_path / "gold.mid"
    result = transpose(source, gold, semitones=1)
    item = BenchmarkItem(
        item_id="x1",
        composition_id="seed_00",
        gold_mode="unique",
        op_family="transpose",
        instruction="Transpose up 1 semitone.",
        midi_in=str(source),
        gold_midi=str(gold),
        plan=result.plan,
        must_change=result.must_change,
        must_preserve=result.must_preserve,
        edit_mask=result.edit_mask,
        split="test",
    )
    manifest = tmp_path / "gold.jsonl"
    save_jsonl(manifest, [item])
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    results = score_records(manifest, empty)
    assert results["overall"]["joint"] == 0.0
