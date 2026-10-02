from pathlib import Path

from musicinstruct.dataset import load_jsonl, resolve_item_paths
from musicinstruct.evaluation import gold_predictions, score_records, write_predictions
from musicinstruct.metrics import ItemScores, aggregate_by_composition
from musicinstruct.schema import BenchmarkItem


def _item(composition_id: str, item_id: str) -> BenchmarkItem:
    return BenchmarkItem(
        item_id=item_id,
        composition_id=composition_id,
        gold_mode="unique",
        op_family="transpose",
        instruction="noop",
        midi_in="in.mid",
        gold_midi="gold.mid",
        must_change=[],
        must_preserve=[],
        split="test",
    )


def test_composition_macro_differs_from_item_micro() -> None:
    items = [_item("comp_a", f"a{i}") for i in range(10)] + [_item("comp_b", f"b{i}") for i in range(2)]
    rows = [
        ItemScores(
            item_id=item.item_id,
            validity=1.0,
            edit_success=1.0,
            preserve=1.0,
            joint=1.0 if item.composition_id == "comp_a" else 0.0,
            over_edit=0.0,
            gold_note_f1=1.0,
            plan_match=None,
            edit_details=[],
            preserve_details=[],
        )
        for item in items
    ]

    composition_macro, by_composition = aggregate_by_composition(items, rows)

    assert composition_macro["n_compositions"] == 2
    assert composition_macro["joint"] == 0.5
    assert by_composition["comp_a"]["joint"] == 1.0
    assert by_composition["comp_b"]["joint"] == 0.0
    item_micro_joint = sum(row.joint for row in rows) / len(rows)
    assert item_micro_joint > composition_macro["joint"]


def test_score_records_includes_composition_macro(tmp_path: Path) -> None:
    from musicinstruct.generate import TARGET_ITEMS, generate_pilot_dataset

    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=TARGET_ITEMS)
    manifest = out / "pilot.jsonl"
    items = [resolve_item_paths(item, out) for item in load_jsonl(manifest)]
    preds = tmp_path / "preds.jsonl"
    write_predictions(preds, gold_predictions(items))
    results = score_records(manifest, preds, split="test")
    assert "composition_macro" in results
    assert "by_composition" in results
    assert results["composition_macro"]["n_compositions"] == 2
    assert results["composition_macro"]["joint"] == 1.0
