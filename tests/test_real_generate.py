import json
from pathlib import Path

from musicinstruct.dataset import load_jsonl, resolve_item_paths
from musicinstruct.evaluation import (
    gold_predictions,
    score_records,
    self_test_passed,
    write_predictions,
)
from musicinstruct.generate import REAL_TARGET_ITEMS, generate_real_dataset
from musicinstruct.seeds import SeedRecord
from musicinstruct.transforms import make_seed_midi


def _seed_record(path: Path, composition_id: str, split: str) -> SeedRecord:
    return SeedRecord(
        composition_id=composition_id,
        source_path=path,
        split=split,
        cluster_id=f"cluster_{composition_id}",
    )


def test_generate_real_dataset_self_test(tmp_path: Path) -> None:
    seeds = []
    for index in range(3):
        midi = make_seed_midi(tmp_path / f"seed_{index}.mid", seed_index=index, bars=16)
        split = "train" if index < 2 else "test"
        seeds.append(_seed_record(midi, f"lmd_seed{index:02d}", split))

    out = tmp_path / "real"
    items = generate_real_dataset(out, seeds, target=REAL_TARGET_ITEMS, force=True)
    manifest = out / "manifest.jsonl"

    assert manifest.is_file()
    assert len(items) <= REAL_TARGET_ITEMS
    assert all(item.source == "midicaps" for item in items)
    assert all(item.license == "CC-BY-4.0" for item in items)

    test_compositions = {item.composition_id for item in items if item.split == "test"}
    assert "lmd_seed02" in test_compositions

    preds_path = out / "preds.jsonl"
    resolved_items = [resolve_item_paths(item, out) for item in load_jsonl(manifest)]
    write_predictions(preds_path, gold_predictions(resolved_items))
    results = score_records(manifest, preds_path, strict=True)

    assert self_test_passed(results["overall"])


def test_generate_real_manifest_is_valid_jsonl(tmp_path: Path) -> None:
    midi = make_seed_midi(tmp_path / "one.mid", seed_index=0, bars=16)
    seed = _seed_record(midi, "lmd_one", "validation")
    out = tmp_path / "real_small"
    generate_real_dataset(out, [seed], target=50, force=True)
    rows = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows
    assert all(row["composition_id"] == "lmd_one" for row in rows)
    assert all(row["split"] == "validation" for row in rows)
