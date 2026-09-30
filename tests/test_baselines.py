from pathlib import Path

from musicinstruct.baselines import predict_baseline, run_baselines
from musicinstruct.benchmark_runner import run_baseline_suite
from musicinstruct.dataset import load_jsonl, resolve_item_paths
from musicinstruct.evaluation import score_records, write_predictions
from musicinstruct.generate import TARGET_ITEMS, generate_pilot_dataset


def test_baselines_on_pilot_test_split(tmp_path: Path) -> None:
    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=TARGET_ITEMS)
    manifest = out / "pilot.jsonl"

    results_dir = tmp_path / "baseline_results"
    summary = run_baseline_suite(manifest, results_dir, split="test")
    assert "error" not in summary
    assert summary["item_count"] > 0

    oracle = summary["baselines"]["oracle"]["overall"]
    copy_src = summary["baselines"]["copy_source"]["overall"]
    wrong = summary["baselines"]["wrong_transform"]["overall"]
    missing = summary["baselines"]["no_output"]["overall"]

    assert oracle["joint"] == 1.0
    assert copy_src["edit_success"] < 0.5
    assert wrong["joint"] == 0.0
    assert missing["joint"] == 0.0


def test_predict_baseline_oracle_matches_gold(tmp_path: Path) -> None:
    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=10)
    item = resolve_item_paths(load_jsonl(out / "pilot.jsonl")[0], out)
    pred = predict_baseline(item, "oracle", tmp_path / "cache")
    assert pred.midi_path == item.gold_midi


def test_run_baselines_writes_predictions(tmp_path: Path) -> None:
    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=TARGET_ITEMS)
    items = [
        resolve_item_paths(i, out)
        for i in load_jsonl(out / "pilot.jsonl")
        if i.split == "test"
    ]
    preds = run_baselines(items, tmp_path / "cache", baselines=["oracle"])
    pred_path = tmp_path / "preds.jsonl"
    write_predictions(pred_path, preds["oracle"])
    scores = score_records(out / "pilot.jsonl", pred_path, split="test")
    assert scores["overall"]["joint"] == 1.0
