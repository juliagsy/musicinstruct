from pathlib import Path

from musicinstruct.dataset import load_jsonl, resolve_item_paths
from musicinstruct.evaluation import score_records, write_predictions
from musicinstruct.generate import TARGET_ITEMS, generate_pilot_dataset
from musicinstruct.plan_executor import (
    StubPlanClient,
    build_plan_prompt,
    execute_plan,
    normalize_plan,
    parse_plan_text,
    predict_plan_executor,
    run_plan_executor,
)
from musicinstruct.llm_client import load_env_file, resolve_hf_token
from musicinstruct.plan_runner import run_plan_executor_suite
from musicinstruct.schema import BenchmarkItem, Plan
from musicinstruct.transforms import make_seed_midi


def test_resolve_hf_token_prefers_explicit(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env_token")
    assert resolve_hf_token("cli_token") == "cli_token"
    assert resolve_hf_token() == "env_token"


def test_load_env_file_sets_missing_vars(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('HF_TOKEN="from_dotenv"\n', encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    import musicinstruct.llm_client as llm_client

    llm_client._ENV_LOADED = False
    load_env_file()
    assert resolve_hf_token() == "from_dotenv"
    llm_client._ENV_LOADED = False


def test_parse_plan_text_from_markdown_fence() -> None:
    raw = 'Here is the plan:\n```json\n{"op": "tempo_scale", "params": {"factor": 1.25}}\n```'
    plan = parse_plan_text(raw)
    assert plan.op == "tempo_scale"
    assert plan.params["factor"] == 1.25


def test_execute_transpose_writes_output(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "source.mid", seed_index=0)
    destination = tmp_path / "out.mid"
    plan = Plan(op="transpose", params={"semitones": 3, "tracks": [0, 1]})
    execute_plan(source, destination, plan)
    assert destination.is_file()


def test_normalize_plan_fills_transpose_tracks(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "source.mid", seed_index=0)
    item = BenchmarkItem(
        item_id="x",
        composition_id="seed_00",
        gold_mode="unique",
        op_family="transpose",
        instruction="Transpose up 2 semitones.",
        midi_in=str(source),
    )
    plan = normalize_plan(Plan(op="transpose", params={"semitones": 2}), item)
    assert plan.params["tracks"] == [0, 1]


def test_stub_client_achieves_perfect_score_on_test_split(tmp_path: Path) -> None:
    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=TARGET_ITEMS)
    manifest = out / "pilot.jsonl"
    items = [
        resolve_item_paths(item, out)
        for item in load_jsonl(manifest)
        if item.split == "test" and item.plan is not None
    ]
    client = StubPlanClient({item.item_id: item.plan for item in items if item.plan})
    summary = run_plan_executor_suite(
        manifest,
        client,
        tmp_path / "results",
        split="test",
    )
    assert summary["overall"]["joint"] == 1.0


def test_predict_plan_executor_records_parse_errors(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "source.mid", seed_index=0)
    item = BenchmarkItem(
        item_id="bad",
        composition_id="seed_00",
        gold_mode="unique",
        op_family="transpose",
        instruction="Transpose up 2 semitones.",
        midi_in=str(source),
    )
    client = StubPlanClient({})
    pred = predict_plan_executor(item, client, tmp_path / "out")
    assert pred.midi_path is None
    assert pred.metadata.get("error") == "no_plan_returned"


def test_build_plan_prompt_includes_instruction(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "source.mid", seed_index=0)
    item = BenchmarkItem(
        item_id="x",
        composition_id="seed_00",
        gold_mode="unique",
        op_family="tempo_scale",
        instruction="Change the tempo by factor 1.5.",
        midi_in=str(source),
    )
    prompt = build_plan_prompt(item)
    assert "Change the tempo by factor 1.5." in prompt
    assert '"index": 0' in prompt


def test_run_plan_executor_writes_scorable_predictions(tmp_path: Path) -> None:
    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=TARGET_ITEMS)
    items = [resolve_item_paths(item, out) for item in load_jsonl(out / "pilot.jsonl")[:5]]
    mini_manifest = tmp_path / "mini.jsonl"
    with mini_manifest.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(item.model_dump_json() + "\n")
    client = StubPlanClient({item.item_id: item.plan for item in items if item.plan})
    preds = run_plan_executor(items, client, tmp_path / "midi")
    pred_path = tmp_path / "preds.jsonl"
    write_predictions(pred_path, preds)
    scores = score_records(mini_manifest, pred_path)
    assert scores["overall"]["joint"] == 1.0
