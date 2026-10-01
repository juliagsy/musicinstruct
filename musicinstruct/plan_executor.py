"""Parse edit instructions into structured plans and execute them."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

from .midi import load_midi, midi_summary
from .schema import BenchmarkItem, Plan, Prediction
from .transforms import (
    mute_tracks,
    non_drum_tracks,
    program_change,
    tempo_scale,
    transpose,
    velocity_scale,
)

SUPPORTED_OPS = frozenset({"transpose", "tempo_scale", "velocity_scale", "mute_tracks", "program_change"})

PLAN_PROMPT = """You are a MIDI editing assistant. Given a source MIDI file summary and a natural-language edit instruction, output exactly one JSON object describing the edit to apply.

Supported operations:
- transpose: params {{"semitones": int, "tracks": [int, ...] optional}}
  Use positive semitones for up/raise/higher, negative for down/lower.
  If tracks omitted, apply to all non-drum tracks.
- tempo_scale: params {{"factor": float}}
  Values >1 mean faster, <1 mean slower.
- velocity_scale: params {{"factor": float, "tracks": [int, ...] optional}}
- mute_tracks: params {{"tracks": [int, ...]}}
- program_change: params {{"track": int, "program": int}}
  program is General MIDI program number 0-127.

Respond with JSON only, no markdown or explanation:
{{"op": "<operation>", "params": {{...}}}}

Source MIDI summary:
{context}

Edit instruction:
{instruction}
"""


def build_track_context(midi_path: str | Path) -> dict:
    midi = load_midi(midi_path)
    tracks = []
    for index, instrument in enumerate(midi.instruments):
        note_count = len(instrument.notes)
        tracks.append(
            {
                "index": int(index),
                "name": instrument.name or f"track_{index}",
                "program": int(instrument.program),
                "is_drum": bool(instrument.is_drum),
                "note_count": int(note_count),
            }
        )
    summary = midi_summary(midi_path)
    return {
        "tracks": tracks,
        "duration_sec": float(summary["duration_sec"]),
        "estimated_bpm": float(summary["estimated_bpm"]),
        "note_count": int(summary["notes"]),
    }


def build_plan_prompt(item: BenchmarkItem) -> str:
    context = json.dumps(build_track_context(item.midi_in), indent=2)
    return PLAN_PROMPT.format(context=context, instruction=item.instruction)


def _extract_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    objects.append(text[start : end + 1])
                    break
    return objects


def parse_plan_text(text: str) -> Plan:
    """Parse a model response into a structured plan.

    Extracts each balanced JSON object from the text and returns the last
    object that passes plan validation. Models sometimes emit draft JSON
    before the final answer; the last valid plan is treated as authoritative.
    """
    stripped = text.strip()
    candidates: list[str] = []
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidates.append(fence_match.group(1))
    candidates.extend(_extract_json_objects(stripped))
    if stripped not in candidates:
        candidates.append(stripped)

    last_error: Exception | None = None
    valid_plans: list[Plan] = []
    seen_candidates: set[str] = set()
    for candidate in candidates:
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        try:
            payload = json.loads(candidate)
            plan = Plan.model_validate(payload)
            validate_plan(plan)
            valid_plans.append(plan)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            last_error = exc
            continue
    if valid_plans:
        return valid_plans[-1]
    raise ValueError(f"could not parse plan JSON from model output: {last_error}")


def validate_plan(plan: Plan) -> None:
    if plan.op not in SUPPORTED_OPS:
        raise ValueError(f"unsupported op {plan.op!r}")
    params = plan.params

    if plan.op == "transpose":
        if "semitones" not in params:
            raise ValueError("transpose requires semitones")
        int(params["semitones"])
        if "tracks" in params and not isinstance(params["tracks"], list):
            raise ValueError("transpose tracks must be a list")
    elif plan.op == "tempo_scale":
        if "factor" not in params:
            raise ValueError("tempo_scale requires factor")
        float(params["factor"])
    elif plan.op == "velocity_scale":
        if "factor" not in params:
            raise ValueError("velocity_scale requires factor")
        float(params["factor"])
        if "tracks" in params and not isinstance(params["tracks"], list):
            raise ValueError("velocity_scale tracks must be a list")
    elif plan.op == "mute_tracks":
        tracks = params.get("tracks")
        if not isinstance(tracks, list) or not tracks:
            raise ValueError("mute_tracks requires non-empty tracks list")
    elif plan.op == "program_change":
        if "track" not in params or "program" not in params:
            raise ValueError("program_change requires track and program")
        program = int(params["program"])
        if not 0 <= program <= 127:
            raise ValueError(f"program must be 0..127, got {program}")
        int(params["track"])


def validate_plan_for_source(plan: Plan, source: str | Path) -> None:
    validate_plan(plan)
    n_tracks = len(load_midi(source).instruments)
    params = plan.params

    def _check_tracks(tracks: list) -> None:
        for track_idx in tracks:
            idx = int(track_idx)
            if idx < 0 or idx >= n_tracks:
                raise ValueError(f"track {idx} out of range 0..{n_tracks - 1}")

    if plan.op in {"transpose", "velocity_scale", "mute_tracks"}:
        tracks = params.get("tracks")
        if tracks is not None:
            _check_tracks(list(tracks))
    if plan.op == "program_change":
        _check_tracks([params["track"]])


def normalize_plan(plan: Plan, item: BenchmarkItem) -> Plan:
    params = dict(plan.params)
    if plan.op == "transpose" and "tracks" not in params:
        params["tracks"] = non_drum_tracks(item.midi_in)
    return Plan(op=plan.op, params=params)


def execute_plan(source: str | Path, destination: str | Path, plan: Plan) -> Path:
    validate_plan_for_source(plan, source)
    op = plan.op
    params = plan.params

    if op == "transpose":
        tracks = params.get("tracks")
        result = transpose(
            source,
            destination,
            semitones=int(params["semitones"]),
            tracks=list(tracks) if tracks is not None else None,
        )
    elif op == "tempo_scale":
        result = tempo_scale(source, destination, factor=float(params["factor"]))
    elif op == "velocity_scale":
        tracks = params.get("tracks")
        result = velocity_scale(
            source,
            destination,
            factor=float(params["factor"]),
            tracks=list(tracks) if tracks is not None else None,
        )
    elif op == "mute_tracks":
        result = mute_tracks(source, destination, tracks=[int(t) for t in params["tracks"]])
    elif op == "program_change":
        result = program_change(
            source,
            destination,
            track=int(params["track"]),
            program=int(params["program"]),
        )
    else:
        raise ValueError(f"unsupported op {op!r}")
    return result.gold_path


class PlanClient(Protocol):
    def propose_plan(self, item: BenchmarkItem) -> tuple[Plan | None, str, dict]:
        """Return parsed plan, raw model text, and extra metadata."""


class StubPlanClient:
    """Test helper that returns preconfigured plans keyed by item_id."""

    def __init__(self, plans: dict[str, Plan]) -> None:
        self.plans = plans

    def propose_plan(self, item: BenchmarkItem) -> tuple[Plan | None, str, dict]:
        plan = self.plans.get(item.item_id)
        raw = plan.model_dump_json() if plan else ""
        return plan, raw, {"client": "stub"}


def predict_plan_executor(
    item: BenchmarkItem,
    client: PlanClient,
    output_dir: str | Path,
) -> Prediction:
    output_dir = Path(output_dir)
    plan: Plan | None = None
    raw_response = ""
    metadata: dict = {"runner": "plan_executor"}

    try:
        plan, raw_response, client_meta = client.propose_plan(item)
        metadata.update(client_meta)
        metadata["raw_response"] = raw_response
        if plan is None:
            metadata["error"] = "no_plan_returned"
            return Prediction(item_id=item.item_id, midi_path=None, plan=None, metadata=metadata)

        plan = normalize_plan(plan, item)
        validate_plan_for_source(plan, item.midi_in)
        metadata["parsed_plan"] = plan.model_dump()
        destination = output_dir / f"{item.item_id}.mid"
        execute_plan(item.midi_in, destination, plan)
        return Prediction(item_id=item.item_id, midi_path=str(destination), plan=plan, metadata=metadata)
    except Exception as exc:  # noqa: BLE001 — collect per-item failures for scoring
        metadata["error"] = str(exc) or f"{type(exc).__name__}"
        if plan is not None:
            metadata["parsed_plan"] = plan.model_dump()
        metadata["raw_response"] = raw_response
        return Prediction(item_id=item.item_id, midi_path=None, plan=plan, metadata=metadata)


def run_plan_executor(
    items: list[BenchmarkItem],
    client: PlanClient,
    output_dir: str | Path,
) -> list[Prediction]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [predict_plan_executor(item, client, output_dir) for item in items]
