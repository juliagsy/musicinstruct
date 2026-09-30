"""Machine-checkable must-change and must-preserve predicates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .midi import (
    NoteEvent,
    extract_notes,
    ioi_sequence,
    js_divergence,
    load_midi,
    note_in_mask,
    partition_notes,
    pitch_histogram,
)
from .schema import EditMask, Predicate


@dataclass
class PredicateResult:
    name: str
    passed: bool
    score: float
    detail: str


def _fail(name: str, detail: str) -> PredicateResult:
    return PredicateResult(name, False, 0.0, detail)


def _require_int(params: dict, key: str, name: str) -> tuple[int | None, PredicateResult | None]:
    if key not in params:
        return None, _fail(name, f"missing param {key!r}")
    try:
        return int(params[key]), None
    except (TypeError, ValueError):
        return None, _fail(name, f"invalid int for {key!r}: {params[key]!r}")


def _require_float(params: dict, key: str, name: str) -> tuple[float | None, PredicateResult | None]:
    if key not in params:
        return None, _fail(name, f"missing param {key!r}")
    try:
        return float(params[key]), None
    except (TypeError, ValueError):
        return None, _fail(name, f"invalid float for {key!r}: {params[key]!r}")


def _parse_tracks(params: dict, key: str, name: str) -> tuple[list[int] | None, PredicateResult | None]:
    if key not in params:
        return None, _fail(name, f"missing param {key!r}")
    raw = params[key]
    if not isinstance(raw, list):
        return None, _fail(name, f"{key} must be a list, got {type(raw).__name__}")
    tracks: list[int] = []
    for value in raw:
        try:
            tracks.append(int(value))
        except (TypeError, ValueError):
            return None, _fail(name, f"invalid track index {value!r}")
    return tracks, None


def _validate_track_indices(
    name: str,
    tracks: list[int],
    n_instruments: int,
) -> PredicateResult | None:
    for track_idx in tracks:
        if track_idx < 0 or track_idx >= n_instruments:
            return _fail(name, f"track {track_idx} out of range 0..{n_instruments - 1}")
    return None


def _median_pitch_delta(
    source: list[NoteEvent],
    hypothesis: list[NoteEvent],
    tracks: list[int],
    mask: EditMask,
) -> float:
    deltas: list[float] = []
    hyp_by_key: dict[tuple[int, float], int] = {}
    for note in hypothesis:
        if note.track in tracks and note_in_mask(note, mask):
            hyp_by_key[(note.track, round(note.start, 3))] = note.pitch
    for note in source:
        if note.track in tracks and note_in_mask(note, mask):
            key = (note.track, round(note.start, 3))
            if key in hyp_by_key:
                deltas.append(hyp_by_key[key] - note.pitch)
    if not deltas:
        src_pitches = [n.pitch for n in source if n.track in tracks and note_in_mask(n, mask)]
        hyp_pitches = [n.pitch for n in hypothesis if n.track in tracks and note_in_mask(n, mask)]
        if len(src_pitches) == len(hyp_pitches) and src_pitches:
            deltas = [h - s for s, h in zip(sorted(src_pitches), sorted(hyp_pitches), strict=True)]
    return float(np.median(deltas)) if deltas else 0.0


def eval_predicate(
    name: str,
    params: dict,
    source_path: str,
    hypothesis_path: str,
    mask: EditMask,
) -> PredicateResult:
    try:
        source_notes = extract_notes(load_midi(source_path))
        hyp_notes = extract_notes(load_midi(hypothesis_path))
    except Exception as exc:  # noqa: BLE001 — invalid MIDI should score as failure
        return _fail(name, f"midi load failed: {exc}")

    _, preserve_src = partition_notes(source_notes, mask)
    _, preserve_hyp = partition_notes(hyp_notes, mask)
    n_instruments = len(load_midi(source_path).instruments)

    if name == "pitch_shifted_by":
        semitones, err = _require_int(params, "semitones", name)
        if err:
            return err
        tracks, err = _parse_tracks(params, "tracks", name)
        if err:
            return err
        assert tracks is not None
        track_err = _validate_track_indices(name, tracks, n_instruments)
        if track_err:
            return track_err
        assert semitones is not None
        tolerance = float(params.get("tolerance", 0.5))
        delta = _median_pitch_delta(source_notes, hyp_notes, tracks, mask)
        passed = abs(delta - semitones) <= tolerance
        score = max(0.0, 1.0 - abs(delta - semitones) / max(abs(semitones), 1))
        return PredicateResult(name, passed, score, f"median_delta={delta:.2f}, expected={semitones}")

    if name == "track_muted":
        tracks, err = _parse_tracks(params, "tracks", name)
        if err:
            return err
        assert tracks is not None
        track_err = _validate_track_indices(name, tracks, n_instruments)
        if track_err:
            return track_err
        max_velocity = int(params.get("max_velocity", 1))
        src_midi = load_midi(source_path)
        hyp_midi = load_midi(hypothesis_path)
        failed: list[str] = []
        for track_idx in tracks:
            src_inst = src_midi.instruments[track_idx]
            identity = (int(src_inst.program), bool(src_inst.is_drum), src_inst.name)
            hyp_inst = next(
                (
                    inst
                    for inst in hyp_midi.instruments
                    if (int(inst.program), bool(inst.is_drum), inst.name) == identity
                ),
                None,
            )
            if hyp_inst is None:
                failed.append(f"missing:{identity}")
                continue
            peak = max((note.velocity for note in hyp_inst.notes), default=0)
            if peak > max_velocity:
                failed.append(f"loud:{identity}:peak={peak}")
        passed = len(failed) == 0
        return PredicateResult(name, passed, 1.0 if passed else 0.0, f"failures={failed}")

    if name == "tempo_scaled_by":
        factor, err = _require_float(params, "factor", name)
        if err:
            return err
        assert factor is not None
        tolerance = float(params.get("tolerance", 0.08))
        src_midi = load_midi(source_path)
        hyp_midi = load_midi(hypothesis_path)
        src_dur = max(src_midi.get_end_time(), 1e-6)
        hyp_dur = max(hyp_midi.get_end_time(), 1e-6)
        ratio = src_dur / hyp_dur
        passed = abs(ratio - factor) <= tolerance
        score = max(0.0, 1.0 - abs(ratio - factor) / max(factor, 0.01))
        return PredicateResult(name, passed, score, f"duration_ratio={ratio:.3f}, expected={factor}")

    if name == "velocity_scaled_by":
        factor, err = _require_float(params, "factor", name)
        if err:
            return err
        assert factor is not None
        tracks, err = _parse_tracks(params, "tracks", name)
        if err:
            return err
        assert tracks is not None
        track_err = _validate_track_indices(name, tracks, n_instruments)
        if track_err:
            return track_err
        tolerance = float(params.get("tolerance", 0.15))
        src_vel = [n.velocity for n in source_notes if n.track in tracks and note_in_mask(n, mask)]
        hyp_vel = [n.velocity for n in hyp_notes if n.track in tracks and note_in_mask(n, mask)]
        if not src_vel or not hyp_vel:
            return PredicateResult(name, False, 0.0, "no velocities in mask")
        ratio = float(np.mean(hyp_vel) / np.mean(src_vel))
        passed = abs(ratio - factor) <= tolerance
        score = max(0.0, 1.0 - abs(ratio - factor) / max(factor, 0.01))
        return PredicateResult(name, passed, score, f"ratio={ratio:.3f}, expected={factor}")

    if name == "program_is":
        track, err = _require_int(params, "track", name)
        if err:
            return err
        program, err = _require_int(params, "program", name)
        if err:
            return err
        assert track is not None and program is not None
        track_err = _validate_track_indices(name, [track], n_instruments)
        if track_err:
            return track_err
        hyp_midi = load_midi(hypothesis_path)
        actual = hyp_midi.instruments[track].program
        passed = actual == program
        return PredicateResult(name, passed, 1.0 if passed else 0.0, f"program={actual}, expected={program}")

    if name == "notes_unchanged_outside_mask":
        from .midi import note_set

        passed = note_set(preserve_src) == note_set(preserve_hyp)
        return PredicateResult(name, passed, 1.0 if passed else 0.0, f"preserve_notes={len(preserve_src)}")

    if name == "pitch_histogram_unchanged_outside_mask":
        max_js = float(params.get("max_js", 0.01))
        divergence = js_divergence(pitch_histogram(preserve_src), pitch_histogram(preserve_hyp))
        passed = divergence <= max_js
        score = max(0.0, 1.0 - divergence / max(max_js, 1e-6))
        return PredicateResult(name, passed, score, f"js={divergence:.4f}, max={max_js}")

    if name == "ioi_unchanged_outside_mask":
        max_delta = float(params.get("max_mean_delta", 0.05))
        src_ioi = ioi_sequence(preserve_src)
        hyp_ioi = ioi_sequence(preserve_hyp)
        if src_ioi.size == 0 and hyp_ioi.size == 0:
            return PredicateResult(name, True, 1.0, "empty ioi")
        if src_ioi.size == 0 or hyp_ioi.size == 0:
            return PredicateResult(name, False, 0.0, "ioi length mismatch")
        delta = abs(float(np.mean(src_ioi) - np.mean(hyp_ioi)))
        passed = delta <= max_delta
        score = max(0.0, 1.0 - delta / max(max_delta, 1e-6))
        return PredicateResult(name, passed, score, f"mean_ioi_delta={delta:.4f}")

    if name == "track_set_unchanged":
        src_tracks = {n.track for n in source_notes}
        hyp_tracks = {n.track for n in hyp_notes}
        passed = src_tracks == hyp_tracks
        return PredicateResult(name, passed, 1.0 if passed else 0.0, f"src={sorted(src_tracks)}, hyp={sorted(hyp_tracks)}")

    if name == "pitches_unchanged":
        src = sorted((n.track, n.pitch) for n in source_notes)
        hyp = sorted((n.track, n.pitch) for n in hyp_notes)
        passed = src == hyp
        return PredicateResult(name, passed, 1.0 if passed else 0.0, f"pitch_events={len(src)}")

    return PredicateResult(name, False, 0.0, f"unknown predicate: {name}")


def evaluate_predicates(
    predicates: list[Predicate],
    source_path: str,
    hypothesis_path: str,
    mask: EditMask,
) -> list[PredicateResult]:
    return [
        eval_predicate(p.name, p.params, source_path, hypothesis_path, mask)
        for p in predicates
    ]
