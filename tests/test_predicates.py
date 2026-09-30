from pathlib import Path

import pretty_midi

from musicinstruct.midi import load_midi
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


def test_track_muted_checks_track_index_not_identity(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    wrong = tmp_path / "wrong.mid"
    src = load_midi(source)
    piano = pretty_midi.Instrument(program=0, name="piano")
    for note in src.instruments[0].notes:
        piano.notes.append(pretty_midi.Note(velocity=1, pitch=note.pitch, start=note.start, end=note.end))
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    for note in src.instruments[2].notes:
        drums.notes.append(pretty_midi.Note(velocity=100, pitch=note.pitch, start=note.start, end=note.end))
    reordered = pretty_midi.PrettyMIDI()
    reordered.instruments.extend([drums, src.instruments[1], piano])
    reordered.write(str(wrong))

    preds = evaluate_predicates(
        [Predicate(name="track_muted", params={"tracks": [0], "max_velocity": 1})],
        str(source),
        str(wrong),
        EditMask(),
    )
    assert not preds[0].passed
    assert "track=0" in preds[0].detail


def test_pitch_shifted_fails_on_permuted_pitches(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    permuted = tmp_path / "permuted.mid"
    midi = load_midi(source)
    notes = list(midi.instruments[0].notes)
    if len(notes) >= 2:
        notes[0].pitch, notes[1].pitch = notes[1].pitch, notes[0].pitch
    midi.write(str(permuted))
    preds = evaluate_predicates(
        [
            Predicate(
                name="pitch_shifted_by",
                params={"semitones": 2, "tracks": [0], "tolerance": 0.5},
            )
        ],
        str(source),
        str(permuted),
        EditMask(tracks=[0], include_drums=False),
    )
    assert not preds[0].passed
