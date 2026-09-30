from pathlib import Path

import pretty_midi

from musicinstruct.midi import extract_notes, load_midi
from musicinstruct.transforms import make_seed_midi, mute_tracks, transpose, velocity_scale


def test_transpose_shifts_pitches(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    gold = tmp_path / "gold.mid"
    result = transpose(source, gold, semitones=3)
    src_notes = [n for n in extract_notes(load_midi(source)) if not n.is_drum]
    gold_notes = [n for n in extract_notes(load_midi(gold)) if not n.is_drum]
    assert gold_notes
    for src, dst in zip(sorted(src_notes, key=lambda n: (n.track, n.start)), sorted(gold_notes, key=lambda n: (n.track, n.start)), strict=True):
        assert dst.pitch == src.pitch + 3
    assert result.plan.op == "transpose"


def test_mute_tracks_silences_target(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=1)
    gold = tmp_path / "muted.mid"
    mute_tracks(source, gold, tracks=[0])
    piano = load_midi(gold).instruments[0]
    assert piano.notes
    assert max(note.velocity for note in piano.notes) <= 1
    assert max(note.velocity for note in load_midi(source).instruments[1].notes) > 1


def test_velocity_scale_changes_mean(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=2)
    gold = tmp_path / "vel.mid"
    velocity_scale(source, gold, factor=0.5, tracks=[0])
    src = load_midi(source).instruments[0].notes
    dst = load_midi(gold).instruments[0].notes
    assert dst
    assert sum(n.velocity for n in dst) < sum(n.velocity for n in src)
