from pathlib import Path

from musicinstruct.stress import build_stress_items, discover_midi_files, run_stress_test
from musicinstruct.transforms import make_seed_midi


def test_discover_midi_dir(tmp_path: Path) -> None:
    make_seed_midi(tmp_path / "a.mid", seed_index=0)
    make_seed_midi(tmp_path / "nested/b.mid", seed_index=1)
    paths, source = discover_midi_files(midi_dir=tmp_path, max_files=10)
    assert source == "midi_dir"
    assert len(paths) == 2


def test_stress_test_on_local_midis(tmp_path: Path) -> None:
    midi_dir = tmp_path / "midis"
    midi_dir.mkdir()
    for idx in range(5):
        make_seed_midi(midi_dir / f"seed_{idx}.mid", seed_index=idx)

    report = run_stress_test(
        tmp_path / "stress_out",
        midi_dir=midi_dir,
        max_files=5,
    )
    assert report["passed"] is True
    assert report["items"] == 10
    assert report["failure_count"] == 0


def test_build_stress_items_skips_no_pitched_tracks(tmp_path: Path) -> None:
    # Drums-only would be skipped; our seeds always have pitched tracks
    source = make_seed_midi(tmp_path / "seed.mid", seed_index=0)
    items = build_stress_items([source], tmp_path / "out")
    assert len(items) == 2
    assert {item.op_family for item in items} == {"transpose", "tempo_scale"}
