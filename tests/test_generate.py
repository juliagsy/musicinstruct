from musicinstruct.generate import TARGET_ITEMS, generate_pilot_dataset


def test_trim_keeps_all_test_compositions(tmp_path) -> None:
    out = tmp_path / "pilot"
    items = generate_pilot_dataset(out, target=TARGET_ITEMS)
    test_compositions = {item.composition_id for item in items if item.split == "test"}
    assert test_compositions == {"seed_10", "seed_11"}
    assert sum(1 for item in items if item.split == "test") == 50
    assert len(items) == TARGET_ITEMS
