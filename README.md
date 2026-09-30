# MIDI-Instruct / MIDI-Reason

A benchmark and evaluation toolkit for **instruction-conditioned MIDI editing**.

Each item pairs a source MIDI file with a natural-language instruction. Models must emit an edited MIDI file. Scoring is **dual-axis**:

- **EditSuccess** — did the requested change happen?
- **Preserve** — did everything else stay put?

A secondary **MIDI-Reason** track scores structured plans (`op` + parameters) separately from the output file.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```bash
# Generate ~300 synthetic unique-gold items
musicinstruct generate-pilot --output-dir data/pilot --target 300

# Validate manifest and MIDI paths
musicinstruct validate data/pilot/pilot.jsonl

# Sanity-check grader (gold vs gold → joint = 1.0)
musicinstruct self-test data/pilot/pilot.jsonl

# Score model predictions (optionally one split)
musicinstruct score data/pilot/pilot.jsonl predictions.jsonl --output results.json --split test

# Reference baselines on the test split
musicinstruct run-baselines data/pilot/pilot.jsonl --output-dir results/baselines --split test

# Llama plan-then-execute (Phase 1 open model baseline)
pip install -e ".[llm]"
# Put HF_TOKEN=hf_... in .env (gitignored), or export it in your shell
musicinstruct run-plan-executor data/pilot/pilot.jsonl \
  --output-dir results/plan_executor_llama \
  --split test \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --device cpu \
  --max-items 5   # ~15 min/item on Mac CPU; drop for full split
# Or pass the token directly: --hf-token hf_...

# Grader stress test on local MIDIs (or MidiCaps + Lakh — see below)
musicinstruct stress-test --midi-dir /path/to/midis --output-dir data/stress --max-files 50
```

### MidiCaps / Lakh stress test

MidiCaps metadata resolves against a local [Lakh MIDI](https://colinraffel.com/projects/lmd/) tree:

```bash
pip install -e ".[stress]"
musicinstruct stress-test \
  --lakh-root /path/to/lmd_full \
  --midicaps-limit 100 \
  --output-dir data/stress_midicaps \
  --max-files 50
```

This validates the grader on real files (transpose + tempo per MIDI). It does **not** use MidiCaps captions as edit instructions.

## Metrics

| Metric | Meaning |
|--------|---------|
| `validity` | Output parses as valid MIDI |
| `edit_success` | Mean score of `must_change` predicates |
| `preserve` | Mean score of `must_preserve` predicates |
| `joint` | Pass if both axes ≥ threshold (default 0.9) |
| `over_edit` | Fraction of notes changed outside the edit mask |
| `gold_note_f1` | Note F1 vs gold MIDI on the edit region (`unique` gold) |
| `plan_match` | Exact match on structured plan (MIDI-Reason) |

## v0 scope

- **Unique gold** via deterministic transforms (transpose, tempo, velocity, mute, program change)
- **Constraint gold** deferred until the grader is validated on unique gold
- Optional Llama plan-then-execute runner (`run-plan-executor`, requires `.[llm]`)

See `docs/data_schema.md` and `paper/` for the research draft.

## Citation

If you use this benchmark, cite the MIDI-Instruct paper (draft in `paper/main.tex`).
