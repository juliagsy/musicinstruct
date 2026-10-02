# Google Colab — MIDI-Instruct plan executor

Notebooks to run **Llama plan-then-execute** on the MIDI-Instruct test split.

## One runtime per notebook

Colab gives **each notebook its own VM**. Pass artifacts through **Google Drive** (`USE_DRIVE=True`, default).

| Drive path | Contents |
|------------|----------|
| `MyDrive/musicinstruct/data/pilot/` | Synthetic pilot dataset (`pilot.jsonl`, 300 items) |
| `MyDrive/musicinstruct/data/v0.2/` | Real MidiCaps/Lakh dataset (`manifest.jsonl`) |
| `MyDrive/musicinstruct/midicaps/midicaps.tar.gz` | MidiCaps tarball (~1.6 GB; extract locally — do **not** sync extracted tree) |
| `MyDrive/musicinstruct/runs/pilot_v1/plan_executor_llama/` | **Pilot** Llama predictions (done — do not rerun) |
| `MyDrive/musicinstruct/runs/real_v0.2_v1/plan_executor_llama/` | **Real v0.2** Llama predictions |
| `MyDrive/musicinstruct/runs/<RUN_ID>/baselines/` | Programmatic baseline predictions + report |

Set **`DATASET_MODE`** and **`RUN_ID`** consistently in notebooks **01–03**:

| Mode | `DATASET_MODE` | `RUN_ID` | Manifest |
|------|----------------|----------|----------|
| Pilot (already scored) | `"pilot"` | `pilot_v1` | `data/pilot/pilot.jsonl` |
| Real v0.2 (current focus) | `"real_v0.2"` | `real_v0.2_v1` | `data/v0.2/manifest.jsonl` |

## Runtime requirements

| Notebook | GPU | Notes |
|----------|-----|-------|
| `01_setup_and_data` | CPU OK | Generate/restore data, validate, self-test |
| `02_run_plan_executor` | **T4 required** | Llama inference on CUDA |
| `03_score_and_baselines` | CPU OK | Rescore + baselines only |

## Prerequisites

1. [Accept the Llama 3.2 license](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) on Hugging Face.
2. Create a [HF access token](https://huggingface.co/settings/tokens) (read access).
3. For notebook **02**: **Runtime → Change runtime type → T4 GPU**.

## Real v0.2 workflow (recommended)

Pilot results (joint 0.44, n=50) are already on Drive under `runs/pilot_v1/`. Use a **separate `RUN_ID`** so real runs do not overwrite them.

1. Push `musicinstruct` to GitHub.
2. **01** (CPU): `DATASET_MODE="real_v0.2"` → generate or restore from Drive → sync `data/v0.2/`.
3. **02** (T4): `DATASET_MODE="real_v0.2"`, `RUN_ID="real_v0.2_v1"` → smoke with `MAX_ITEMS=5`, then `MAX_ITEMS=None`.
4. **03** (CPU): same mode + `RUN_ID` → rescore; check `overall` and `composition_macro` in `score_report.json`.

Timing (T4, one session): real v0.2 test ~180 subsampled items → ~2–3 h.

## Tips

- Use **`02`** with `RESUME=True` to continue after disconnect (merges into existing `predictions.jsonl`).
- Reuse MidiCaps tarball from midi-llm: set `DRIVE_MIDICAPS_TAR="/content/drive/MyDrive/midi-llm/midicaps/midicaps.tar.gz"` in notebook **01**.
- If you previously synced an extracted `lmd_full/` tree to Drive, delete that folder and re-sync the tarball only (~10–25 min vs hours).
- Set `RUN_BASELINES=True` in **03** once to generate oracle/copy-source/wrong-transform controls for v0.2.
