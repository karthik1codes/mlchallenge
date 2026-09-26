# Business Entity Resolution — Runnable Pipeline

Target metric: **macro F₀.₅** (precision-heavy). See `APPROACH.md` for the strategy toward ~0.98.

## Layout

```
business_entity_resolution/
├── APPROACH.md          # how we aim for ~0.98 F₀.₅
├── README.md            # this file
├── requirements.txt
├── artifacts/           # trained model + threshold (created by train)
└── src/
    ├── config.py
    ├── normalize.py
    ├── blocking.py
    ├── features.py
    ├── metrics.py
    ├── io_utils.py
    ├── train.py
    └── infer.py
```

Outputs are written to `student_resource/output/`:
- `matching_results.tsv` — leaderboard file
- `candidate_pairs.tsv` — last blocking set fed to the model

## Setup

From this folder:

```bash
cd student_resource/code/business_entity_resolution
python -m pip install -r requirements.txt
```

## Quick smoke test (small subsample)

```bash
# Train on a linked slice (cached after first run — much faster on repeat)
python -m src.train --nrows 3000

# Infer on a test slice
python -m src.infer --split test --nrows 3000
```

Second `--nrows 3000` train reuses `artifacts/linked_smoke_s1*.pkl`.

## Faster full-ish train (recommended before full infer)

Train the matcher on a capped S1 sample while still using the full S2/S3 pool:

```bash
python -m src.train --max-train-s1 100000
python -m src.infer --split test --n-jobs 8 --chunk-size 25000
```

## Full run

```bash
python -m src.train
python -m src.infer --split test
```

## Speed knobs

| Flag / setting | Effect |
|----------------|--------|
| `--nrows N` | Linked smoke sample + disk cache |
| `--max-train-s1 N` | Cap S1 when training on full files |
| `--n-jobs` | Parallel blocking threads |
| `--chunk-size` | Infer S1 batch size |
| `USE_METAPHONE=False` in `config.py` | Faster blocking keys (default) |
| `MAX_CANDIDATES_PER_S1` | Fewer candidates → faster scoring |
## Validate submission (required before Portal upload)

Run from **`student_resource/`** (not from `code/`):

### Windows (PowerShell)

```powershell
cd E:\mlchallenge\student_resource

# Fast format check (recommended every time)
python utils\validate_submission.py `
  --matching output\matching_results.tsv `
  --candidate output\candidate_pairs.tsv `
  --test-dir dataset\test

# Stricter: also verify every S2/S3 ID exists (uses more RAM)
python utils\validate_submission.py `
  --matching output\matching_results.tsv `
  --candidate output\candidate_pairs.tsv `
  --test-dir dataset\test `
  --check-ids
```

### Linux / macOS

```bash
cd student_resource

python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test

python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```

You want: **`PASS — no blocking issues found. Safe to submit.`**

Notes from the official validator:
- Exit `0` = safe; exit `1` = fix listed errors
- `--check-ids` is optional (diagnostic; loads all S2/S3 IDs into memory)
- Matching IDs absent from `candidate_pairs.tsv` only **warn** (still fix them)

## Leaderboard upload

Upload only:

`student_resource/output/matching_results.tsv`

## Path to ~0.98 F₀.₅

1. Confirm val blocking recall ≥ 0.99 (`src.train` prints it)
2. Sweep threshold for max macro F₀.₅ (already in `src.train`)
3. Raise `MAX_CANDIDATES_PER_S1` / add blocking keys if recall is low
4. Add hard-negative mining + more features if precision is low
5. Optional: offline embedding cosine feature (Apache MiniLM) as Phase-2

Details: `APPROACH.md`.
