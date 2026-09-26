# Submission validation commands

Always run these from `student_resource/` after `matching_results.tsv` and `candidate_pairs.tsv` exist under `output/`.

## Fast check (use every time)

```powershell
cd E:\mlchallenge\student_resource

python utils\validate_submission.py `
  --matching output\matching_results.tsv `
  --candidate output\candidate_pairs.tsv `
  --test-dir dataset\test
```

Or:

```powershell
powershell -File scripts\validate_submission.ps1
```

Expected success line:

```text
PASS — no blocking issues found. Safe to submit.
```

## Full ID-existence check (more RAM)

```powershell
python utils\validate_submission.py `
  --matching output\matching_results.tsv `
  --candidate output\candidate_pairs.tsv `
  --test-dir dataset\test `
  --check-ids
```

Or:

```powershell
powershell -File scripts\validate_submission.ps1 -CheckIds
```

## Matching file only

```powershell
python utils\validate_submission.py -m output\matching_results.tsv -t dataset\test
```

(Candidate checks are skipped with a warning if the file is omitted.)
