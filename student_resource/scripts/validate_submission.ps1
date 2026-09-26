# Validate submission outputs (Windows)

# Usage (from student_resource/):
#   powershell -File scripts\validate_submission.ps1
#   powershell -File scripts\validate_submission.ps1 -CheckIds

param(
    [switch]$CheckIds
)

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$argsList = @(
    "utils\validate_submission.py",
    "--matching", "output\matching_results.tsv",
    "--candidate", "output\candidate_pairs.tsv",
    "--test-dir", "dataset\test"
)

if ($CheckIds) {
    $argsList += "--check-ids"
}

Write-Host "Running: python $($argsList -join ' ')"
python @argsList
exit $LASTEXITCODE
