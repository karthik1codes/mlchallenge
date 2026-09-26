"""Central paths and hyperparameters for the ER pipeline."""
from pathlib import Path
import os

# Repo root = student_resource/
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset"
TRAIN = DATA / "train"
TEST = DATA / "test"
OUTPUT = ROOT / "output"
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

TRAIN_S1 = TRAIN / "train_source1.tsv"
TRAIN_S2 = TRAIN / "train_source2.tsv"
TRAIN_S3 = TRAIN / "train_source3.tsv"
TRAIN_GT = TRAIN / "train_ground_truth.tsv"

TEST_S1 = TEST / "test_source1.tsv"
TEST_S2 = TEST / "test_source2.tsv"
TEST_S3 = TEST / "test_source3.tsv"

MATCHING_OUT = OUTPUT / "matching_results.tsv"
CANDIDATE_OUT = OUTPUT / "candidate_pairs.tsv"

# --- Pipeline knobs ---
VAL_FRACTION = 0.05
RANDOM_SEED = 42
N_JOBS = max(1, (os.cpu_count() or 4) - 1)

# Blocking (tighter buckets = much faster ranking)
MAX_CANDIDATES_PER_S1 = 50
MAX_BLOCK_BUCKET = 600
BLOCK_RARE_TOKEN_MAX_DF = 3000
PHONETIC_PREFIX_TOKENS = 2
# Metaphone is accurate but slow; prefix codes are ~10x faster for blocking
USE_METAPHONE = False

# Matching / training
NEG_PER_POS = 2
INFER_CHUNK_SIZE = 25_000
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.1,
    "num_leaves": 47,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 40,
    "verbosity": -1,
    "n_estimators": 400,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}
DEFAULT_THRESHOLD = 0.72
