# Approach: Targeting F₀.₅ ≈ 0.98

> Official metric is **F₀.₅** (precision-heavy), not classic F1.  
> Macro-averaged over every Source-1 entity, including singletons.

## Why 0.98 is hard (and reachable)

| Fact | Implication |
|------|-------------|
| ~2.2M train / ~1.7M test S1 entities | Blocking must be O(N log N), not O(N²) |
| ~94% of S1 have ≥1 match; ~avg 3.5 matches | Recall still matters, but **false merges kill F₀.₅** |
| Names/addresses are noisy but often near-duplicates | Strong normalization + similarity features work |
| Test adds **France** (unseen in train) | Never hard-code `{US, India}` |
| Singletons score 1.0 if empty, 0.0 if any wrong match | Prefer abstaining over guessing |

To reach **~0.98**, you roughly need:
- Blocking recall ≥ **0.995** (true match must enter the candidate set)
- Matching precision ≥ **0.99** at operating threshold
- Near-perfect singleton handling

## Pipeline (4 stages)

```
1. Normalize  →  cleaned name/address + extracted geo tokens
2. Block      →  multi-key candidate generation  →  candidate_pairs.tsv
3. Score      →  LightGBM on pair features (+ optional embedding cosine)
4. Decide     →  precision-tuned threshold → matching_results.tsv
```

### Stage 1 — Normalize
- Lowercase, strip punctuation, unify `&`/`and`
- Expand Rd/St/Ave, Pvt/Ltd/Corp, remove legal suffixes for a *core name*
- Extract zip/PIN, city-ish tokens, street numbers
- Keep `country` as a free string (works for France)

### Stage 2 — Multi-key blocking (recall ceiling)
Union of several cheap keys (same country enforced when present):

1. Exact core-name + country  
2. Phonetic (Double Metaphone) of first 2 name tokens + country  
3. Shared rare TF-IDF char-ngrams / word tokens (top-k inverted index)  
4. ZIP/PIN + first name token  
5. Sorted-token name signature (handles word-order swaps like `O.D., Sofie Greenman`)

Cap candidates per S1 (e.g. 50–150) by a cheap pre-score so inference stays feasible.  
Whatever is fed to the model **is** `candidate_pairs.tsv`.

### Stage 3 — Pair features + LightGBM
Features (fast, no external lookup):
- Name: Jaccard, token F1, char 3-gram Dice, Levenshtein ratio, sorted-token Jaccard, phonetic match
- Address: same + ZIP equality, street-number equality, token overlap
- Meta: same country, length ratios, exact-normalized match flags

Model: **LightGBM** binary classifier (MIT license, tiny vs 8B limit).  
Optional Phase-2: small Apache sentence embedding (e.g. `all-MiniLM-L6-v2`) cosine as an extra feature / reranker — only if you can run it locally offline after one Hub download during development (no live external lookup at inference time beyond the licensed model weights you ship).

### Stage 4 — Threshold for F₀.₅
- Hold out ~5–10% of S1 IDs for validation
- Sweep probability thresholds; pick max **macro F₀.₅**
- Bias slightly toward higher threshold if public LB is noisy (precision > recall)

## Training recipe

1. Build positives from `train_ground_truth.tsv`  
2. Hard negatives = blocked non-matches (not random distant pairs)  
3. Sample ~1–3 negatives per positive for class balance  
4. Train LightGBM with early stopping on val F₀.₅ / AUC  
5. Calibrate / threshold on held-out S1 entities  
6. Full retrain on all train (optional) with frozen threshold from val

## Iteration path to 0.98

| Step | Focus | Target |
|------|-------|--------|
| A | Baseline normalize + name blocking + rules | F₀.₅ ≥ 0.70 |
| B | Multi-key blocking + LightGBM | ≥ 0.90 |
| C | Better features, hard-neg mining, threshold | ≥ 0.95 |
| D | Embedding rerank + error analysis on FPs | ≥ 0.98 |

**Error analysis loop (critical):**
- False positives → raise threshold / add mismatch features (different ZIP, different street #)
- False negatives → expand blocking keys / lower pre-filter
- Singleton FPs → require higher score when S1 has weak evidence

## Fair play
No geocoding APIs, business registries, or external ER services.  
Only provided TSVs + licensed local models ≤ 8B params (MIT/Apache 2.0).
