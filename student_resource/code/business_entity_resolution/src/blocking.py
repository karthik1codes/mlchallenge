"""Multi-key blocking / candidate generation (parallel + tight buckets)."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .config import (
    BLOCK_RARE_TOKEN_MAX_DF,
    MAX_BLOCK_BUCKET,
    MAX_CANDIDATES_PER_S1,
    N_JOBS,
)
from .features import cheap_pre_score
from .normalize import enrich_row


Record = Dict


def enrich_dataframe(df) -> List[Record]:
    records: List[Record] = []
    for row in df.itertuples(index=False):
        e = enrich_row(row.business_name, row.business_address, row.country)
        e["entity_id"] = row.entity_id
        records.append(e)
    return records


def load_and_enrich(path: str, nrows: Optional[int] = None) -> List[Record]:
    import pandas as pd

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=nrows)
    return enrich_dataframe(df)


def load_linked_smoke_sample(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    gt_path: str,
    n_s1: int,
    extra_negatives: int = 5000,
    cache_dir: Optional[str] = None,
) -> Tuple[List[Record], List[Record], Dict[str, List[str]]]:
    """Load a small S1 slice plus its true S2/S3 matches (and distractors)."""
    import pickle
    from pathlib import Path

    import pandas as pd

    from .metrics import parse_id_list

    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"linked_smoke_s1{n_s1}_neg{extra_negatives}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False, nrows=n_s1)
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    gt_map = {
        r.source1_entity_id: parse_id_list(r.matched_entity_ids)
        for r in gt_df.itertuples(index=False)
    }
    needed: Set[str] = set()
    gt_sub: Dict[str, List[str]] = {}
    for eid in s1_df["entity_id"]:
        mids = gt_map.get(eid, [])
        gt_sub[eid] = mids
        needed.update(mids)

    empty_cols = ["entity_id", "business_name", "business_address", "country"]

    def _collect(path: str, extra_n: int) -> pd.DataFrame:
        hit_parts = []
        extra_parts = []
        extra_count = 0
        found: Set[str] = set()
        for chunk in pd.read_csv(
            path, sep="\t", dtype=str, keep_default_na=False, chunksize=300_000
        ):
            if needed:
                miss = needed - found
                if miss:
                    hit = chunk[chunk["entity_id"].isin(miss)]
                    if len(hit):
                        hit_parts.append(hit)
                        found.update(hit["entity_id"].tolist())
            if extra_count < extra_n:
                extra = chunk[~chunk["entity_id"].isin(needed)]
                take = extra_n - extra_count
                if len(extra):
                    part = extra.head(take)
                    extra_parts.append(part)
                    extra_count += len(part)
            if len(found) >= len(needed) and extra_count >= extra_n:
                break
        parts = hit_parts + extra_parts
        if not parts:
            return pd.DataFrame(columns=empty_cols)
        return pd.concat(parts, ignore_index=True)

    per_source_extra = max(500, extra_negatives // 2)
    s23_df = pd.concat(
        [_collect(s2_path, per_source_extra), _collect(s3_path, per_source_extra)],
        ignore_index=True,
    ).drop_duplicates(subset=["entity_id"])

    result = (enrich_dataframe(s1_df), enrich_dataframe(s23_df), gt_sub)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def _add(index: Dict[str, List[int]], key: str, idx: int) -> None:
    if key:
        index[key].append(idx)


def build_block_indexes(candidates: Sequence[Record]) -> Dict:
    """Build inverted indexes over S2+S3 records."""
    by_core: Dict[str, List[int]] = defaultdict(list)
    by_sorted: Dict[str, List[int]] = defaultdict(list)
    by_phon: Dict[str, List[int]] = defaultdict(list)
    by_zip_name: Dict[str, List[int]] = defaultdict(list)
    token_df: Dict[str, int] = defaultdict(int)

    for i, rec in enumerate(candidates):
        country = rec["country"].lower()
        _add(by_core, f"{country}||{rec['core_name']}", i)
        _add(by_sorted, f"{country}||{rec['sorted_name']}", i)
        _add(by_phon, f"{country}||{rec['phonetic']}", i)
        if rec["zip"] and rec["name_tokens"]:
            _add(by_zip_name, f"{country}||{rec['zip']}||{rec['name_tokens'][0]}", i)
        for tok in rec["name_token_set"]:
            if len(tok) >= 4:  # rarer, fewer mega-buckets
                token_df[tok] += 1

    by_rare: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(candidates):
        country = rec["country"].lower()
        for tok in rec["name_token_set"]:
            if len(tok) >= 4 and token_df[tok] <= BLOCK_RARE_TOKEN_MAX_DF:
                _add(by_rare, f"{country}||{tok}", i)

    return {
        "core": by_core,
        "sorted": by_sorted,
        "phon": by_phon,
        "zip_name": by_zip_name,
        "rare": by_rare,
        "token_df": token_df,
    }


def candidate_indices_for(
    s1: Record,
    indexes: Dict,
    pool: Sequence[Record],
    max_candidates: int = MAX_CANDIDATES_PER_S1,
    max_bucket: int = MAX_BLOCK_BUCKET,
) -> List[int]:
    country = s1["country"].lower()
    hits: Set[int] = set()

    keys = [
        ("core", f"{country}||{s1['core_name']}"),
        ("sorted", f"{country}||{s1['sorted_name']}"),
        ("phon", f"{country}||{s1['phonetic']}"),
    ]
    if s1["zip"] and s1["name_tokens"]:
        keys.append(("zip_name", f"{country}||{s1['zip']}||{s1['name_tokens'][0]}"))

    for idx_name, key in keys:
        bucket = indexes[idx_name].get(key)
        if bucket and len(bucket) <= max_bucket:
            hits.update(bucket)

    token_df = indexes["token_df"]
    rare = indexes["rare"]
    for tok in s1["name_token_set"]:
        if len(tok) >= 4 and token_df.get(tok, 10**9) <= BLOCK_RARE_TOKEN_MAX_DF:
            bucket = rare.get(f"{country}||{tok}")
            if bucket and len(bucket) <= max_bucket:
                hits.update(bucket)

    if not hits:
        return []

    # Cap raw hits before expensive ranking
    if len(hits) > max_candidates * 8:
        # Prefer exact core / zip overlap without full sort of huge sets
        exact = [
            i
            for i in hits
            if pool[i]["core_name"] == s1["core_name"]
            or (s1["zip"] and pool[i]["zip"] == s1["zip"])
        ]
        if len(exact) >= max_candidates:
            hits = set(exact)
        else:
            # keep exact + sample of rest via cheap score on limited set
            rest = list(hits - set(exact))
            rest.sort(key=lambda i: cheap_pre_score(s1, pool[i]), reverse=True)
            hits = set(exact) | set(rest[: max_candidates * 4])

    scored = [(cheap_pre_score(s1, pool[i]), i) for i in hits]
    scored.sort(reverse=True)
    return [i for _, i in scored[:max_candidates]]


def _ids_for_s1(args) -> Tuple[str, List[str]]:
    s1, indexes, pool, max_candidates = args
    idxs = candidate_indices_for(s1, indexes, pool, max_candidates)
    ids: List[str] = []
    seen = set()
    for i in idxs:
        eid = pool[i]["entity_id"]
        if eid not in seen:
            seen.add(eid)
            ids.append(eid)
    return s1["entity_id"], ids


def generate_candidates(
    s1_records: Sequence[Record],
    s23_records: Sequence[Record],
    max_candidates: int = MAX_CANDIDATES_PER_S1,
    indexes: Optional[Dict] = None,
    n_jobs: int = N_JOBS,
) -> Dict[str, List[str]]:
    """Return mapping S1 entity_id -> ranked S2/S3 candidate entity_ids."""
    if indexes is None:
        indexes = build_block_indexes(s23_records)

    if n_jobs <= 1 or len(s1_records) < 200:
        out: Dict[str, List[str]] = {}
        for s1 in s1_records:
            sid, ids = _ids_for_s1((s1, indexes, s23_records, max_candidates))
            out[sid] = ids
        return out

    payloads = [(s1, indexes, s23_records, max_candidates) for s1 in s1_records]
    out = {}
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        for sid, ids in ex.map(_ids_for_s1, payloads, chunksize=64):
            out[sid] = ids
    return out
