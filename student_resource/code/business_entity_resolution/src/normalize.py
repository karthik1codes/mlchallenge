"""Text normalization for noisy business names and addresses."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from .config import PHONETIC_PREFIX_TOKENS, USE_METAPHONE

LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "llc",
    "llp",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "co",
    "company",
    "plc",
    "pvt",
    "private",
    "pc",
    "p.c",
    "lp",
    "gmbh",
    "sarl",
    "sa",
    "sas",
    "bv",
    "nv",
}

ABBREV = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "hwy": "highway",
    "pkwy": "parkway",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "dept": "department",
    "intl": "international",
    "natl": "national",
    "mgmt": "management",
    "svc": "service",
    "svcs": "services",
    "mfg": "manufacturing",
    "tech": "technology",
    "ctr": "center",
    "centre": "center",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
}

PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
MULTI_SPACE = re.compile(r"\s+")
ZIP_US = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
PIN_IN = re.compile(r"\b(\d{6})\b")
STREET_NUM = re.compile(r"\b(\d{1,6})\b")
ANY_ZIP5 = re.compile(r"\b(\d{5})\b")


def _fold(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def normalize_text(text: str) -> str:
    text = _fold(text)
    text = text.replace("&", " and ")
    text = PUNCT_RE.sub(" ", text)
    text = MULTI_SPACE.sub(" ", text).strip()
    if not text:
        return ""
    return " ".join(ABBREV.get(tok, tok) for tok in text.split())


def _strip_legal(toks: list) -> list:
    while toks and toks[-1] in LEGAL_SUFFIXES:
        toks.pop()
    while toks and toks[0] in LEGAL_SUFFIXES:
        toks.pop(0)
    return toks


def extract_zip(address: str, country: str = "") -> str:
    addr = address if isinstance(address, str) else ""
    country = (country or "").strip().lower()
    if country in {"india", "in"}:
        m = PIN_IN.search(addr)
        return m.group(1) if m else ""
    m = ZIP_US.search(addr)
    if m:
        return m.group(1)
    m = ANY_ZIP5.search(addr)
    return m.group(1) if m else ""


@lru_cache(maxsize=1)
def _metaphone_ready():
    try:
        from metaphone import doublemetaphone

        return doublemetaphone
    except ImportError:
        return None


def phonetic_key_from_tokens(toks: tuple, n_tokens: int = PHONETIC_PREFIX_TOKENS) -> str:
    use = toks[:n_tokens]
    if not use:
        return ""
    if not USE_METAPHONE:
        return "|".join(t[:4] for t in use)
    fn = _metaphone_ready()
    if fn is None:
        return "|".join(t[:4] for t in use)
    return "|".join((fn(t)[0] or t[:4]) for t in use)


def _char_ngrams(text: str, n: int = 3) -> frozenset:
    text = f" {text} "
    if len(text) < n:
        return frozenset({text}) if text.strip() else frozenset()
    return frozenset(text[i : i + n] for i in range(len(text) - n + 1))


def enrich_row(name: str, address: str, country: str) -> dict:
    """Single-pass normalize + precomputed sets for fast blocking/features."""
    n_name = normalize_text(name)
    n_addr = normalize_text(address)
    name_toks = tuple(_strip_legal(n_name.split()))
    addr_toks = tuple(n_addr.split())
    c_name = " ".join(name_toks)
    sorted_name = " ".join(sorted(name_toks))
    street_no = ""
    if addr_toks:
        m = STREET_NUM.search(n_addr)
        street_no = m.group(1) if m else ""

    return {
        "name_norm": n_name,
        "addr_norm": n_addr,
        "core_name": c_name,
        "sorted_name": sorted_name,
        "phonetic": phonetic_key_from_tokens(name_toks),
        "zip": extract_zip(address, country),
        "street_no": street_no,
        "country": (country or "").strip(),
        "name_tokens": name_toks,
        "addr_tokens": addr_toks,
        "name_token_set": frozenset(name_toks),
        "addr_token_set": frozenset(addr_toks),
        "name_ngrams": _char_ngrams(c_name, 3),
        "addr_ngrams": _char_ngrams(n_addr, 3),
    }
