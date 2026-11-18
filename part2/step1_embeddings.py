"""
NLP1 Project 2 : embedding-model evidence.

Goal: choose the embedding model by measurable criteria, not by habit, and keep
the evidence so the choice is defensible.

Efficiency note: the string-token tokenization
is the SAME for every candidate (it depends only on NLTK + lowercasing).
So we tokenize each unique sentence ONCE, then only swap
the vocabulary set per candidate. And coverage needs only the embedding's
vocabulary, not its matrix, so we scan the vocab cheaply and defer the slow
full-matrix load to the single winner.
"""
from __future__ import annotations

import gzip
import zipfile
from collections import Counter
from io import TextIOWrapper


# Read an embedding file's VOCABULARY only (no matrix), replicating the loader
# in part2.clslib.PretrainedEmbeddings so the measured vocab is exactly what the
# model would expose lowercased, first occurrence wins, wrong-dim lines skipped.
def _iter_lines(path, encoding="utf-8"):
    if path.endswith(".txt"):
        with open(path, "r", encoding=encoding) as f:
            for line in f:
                yield line
    elif path.endswith(".gz"):
        with gzip.open(path, "rt", encoding=encoding) as f:
            for line in f:
                yield line
    elif path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            members = z.namelist()
            if len(members) != 1:
                raise ValueError(f"expected one member in {path}, found {members}")
            with z.open(members[0]) as raw, TextIOWrapper(raw, encoding=encoding) as f:
                for line in f:
                    yield line
    else:
        raise ValueError(f"unknown format: {path}")


def read_embedding_vocab(path, num_dims, skip_first_line=False, encoding="utf-8"):
    """Set of lowercased words the model would actually expose."""
    vocab = set()
    for i, line in enumerate(_iter_lines(path, encoding)):
        if skip_first_line and i == 0:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        word = parts[0].lower()
        values = parts[1:]
        if len(values) != num_dims:      # same guard as the loader
            continue
        if word in vocab:                # first occurrence wins
            continue
        vocab.add(word)
    return vocab


# Tokenize once (candidate-independent), then measure coverage per vocab.
def build_token_cache(sentences, tokenizer):
    """sentence -> list of string tokens (uses tokenizer.tokens, not .ids)."""
    return {s: tokenizer.tokens(s) for s in sentences}


def _side_token_lists(pairs, token_cache, side):
    out = []
    for (p, h), _ in pairs:
        if side in ("premise", "both"):
            out.append(token_cache[p])
        if side in ("hypothesis", "both"):
            out.append(token_cache[h])
    return out


def coverage_from_token_lists(token_lists, vocab_set):
    types = Counter()
    oov_tokens = 0
    n_tokens = 0
    for toks in token_lists:
        for t in toks:
            n_tokens += 1
            types[t] += 1
            if t not in vocab_set:
                oov_tokens += 1
    oov_types = {t for t in types if t not in vocab_set}
    n_types = len(types)
    return {
        "n_tokens": n_tokens,
        "oov_tokens": oov_tokens,
        "token_cov": (1 - oov_tokens / n_tokens) if n_tokens else 0.0,
        "n_types": n_types,
        "oov_types": len(oov_types),
        "type_cov": (1 - len(oov_types) / n_types) if n_types else 0.0,
        "oov_counter": Counter({t: types[t] for t in oov_types}),
    }


def coverage_table(splits, token_cache, candidate_vocabs, sides=("premise", "hypothesis", "both")):
    """splits: {split_name: pairs}. candidate_vocabs: {name: vocab_set}.
    Returns a list of row dicts, one per (candidate, split, side)."""
    rows = []
    for cand, vocab in candidate_vocabs.items():
        for split_name, pairs in splits.items():
            for side in sides:
                tl = _side_token_lists(pairs, token_cache, side)
                c = coverage_from_token_lists(tl, vocab)
                rows.append({
                    "embedding": cand,
                    "split": split_name,
                    "side": side,
                    "token_cov": round(c["token_cov"], 4),
                    "type_cov": round(c["type_cov"], 4),
                    "n_tokens": c["n_tokens"],
                    "oov_tokens": c["oov_tokens"],
                })
    return rows


# Out-of-vocabulary inspection: the evidence behind the coverage number.
def categorize_oov(token):
    if any(ch.isdigit() for ch in token):
        return "numeric"
    if not token.isalpha():
        return "nonalpha"      # hyphen, leftover apostrophe, slash, punctuation
    if len(token) <= 2:
        return "short"
    return "alpha"             # likely typo or genuinely rare word


def top_oov(pairs, token_cache, vocab_set, side="hypothesis", k=50):
    tl = _side_token_lists(pairs, token_cache, side)
    cov = coverage_from_token_lists(tl, vocab_set)
    common = cov["oov_counter"].most_common(k)
    by_cat = Counter(categorize_oov(t) for t in cov["oov_counter"])
    return {"top": common, "by_category": dict(by_cat), "n_oov_types": cov["oov_types"]}


def tokenizer_probe(tokenizer, vocab_set, examples):
    """Show how tricky strings tokenize and which pieces have a vector."""
    report = []
    for ex in examples:
        toks = tokenizer.tokens(ex)
        hits = [(t, t in vocab_set) for t in toks]
        report.append({"text": ex, "tokens": toks,
                       "oov": [t for t, ok in hits if not ok]})
    return report


# Stopword check: the justification for KEEPING stopwords + punctuation.
CUE_GROUPS = {
    "negation": ["not", "no", "nor", "never", "nobody", "nothing", "none",
                 "without", "cannot", "n't"],
    "quantifier": ["all", "some", "any", "every", "most", "few", "both", "each"],
    "spatial_prep": ["in", "on", "outside", "inside", "near", "behind", "under", "over"],
}


def stopword_report(pairs, token_cache, stop_set, strip_tokenizer):
    """stop_set: nltk english stopwords (a set).
    strip_tokenizer: a tokenizer with remove_punct=True, remove_stopwords=True.
    Returns: which cue words are stopwords, how many hypotheses carry each cue,
    and how many sentences become EMPTY after stopword+punct removal."""
    # which cue words NLTK would strip
    membership = {grp: {w: (w in stop_set) for w in words}
                  for grp, words in CUE_GROUPS.items()}

    # hypotheses containing each cue (kept-all tokenization, lowercased tokens)
    hyp_tokens = [token_cache[h] for (p, h), _ in pairs]
    cue_hyp_counts = {}
    for grp, words in CUE_GROUPS.items():
        for w in words:
            cue_hyp_counts[w] = sum(1 for toks in hyp_tokens if w in toks)

    # sentences emptied by stripping (all-unk / zero-length risk)
    empty_after_strip = 0
    total_sents = 0
    for (p, h), _ in pairs:
        for s in (p, h):
            total_sents += 1
            if len(strip_tokenizer.tokens(s)) == 0:
                empty_after_strip += 1

    return {
        "cue_is_stopword": membership,
        "cue_hypothesis_counts": cue_hyp_counts,
        "n_hypotheses": len(hyp_tokens),
        "empty_after_strip": empty_after_strip,
        "total_sentences": total_sents,
    }
