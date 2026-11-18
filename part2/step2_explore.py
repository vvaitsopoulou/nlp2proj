"""
NLP1 Project 2: what do the embeddings capture, for NLI specifically.

Four diagnostics, each tied to a prediction we commit to before running:
  A. nearest neighbours of cue words         -> NeighbourIndex
  B. cosine by lexical relation (WordNet)    -> build_relation_pairs, cosine_by_relation
  C. relation probe, split BY WORD not pair  -> build_probe_dataset, word_group_split, run_probe
  D. sentence-level geometry by gold label   -> sentence_diagnostics
  + a compact gender-bias projection         -> gender_direction, project_on_direction

Relations matter for NLI: synonymy/hypernymy support entailment, antonymy and
co-hyponymy (dog vs cat) support contradiction. The guiding question is whether
the geometry already separates these, or whether a classifier has to learn it.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

# cosine + nearest neighbours (reuses the tutorial's idea, vectorised)
def cosine(u, v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


# Light loader for the runner-up embedding (e.g. 840B): scan the file once and
# keep vectors ONLY for the words we need, so the probe comparison costs a
# few thousand vectors instead of a 2 GB matrix. 
class _DictEmb:
    def __init__(self, vecs, num_dims):
        self._v = vecs
        self.num_dims = num_dims
        self._unk = np.zeros(num_dims, dtype=np.float32)

    def __contains__(self, w):
        return w.lower() in self._v

    def embed_word(self, w):
        return self._v.get(w.lower(), self._unk)


def load_subset_embeddings(path, words, num_dims, skip_first_line=False, encoding="utf-8"):
    from step1_embeddings import _iter_lines
    want = {w.lower() for w in words}
    found = {}
    for i, line in enumerate(_iter_lines(path, encoding)):
        if skip_first_line and i == 0:
            continue
        parts = line.split()
        if len(parts) < num_dims + 1:
            continue
        w = parts[0].lower()
        if w not in want or w in found:
            continue
        values = parts[1:]
        if len(values) != num_dims:
            continue
        found[w] = np.asarray(values, dtype=np.float32)
        if len(found) == len(want):
            break
    return _DictEmb(found, num_dims)


class NeighbourIndex:
    """Precompute unit vectors once; query nearest neighbours by cosine.
    restrict_words limits neighbours to a set (e.g. SNLI vocab) for relevance."""

    def __init__(self, emb_model, restrict_words=None):
        V = emb_model.vectors
        if restrict_words is None:
            idxs = list(range(2, V.shape[0]))           # skip pad(0), unk(1)
        else:
            idxs = sorted({emb_model.vocab.index(w) for w in restrict_words
                           if w in emb_model})
            idxs = [i for i in idxs if i >= 2]
        self.idxs = np.asarray(idxs)
        sub = V[self.idxs].astype(np.float32)
        norms = np.linalg.norm(sub, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.unit = sub / norms
        self.words = [emb_model.vocab.word(int(i)) for i in self.idxs]
        self.emb = emb_model

    def neighbours(self, word, k=10):
        if word not in self.emb:
            return []
        q = np.asarray(self.emb.embed_word(word), dtype=np.float32)
        n = np.linalg.norm(q)
        if n == 0:
            return []
        sims = self.unit @ (q / n)
        order = np.argsort(-sims)
        out = []
        for j in order:
            w = self.words[j]
            if w == word:
                continue
            out.append((w, float(sims[j])))
            if len(out) >= k:
                break
        return out


# B. WordNet relation pairs, restricted to our embedding + SNLI vocab.
# Multiword lemmas (with '_') are skipped since our vectors are single-token.
# A pair appearing in more than one relation is dropped, to keep classes clean.
def build_relation_pairs(target_words, emb_model, snli_vocab,
                         max_per_relation=500, per_word_cap=5, seed=0):
    from nltk.corpus import wordnet as wn
    rng = random.Random(seed)

    def ok(w):
        return (isinstance(w, str) and w.isalpha() and w == w.lower()
                and w in emb_model and w in snli_vocab)

    targets = [w for w in dict.fromkeys(target_words) if ok(w)]
    rels = {"synonym": set(), "hypernym": set(), "antonym": set(), "co_hyponym": set()}

    for w in targets:
        caps = {k: 0 for k in rels}
        for s in wn.synsets(w):
            for l in s.lemmas():
                n = l.name().lower()
                if n != w and "_" not in n and ok(n) and caps["synonym"] < per_word_cap:
                    rels["synonym"].add(frozenset((w, n))); caps["synonym"] += 1
                for a in l.antonyms():
                    an = a.name().lower()
                    if "_" not in an and ok(an) and caps["antonym"] < per_word_cap:
                        rels["antonym"].add(frozenset((w, an))); caps["antonym"] += 1
            for hs in s.hypernyms():
                for l in hs.lemmas():
                    n = l.name().lower()
                    if n != w and "_" not in n and ok(n) and caps["hypernym"] < per_word_cap:
                        rels["hypernym"].add(frozenset((w, n))); caps["hypernym"] += 1
                for cs in hs.hyponyms():
                    if cs == s:
                        continue
                    for l in cs.lemmas():
                        n = l.name().lower()
                        if n != w and "_" not in n and ok(n) and caps["co_hyponym"] < per_word_cap:
                            rels["co_hyponym"].add(frozenset((w, n))); caps["co_hyponym"] += 1

    # drop pairs that landed in more than one relation (ambiguous labels)
    seen = Counter()
    for s in rels.values():
        seen.update(s)
    ambiguous = {p for p, c in seen.items() if c > 1}
    for k in rels:
        rels[k] -= ambiguous

    # random / unrelated pairs, excluding any real relation pair
    real = set().union(*rels.values())
    pool = [w for w in targets]
    rand = set()
    attempts = 0
    while len(rand) < max_per_relation and attempts < max_per_relation * 50:
        a, b = rng.sample(pool, 2)
        attempts += 1
        fs = frozenset((a, b))
        if a != b and fs not in real and fs not in rand:
            rand.add(fs)
    rels["random"] = rand

    # frozensets -> ordered tuples, shuffle, cap
    out = {}
    for k, s in rels.items():
        pairs = [tuple(p) if len(p) == 2 else (next(iter(p)), next(iter(p))) for p in s]
        rng.shuffle(pairs)
        out[k] = pairs[:max_per_relation]
    return out


def cosine_by_relation(pairs_by_rel, emb_model):
    out = {}
    for rel, pairs in pairs_by_rel.items():
        cs = [cosine(emb_model.embed_word(a), emb_model.embed_word(b)) for a, b in pairs]
        out[rel] = np.asarray(cs, dtype=np.float32)
    return out


def summarize_relations(cos_by_rel):
    rows = []
    for rel, arr in cos_by_rel.items():
        rows.append({"relation": rel, "n": len(arr),
                     "mean_cos": round(float(arr.mean()), 4) if len(arr) else None,
                     "median_cos": round(float(np.median(arr)), 4) if len(arr) else None,
                     "std": round(float(arr.std()), 4) if len(arr) else None})
    return rows


# C. Relation probe. Features = [u, v, |u-v|, u*v]; cosine-only baseline.
# Split BY WORD so no word is shared between train and test (no memorisation).
def pair_features(a, b, emb_model):
    u = np.asarray(emb_model.embed_word(a), dtype=np.float32)
    v = np.asarray(emb_model.embed_word(b), dtype=np.float32)
    return np.concatenate([u, v, np.abs(u - v), u * v])


def build_probe_dataset(pairs_by_rel, emb_model):
    rels = sorted(pairs_by_rel)
    X_full, X_cos, y, pair_words = [], [], [], []
    for ri, rel in enumerate(rels):
        for a, b in pairs_by_rel[rel]:
            X_full.append(pair_features(a, b, emb_model))
            X_cos.append([cosine(emb_model.embed_word(a), emb_model.embed_word(b))])
            y.append(ri)
            pair_words.append((a, b))
    return (np.asarray(X_full, dtype=np.float32), np.asarray(X_cos, dtype=np.float32),
            np.asarray(y), pair_words, rels)


def word_group_split(pair_words, seed=0, test_frac=0.3):
    """Partition unique words into train/test groups; a pair is train only if
    BOTH words are train words, test only if BOTH are test words; the rest are
    dropped. Guarantees zero word overlap between the two sets."""
    uniq = sorted({w for pair in pair_words for w in pair})
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_test = int(len(uniq) * test_frac)
    test_w = set(uniq[:n_test])
    train_w = set(uniq[n_test:])
    train_idx, test_idx, dropped = [], [], 0
    for i, (a, b) in enumerate(pair_words):
        if a in train_w and b in train_w:
            train_idx.append(i)
        elif a in test_w and b in test_w:
            test_idx.append(i)
        else:
            dropped += 1
    return np.asarray(train_idx), np.asarray(test_idx), dropped


def run_probe(X, y, train_idx, test_idx):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    return {
        "accuracy": round(float(accuracy_score(y[test_idx], pred)), 4),
        "macro_f1": round(float(f1_score(y[test_idx], pred, average="macro")), 4),
        "confusion": confusion_matrix(y[test_idx], pred),
    }


# D. Sentence-level geometry by gold label, no training. Covered tokens only,
# so the numbers reflect real word vectors, not the unknown-mean fallback.
def mean_pool(tokens, emb_model):
    vecs = [emb_model.embed_word(t) for t in tokens if t in emb_model]
    if not vecs:
        return None
    return np.mean(np.stack(vecs), axis=0)


def sentence_cosine(p_tokens, h_tokens, emb_model):
    pp = mean_pool(p_tokens, emb_model)
    hp = mean_pool(h_tokens, emb_model)
    if pp is None or hp is None:
        return None
    return cosine(pp, hp)


def soft_overlap(p_tokens, h_tokens, emb_model):
    """Mean over hypothesis tokens of the best cosine match in the premise."""
    pv = [emb_model.embed_word(t) for t in p_tokens if t in emb_model]
    if not pv:
        return None
    P = np.stack(pv).astype(np.float32)
    P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    sims = []
    for t in h_tokens:
        if t in emb_model:
            q = np.asarray(emb_model.embed_word(t), dtype=np.float32)
            q /= (np.linalg.norm(q) + 1e-9)
            sims.append(float(np.max(P @ q)))
    if not sims:
        return None
    return float(np.mean(sims))


def sentence_diagnostics(pairs, token_cache, emb_model, max_items=None):
    cos_by = defaultdict(list)
    soft_by = defaultdict(list)
    for (p, h), lab in (pairs[:max_items] if max_items else pairs):
        c = sentence_cosine(token_cache[p], token_cache[h], emb_model)
        s = soft_overlap(token_cache[p], token_cache[h], emb_model)
        if c is not None:
            cos_by[lab].append(c)
        if s is not None:
            soft_by[lab].append(s)
    return ({k: np.asarray(v) for k, v in cos_by.items()},
            {k: np.asarray(v) for k, v in soft_by.items()})


# Compact gender-bias projection (reuses the tutorial's method).
def gender_direction(emb_model,
                     pairs=(("he", "she"), ("man", "woman"),
                            ("boy", "girl"), ("male", "female"))):
    diffs = []
    for a, b in pairs:
        if a in emb_model and b in emb_model:
            d = np.asarray(emb_model.embed_word(a), dtype=np.float32) - \
                np.asarray(emb_model.embed_word(b), dtype=np.float32)
            diffs.append(d / (np.linalg.norm(d) + 1e-9))
    if not diffs:
        return None
    g = np.mean(diffs, axis=0)
    return g / (np.linalg.norm(g) + 1e-9)


def project_on_direction(words, direction, emb_model):
    scored = []
    for w in words:
        if w in emb_model:
            v = np.asarray(emb_model.embed_word(w), dtype=np.float32)
            scored.append((w, float(np.dot(v / (np.linalg.norm(v) + 1e-9), direction))))
    scored.sort(key=lambda x: x[1])
    return scored


# helper: frequent SNLI content words to seed WordNet and the neighbour list
def frequent_content_words(pairs, token_cache, emb_model, stop_set, n=2000, min_len=3):
    cnt = Counter()
    for (p, h), _ in pairs:
        for s in (p, h):
            for t in token_cache[s]:
                if (t.isalpha() and len(t) >= min_len and t not in stop_set
                        and t in emb_model):
                    cnt[t] += 1
    return [w for w, _ in cnt.most_common(n)]
