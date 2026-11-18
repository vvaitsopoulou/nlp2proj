"""
NLP1 Project 2 — shared infrastructure (Step 0 deliverable).

Everyone on the team imports this so notebooks stay short and every run is
reproducible and comparable with Project 1.

Design choice: this module imports neither the SNLI dataset nor part1. The
caller passes in the loaded `snli` splits and part1's `cross_validation_split`
callable. That keeps the module importable anywhere (and unit-testable without
a GPU, SNLI or part1).

Covers Step 0 items:
  1. rebuild working set + CV folds   -> clean_training_pool, build_working_set
  2. verify rebuild against the grid  -> verify_against_grid
  3. clean validation + test splits   -> clean_eval_split
  4. token-id cache                   -> build_id_cache, sentences_in, CachedTokenizer
  5. seeding                          -> set_all_seeds, run_seed
  6. per-run JSON log                 -> log_run
  7. results layout + checkpoint      -> run_folder, Checkpoint
  8. timing helper                    -> Timer
"""
from __future__ import annotations

import json
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Shared constants — MUST be identical across everyone's version (see plan).
# --------------------------------------------------------------------------
WORKING_SET_SIZE = 40_000
SUBSAMPLE_SEED = 0            # the stratified-subsample seed from Project 1
CV_SEEDS = [0, 3, 5]
NUM_FOLDS = 5
ROPE = 0.005
LABEL_NAMES = ["entailment", "neutral", "contradiction"]   # ids 0, 1, 2


# --------------------------------------------------------------------------
# Item 1 — rebuild the training pool and working set, VERBATIM from Project 1.
# The iteration order and filter order are kept identical, because the
# stratified subsample is only reproducible if the input list order matches.
# --------------------------------------------------------------------------
def clean_training_pool(snli_train):
    """Project 1's two-stage training filter. Returns (train_pairs_all, stats)."""
    # Stage 1: drop no-gold (-1) and identical-pair mislabels (p==h but label!=entailment).
    base_pool = [
        ((ex["premise"], ex["hypothesis"]), ex["label"])
        for ex in snli_train
        if ex["label"] != -1
        and not (ex["premise"] == ex["hypothesis"] and ex["label"] != 0)
    ]

    # Stage 2: drop one-word hypotheses and cross-row conflicting-label inputs.
    labels_by_input = defaultdict(set)
    for (p, h), lab in base_pool:
        labels_by_input[(p, h)].add(lab)
    conflicting = {ph for ph, labs in labels_by_input.items() if len(labs) > 1}

    dropped_one_word = dropped_conflict = 0
    train_pairs_all = []
    for (p, h), lab in base_pool:
        if len(h.split()) == 1:
            dropped_one_word += 1
            continue
        if (p, h) in conflicting:
            dropped_conflict += 1
            continue
        train_pairs_all.append(((p, h), lab))

    stats = {
        "base_pool": len(base_pool),
        "dropped_one_word": dropped_one_word,
        "dropped_conflict": dropped_conflict,
        "final_pool": len(train_pairs_all),
    }
    return train_pairs_all, stats


def build_working_set(train_pairs_all):
    """Stratified 40k subsample, same seed as Project 1 -> identical working set."""
    from sklearn.model_selection import train_test_split

    X = [ph for ph, _ in train_pairs_all]
    y = [lab for _, lab in train_pairs_all]
    X_work, _, y_work, _ = train_test_split(
        X, y,
        train_size=WORKING_SET_SIZE,
        stratify=y,
        random_state=SUBSAMPLE_SEED,
    )
    return list(zip(X_work, y_work))


# --------------------------------------------------------------------------
# Item 3 — clean the evaluation splits. Stage 1 ONLY, exactly as Project 1
# applied to its test split: we remove provable label noise but change nothing
# else, so dev/test stay comparable to the published splits.
# --------------------------------------------------------------------------
def clean_eval_split(snli_split):
    return [
        ((ex["premise"], ex["hypothesis"]), ex["label"])
        for ex in snli_split
        if ex["label"] != -1
        and not (ex["premise"] == ex["hypothesis"] and ex["label"] != 0)
    ]


# --------------------------------------------------------------------------
# Item 2 — verify the rebuild reproduces a Project 1 result. We pool the stored
# best-epoch predictions of one system over seeds x folds, rebuild the golds
# from freshly recomputed CV splits, and check the confusion-matrix accuracy
# equals the stored grid accuracy. A mismatch means the rebuild does not match
# the pickle's cleaning/splits.
# --------------------------------------------------------------------------
def verify_against_grid(grid, cross_validation_split, working_set, system_key,
                        seeds=CV_SEEDS, num_folds=NUM_FOLDS, tol=1e-3):
    from sklearn.metrics import confusion_matrix

    cv = {s: cross_validation_split(working_set, k=num_folds, seed=s) for s in seeds}

    preds_all, golds_all = [], []
    for si, seed in enumerate(seeds):
        run = grid[system_key][si]
        for fold in cv[seed]:
            preds = run["predictions"][fold]
            golds = [lab for _, lab in cv[seed][fold]["validation"]]
            assert len(preds) == len(golds), \
                f"size misalign: {system_key} seed {seed} fold {fold}"
            preds_all += [int(p) for p in preds]
            golds_all += [int(g) for g in golds]

    cm = confusion_matrix(golds_all, preds_all, labels=[0, 1, 2])
    cm_acc = float(np.trace(cm) / cm.sum())
    grid_acc = float(np.mean([list(r["accuracies"].values()) for r in grid[system_key]]))
    return {
        "system": system_key,
        "cm": cm,
        "cm_acc": cm_acc,
        "grid_acc": grid_acc,
        "match": abs(cm_acc - grid_acc) <= tol,
        "n": len(golds_all),
    }


# --------------------------------------------------------------------------
# Item 4 — token-id cache. Tokenize each UNIQUE sentence once; reuse everywhere.
# SNLI repeats premises across ~3 hypotheses, so this is a large saving at
# 549k. CachedTokenizer is a drop-in for part2's tokenizer (same .ids/.tokens).
# --------------------------------------------------------------------------
def sentences_in(*pair_lists):
    """All unique premise/hypothesis strings across one or more pair lists."""
    out = set()
    for pairs in pair_lists:
        for (p, h), _ in pairs:
            out.add(p)
            out.add(h)
    return out


def build_id_cache(sentences, tokenizer):
    """string -> token-id list, one tokenization per unique sentence."""
    cache = {}
    for s in sentences:
        if s not in cache:
            cache[s] = tokenizer.ids(s)
    return cache


class CachedTokenizer:
    """Reads ids from a prebuilt cache; falls back to the real tokenizer for
    any unseen string (so dev/test strings not in the cache still work)."""

    def __init__(self, cache, fallback):
        self.cache = cache
        self.fallback = fallback
        self.vocab = getattr(fallback, "vocab", None)

    def ids(self, text):
        v = self.cache.get(text)
        return v if v is not None else self.fallback.ids(text)

    def tokens(self, text):
        return self.fallback.tokens(text)


# --------------------------------------------------------------------------
# Item 5 — seeding. part2.experiment.train sets no seed, so the custom CV loop
# (Step 3) calls set_all_seeds(run_seed(cv_seed, fold)) before each fold.
# Note: this fixes init + data-shuffling RNGs, which is what the variance study
# needs. Full cudnn bitwise determinism (torch.use_deterministic_algorithms)
# is left off: it can error on LSTM ops and slow training, and we don't need it.
# --------------------------------------------------------------------------
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_seed(cv_seed, fold):
    """Distinct, reproducible per-run seed, kept separate from the split seed."""
    return 100 * cv_seed + fold


# --------------------------------------------------------------------------
# Item 6 — one JSON log per run. Predictions/labels are stripped from the
# per-epoch metrics so the file stays small (they live in the checkpoint).
# --------------------------------------------------------------------------
def _slim_metrics(metrics):
    slim = []
    for m in metrics:
        tm = {k: v for k, v in m["train_metrics"].items() if k in ("loss", "accuracy")}
        vm = {k: v for k, v in m["val_metrics"].items() if k in ("loss", "accuracy")}
        slim.append({"train_metrics": tm, "val_metrics": vm})
    return slim


def log_run(save_to, config, metrics, best_epoch, wall_time_sec):
    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    record = {
        "config": config,
        "seeds": {k: config[k] for k in ("cv_seed", "fold", "run_seed") if k in config},
        "metrics": _slim_metrics(metrics),
        "best_epoch": int(best_epoch),
        "wall_time_sec": float(wall_time_sec),
    }
    with open(f"{save_to}.run.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


# --------------------------------------------------------------------------
# Item 7 — results layout (same tree as Project 1) + resumable checkpoint.
# --------------------------------------------------------------------------
def run_folder(results_folder, system_key, seed):
    return f"{results_folder}/{system_key}/seed-{seed}"


class Checkpoint:
    """Resume-aware dict backed by a pickle. `in`, get, and set that persists."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = {}
        if self.path.exists():
            with open(self.path, "rb") as f:
                self.data = pickle.load(f)

    def __contains__(self, key):
        return key in self.data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def keys(self):
        return list(self.data.keys())


# --------------------------------------------------------------------------
# Item 8 — timing helper (feeds the per-run log and the budget rescale).
# --------------------------------------------------------------------------
class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.sec = time.time() - self.t0
        return False
