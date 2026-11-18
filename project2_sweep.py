"""
NLP1 Project 2 — Step 4 (and reused in Step 5): one-axis-at-a-time sweep.

Protocol (from Step 6): fix an ANCHOR; vary ONE axis at a time; screen on seed 0
x 5 folds; a value replaces the anchor only if it beats it by more than the ROPE
under the Benavoli correlated t-test; confirm winners on seeds 3 and 5.

A `spec` is a plain dict bundling everything a run needs:
  emb_key   -> which embedding (e.g. dim 50/100/300)   -> emb_models[emb_key]
  tok_key   -> which tokenizer (keep / nostop / hyphen) -> tokenizers[tok_key]
  enc_cfg, combo, head_cfg, opt_cfg                     -> passed to the engine
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict

import numpy as np


# ANCHOR — literature + Step 3 defaults, fixed before screening.
ANCHOR = {
    "emb_key": 300,
    "tok_key": "keep",
    "enc_cfg": {"type": "pool", "pool": "mean", "freeze": True},
    "combo": "nli",
    "head_cfg": {"hidden1": 256, "hidden2": 256, "dropout": 0.1},   # input_norm defaults True
    "opt_cfg": {"learning_rate": 1e-3, "weight_decay": 1e-4,
                "batch_size": 128, "num_epochs": 20},
}


# RNN_ANCHOR — Step 5. Identical to ANCHOR except the encoder, so that
# "mean-pool-frozen vs LSTM-last-frozen" is a one-variable comparison isolating
# what recurrence adds over pooling. Tutorial defaults: a gated cell (LSTM),
# single final state, frozen embeddings; direction/readout/fine-tune are axes.
RNN_ANCHOR = {
    "emb_key": 300,
    "tok_key": "keep",
    "enc_cfg": {"type": "rnn", "cell": "lstm", "hidden": 256, "bi": False,
                "output": "last", "freeze": True, "word_dropout": 0.0},
    "combo": "nli",
    "head_cfg": {"hidden1": 256, "hidden2": 256, "dropout": 0.1},
    "opt_cfg": {"learning_rate": 1e-3, "weight_decay": 1e-4,
                "batch_size": 128, "num_epochs": 20},
}


def _spec(anchor, **changes):
    """Deep-copy the anchor and set dotted paths, e.g. _spec(A, **{'enc_cfg.pool':'max'})."""
    s = copy.deepcopy(anchor)
    for path, val in changes.items():
        keys = path.split(".")
        d = s
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = val
    return s


def axis(name, anchor=ANCHOR):
    """Return an ordered {label: spec} for one axis; the anchor value is included."""
    A = anchor
    if name == "composition":
        return OrderedDict([
            ("joined",        _spec(A, combo="joined")),
            ("concat",        _spec(A, combo="concat")),
            ("nli [anchor]",  _spec(A, combo="nli")),
        ])
    if name == "pooling":
        return OrderedDict([
            ("mean [anchor]", _spec(A)),
            ("sum",           _spec(A, **{"enc_cfg.pool": "sum"})),
            ("max",           _spec(A, **{"enc_cfg.pool": "max"})),
            ("meanmax",       _spec(A, **{"enc_cfg.pool": "meanmax"})),
        ])
    if name == "embedding_training":
        return OrderedDict([
            ("frozen [anchor]", _spec(A)),
            ("fine-tuned",      _spec(A, **{"enc_cfg.freeze": False, "opt_cfg.emb_lr": 1e-4})),
            ("random init",     _spec(A, **{"enc_cfg.random_init": True})),
        ])
    if name == "embedding_size":
        return OrderedDict([
            ("50d",          _spec(A, emb_key=50)),
            ("100d",         _spec(A, emb_key=100)),
            ("300d [anchor]", _spec(A, emb_key=300)),
        ])
    if name == "stopwords":
        return OrderedDict([
            ("keep [anchor]", _spec(A, tok_key="keep")),
            ("remove",        _spec(A, tok_key="nostop")),
        ])
    if name == "tokenizer_hyphen":
        return OrderedDict([
            ("keep [anchor]", _spec(A, tok_key="keep")),
            ("split hyphens", _spec(A, tok_key="hyphen")),
        ])
    if name == "head_width":
        return OrderedDict([
            ("50",          _spec(A, **{"head_cfg.hidden1": 50,  "head_cfg.hidden2": 50})),
            ("256 [anchor]", _spec(A)),
            ("512",         _spec(A, **{"head_cfg.hidden1": 512, "head_cfg.hidden2": 512})),
        ])
    if name == "head_dropout":
        return OrderedDict([
            ("0.0",          _spec(A, **{"head_cfg.dropout": 0.0})),
            ("0.1 [anchor]", _spec(A)),
            ("0.3",          _spec(A, **{"head_cfg.dropout": 0.3})),
            ("0.5",          _spec(A, **{"head_cfg.dropout": 0.5})),
        ])
    if name == "learning_rate":
        return OrderedDict([
            ("3e-4",          _spec(A, **{"opt_cfg.learning_rate": 3e-4})),
            ("1e-3 [anchor]", _spec(A)),
            ("3e-3",          _spec(A, **{"opt_cfg.learning_rate": 3e-3})),
        ])
    if name == "batch_size":
        return OrderedDict([
            ("128 [anchor]", _spec(A)),
            ("256",          _spec(A, **{"opt_cfg.batch_size": 256})),
        ])
    # ---- recurrent family (Step 5); anchor = RNN_ANCHOR ----
    # Each axis varies ONE recurrent design choice; the anchor value carries the
    # "[anchor]" tag so run_sweep picks it as the ROPE reference. Justifications:
    #   cell      lstm vs gru  -- tutorial: never the vanilla RNN; which gate?
    #   direction uni  vs bi   -- tutorial's default is uni; bi is the extension
    #   readout   last vs max  -- tutorial returns the final state; max = InferSent
    #   hidden    128  vs 256  -- capacity vs overfitting at 40k (two sizes)
    #   embtrain  frozen/fine-tuned/random -- required in BOTH families (item 4)
    #   reg       word dropout -- tutorial's RNN-correct regulariser (not state dropout)
    #   pair      nli/concat/joined -- separate-encode+combine vs read the pair as one seq
    #   lr        3e-4/1e-3/3e-3 -- re-screened; RNN optimisation differs from pooling
    R = RNN_ANCHOR
    if name == "rnn_cell":
        return OrderedDict([
            ("lstm [anchor]", _spec(R)),
            ("gru",           _spec(R, **{"enc_cfg.cell": "gru"})),
        ])
    if name == "rnn_direction":
        return OrderedDict([
            ("uni [anchor]", _spec(R)),
            ("bi",           _spec(R, **{"enc_cfg.bi": True})),
        ])
    if name == "rnn_readout":
        return OrderedDict([
            ("last [anchor]", _spec(R)),
            ("maxpool",       _spec(R, **{"enc_cfg.output": "maxpool"})),
        ])
    if name == "rnn_hidden":
        return OrderedDict([
            ("128",          _spec(R, **{"enc_cfg.hidden": 128})),
            ("256 [anchor]", _spec(R)),
        ])
    if name == "rnn_embtrain":
        return OrderedDict([
            ("frozen [anchor]", _spec(R)),
            ("fine-tuned",      _spec(R, **{"enc_cfg.freeze": False, "opt_cfg.emb_lr": 1e-4})),
            ("random init",     _spec(R, **{"enc_cfg.random_init": True})),
        ])
    if name == "rnn_reg":
        return OrderedDict([
            ("wdrop 0.0 [anchor]", _spec(R)),
            ("wdrop 0.1",          _spec(R, **{"enc_cfg.word_dropout": 0.1})),
            ("wdrop 0.2",          _spec(R, **{"enc_cfg.word_dropout": 0.2})),
        ])
    if name == "rnn_pair":
        return OrderedDict([
            ("nli [anchor]", _spec(R)),
            ("concat",       _spec(R, combo="concat")),
            ("joined",       _spec(R, combo="joined")),
        ])
    if name == "rnn_lr":
        return OrderedDict([
            ("3e-4",          _spec(R, **{"opt_cfg.learning_rate": 3e-4})),
            ("1e-3 [anchor]", _spec(R)),
            ("3e-3",          _spec(R, **{"opt_cfg.learning_rate": 3e-3})),
        ])
    if name == "rnn_final":
        # Step-5 confirmation on seeds 0/3/5. Labels for lstm[anchor], gru and
        # maxpool are byte-identical to their screening labels, so their seed-0
        # folds are reused from the checkpoint; only the stacked configs are new.
        # Tests: does GRU+maxpool hold and beat the baseline; does stacking the
        # two single-axis wins exceed each alone; does bidirectionality earn its
        # keep ONCE maxpool is on (the InferSent BiLSTM-maxpool question, which
        # the screen could not answer because bi was tested with last-state); does
        # the higher-lr lean (3e-3) add on top.
        return OrderedDict([
            ("lstm [anchor]",      _spec(R)),
            ("gru",                _spec(R, **{"enc_cfg.cell": "gru"})),
            ("maxpool",            _spec(R, **{"enc_cfg.output": "maxpool"})),
            ("gru+maxpool",        _spec(R, **{"enc_cfg.cell": "gru", "enc_cfg.output": "maxpool"})),
            ("gru+maxpool+bi",     _spec(R, **{"enc_cfg.cell": "gru", "enc_cfg.output": "maxpool",
                                               "enc_cfg.bi": True})),
            ("gru+maxpool lr3e-3", _spec(R, **{"enc_cfg.cell": "gru", "enc_cfg.output": "maxpool",
                                               "opt_cfg.learning_rate": 3e-3})),
        ])
    if name == "final":
        # Confirmation of the Step-4 combined winner. Labels for the three known
        # configs are kept byte-identical to their original screening labels so
        # system_key() matches and the seed-0 folds are reused from the checkpoint;
        # only COMBINED is new on seed 0. summarize() measures each lift vs the
        # frozen-mean baseline, and the two component rows let us read the
        # pool x fine-tune interaction (does the combination hold, not just add?).
        return OrderedDict([
            ("mean [anchor]", _spec(A)),                                  # frozen-mean baseline
            ("meanmax",       _spec(A, **{"enc_cfg.pool": "meanmax"})),   # pooling winner alone
            ("fine-tuned",    _spec(A, **{"enc_cfg.freeze": False,        # embedding winner alone
                                          "opt_cfg.emb_lr": 1e-4})),
            ("COMBINED",      _spec(A, **{"enc_cfg.pool": "meanmax",      # the shipping config
                                          "enc_cfg.freeze": False,
                                          "opt_cfg.emb_lr": 1e-4,
                                          "opt_cfg.learning_rate": 3e-4,
                                          "opt_cfg.batch_size": 256})),
        ])
    raise ValueError(f"unknown axis: {name}")


def system_key(label, spec):
    """Filesystem-safe unique key (label + config hash) for results/checkpoint."""
    h = hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    slug = "".join(c if c.isalnum() else "_" for c in label).strip("_")
    return f"{slug}_{h}"


def screen(specs, cv_sets, emb_models, tokenizers, checkpoint, results_folder,
           seeds, num_classes=3, device="cpu", verbose=True):
    """Run cross_validation for each spec; return {label: per-fold accuracy array}."""
    import project2_pairs as PP
    out = OrderedDict()
    for label, spec in specs.items():
        key = system_key(label, spec)
        per_seed = PP.cross_validation(
            key, cv_sets, tokenizers[spec["tok_key"]], emb_models[spec["emb_key"]],
            spec["enc_cfg"], spec["combo"], spec["head_cfg"], spec["opt_cfg"],
            num_classes, results_folder, checkpoint, seeds, device)
        accs = np.array([a for s in per_seed for a in s["accuracies"].values()])
        out[label] = accs
        if verbose:
            print(f"  {label:18s} mean={accs.mean():.4f} std={accs.std():.4f} (n={len(accs)})")
    return out


# --------------------------------------------------------------------------
# Benavoli correlated t-test (Bayesian) with a ROPE. For k-fold CV the folds
# overlap in training data, so the naive t-test is over-confident; the
# Nadeau-Bengio correction inflates the variance by rho/(1-rho) with rho = 1/k.
# Returns P(a better), P(practically equivalent), P(b better).
# --------------------------------------------------------------------------
def correlated_ttest(diffs, rope=0.005, rho=0.2):
    from scipy import stats
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean = float(diffs.mean())
    var = float(diffs.var(ddof=1)) if n > 1 else 0.0
    if var <= 0:                                   # identical every fold
        if abs(mean) <= rope:
            return {"p_left": 0.0, "p_rope": 1.0, "p_right": 0.0, "mean_diff": mean}
        return {"p_left": float(mean > rope), "p_rope": 0.0,
                "p_right": float(mean < -rope), "mean_diff": mean}
    scale = np.sqrt(var * (1.0 / n + rho / (1.0 - rho)))
    dof = n - 1
    p_left = float(stats.t.sf((rope - mean) / scale, dof))     # P(diff >  rope)
    p_right = float(stats.t.cdf((-rope - mean) / scale, dof))  # P(diff < -rope)
    p_rope = float(max(0.0, 1.0 - p_left - p_right))
    return {"p_left": p_left, "p_rope": p_rope, "p_right": p_right, "mean_diff": mean}


def summarize(results, anchor_label, rope=0.005, rho=0.2, decisive=0.95):
    """One row per config: mean, spread, difference vs anchor, the three Benavoli
    probabilities, and a verdict."""
    anchor = results[anchor_label]
    rows = []
    for label, accs in results.items():
        if label == anchor_label:
            tt, verdict = {"p_left": 0.0, "p_rope": 1.0, "p_right": 0.0}, "anchor"
        else:
            tt = correlated_ttest(accs - anchor, rope, rho)
            if tt["p_left"] > decisive:
                verdict = "BETTER"
            elif tt["p_right"] > decisive:
                verdict = "worse"
            elif tt["p_rope"] > decisive:
                verdict = "equivalent"
            else:
                verdict = "inconclusive"
        rows.append({
            "config": label,
            "mean_acc": round(float(accs.mean()), 4),
            "std": round(float(accs.std()), 4),
            "d_vs_anchor": round(float(accs.mean() - anchor.mean()), 4),
            "P(better)": round(tt["p_left"], 3),
            "P(rope)": round(tt["p_rope"], 3),
            "P(worse)": round(tt["p_right"], 3),
            "verdict": verdict,
        })
    return rows


# ==========================================================================
# Step 6 — the two confirmed winners, and the init-seed variance study.
# ==========================================================================
def best_specs():
    """The confirmed Step 4 / Step 5 winners, pulled from their confirmation
    axes so there is a single source of truth (no hand-copied config drift)."""
    return {"pooled":    axis("final")["COMBINED"],
            "recurrent": axis("rnn_final")["maxpool"]}


def build_nested_subsets(pool, sizes, seed=0, num_labels=3):
    """Stratified, NESTED training subsets for the Tier-B learning curve. Each
    class is shuffled once (seeded); a size-n subset takes a label-proportional
    prefix of each class, so every smaller subset is contained in every larger
    one — a clean 'more of the same data' progression. sizes with None or a value
    >= len(pool) mean the full pool. Returns {actual_size: subset}, small->large."""
    import random
    rng = random.Random(seed)
    by_label = {y: [] for y in range(num_labels)}
    for item in pool:
        by_label[int(item[1])].append(item)
    for y in by_label:
        rng.shuffle(by_label[y])
    N = len(pool)
    out = {}
    for s in sizes:
        n = N if (s is None or s >= N) else int(s)
        sub = []
        for y in range(num_labels):
            k = round(len(by_label[y]) * n / N)
            sub.extend(by_label[y][:k])
        rng.shuffle(sub)                       # mix labels within the subset
        out[len(sub)] = sub
    return dict(sorted(out.items()))


def init_variance(spec, fold_data, emb_models, tokenizers, init_seeds,
                  num_classes=3, device="cpu", verbose=True):
    """Fix ONE data split; retrain with several initialisation seeds; return the
    val accuracies. This isolates optimisation/initialisation variance (the data
    is held constant) so it can be compared with the across-fold (split)
    variance we already have from the confirmations. Each seed reseeds weight
    init, batch shuffling and dropout through train(seed=...)."""
    import project2_pairs as PP
    tok = tokenizers[spec["tok_key"]]
    emb = emb_models[spec["emb_key"]]
    accs = []
    for s in init_seeds:
        metrics, best = PP.train(
            fold_data["training"], fold_data["validation"], tok, emb,
            spec["enc_cfg"], spec["combo"], spec["head_cfg"], spec["opt_cfg"],
            num_classes, device=device, seed=s)
        a = metrics[best]["val_metrics"]["accuracy"]
        accs.append(a)
        if verbose:
            print(f"    init_seed={s}  val_acc={a:.4f}", flush=True)
    return np.array(accs)
