"""
Breaking NLI (Glockner et al., ACL 2018) loader + scoring.

Dataset: 8193 premise/hypothesis pairs that differ by ONE word/phrase.
Label skew is extreme (contradiction 7164 / entailment 982 / neutral 47),
so overall accuracy is dominated by contradiction. The diagnostic lives in
the PER-CATEGORY and PREDICTED-LABEL-DISTRIBUTION views:

  - synonyms (894)            -> gold is always ENTAILMENT  (did meaning survive the swap?)
  - antonyms / antonyms_wordnet -> gold is ~always CONTRADICTION (distributional vectors place opposites close)
  - countries/nationalities/colors/materials/planets/... -> CONTRADICTION via co-hyponym / mutual exclusivity
  - cardinals/ordinals        -> CONTRADICTION via number swap

A model that just rides lexical overlap (premise ~ hypothesis) will predict
ENTAILMENT everywhere -> great on synonyms, near-zero on everything else.
A model that rides the SNLI "words differ -> contradiction" prior will do the
opposite. Neither is lexical understanding; the per-category split exposes which.

This module is label-mapping agnostic: it works in label STRINGS throughout,
so the caller maps model class-ids to strings using whatever mapping the models
were TRAINED with (must match the training pipeline, not assumed here).
"""
import json
import ast
from collections import Counter, OrderedDict

LABELS = ["entailment", "neutral", "contradiction"]

# Category groupings for the scientific read-out (every category is also reported individually).
CAT_GROUPS = OrderedDict([
    ("synonyms (->entail)",      ["synonyms"]),
    ("antonyms (->contra)",      ["antonyms", "antonyms_wordnet"]),
    ("co-hyponym (->contra)",    ["countries", "nationalities", "colors", "materials",
                                  "planets", "drinks", "vegetables", "instruments", "rooms"]),
    ("numeric (->contra)",       ["cardinals", "ordinals"]),
])


def load_breaking_nli(path):
    """Return list of dicts: {premise, hypothesis, gold_label(str), category}.
    Lines are valid JSON in the released file; ast fallback kept for safety."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                d = ast.literal_eval(line)
            p = d.get("sentence1")
            h = d.get("sentence2")
            g = d.get("gold_label")
            c = d.get("category")
            if not p or not h or g not in LABELS:
                continue  # Breaking NLI has no '-' labels, but guard anyway
            rows.append({"premise": p, "hypothesis": h, "gold_label": g, "category": c})
    return rows


def _acc(gold, pred):
    if not gold:
        return float("nan")
    return sum(int(a == b) for a, b in zip(gold, pred)) / len(gold)


def report(system_name, gold, pred, categories, out=None):
    """Compute + pretty-print the Breaking NLI breakdown for one system.
    gold/pred: lists of label STRINGS. categories: list of category strings.
    Returns a JSON-able dict."""
    assert len(gold) == len(pred) == len(categories)
    n = len(gold)

    overall = _acc(gold, pred)
    # majority-class (always contradiction) baseline on this exact set
    base = gold.count("contradiction") / n

    # per gold-label recall
    per_label = {}
    for lab in LABELS:
        idx = [i for i in range(n) if gold[i] == lab]
        per_label[lab] = {
            "n": len(idx),
            "acc": _acc([gold[i] for i in idx], [pred[i] for i in idx]) if idx else None,
        }

    # predicted-label distribution (diagnoses class-defaulting)
    pred_dist = {lab: pred.count(lab) / n for lab in LABELS}

    # per-category accuracy
    cats = sorted(set(categories))
    per_cat = {}
    for c in cats:
        idx = [i for i in range(n) if categories[i] == c]
        per_cat[c] = {"n": len(idx),
                      "acc": _acc([gold[i] for i in idx], [pred[i] for i in idx])}

    # grouped accuracy
    per_group = {}
    for gname, members in CAT_GROUPS.items():
        idx = [i for i in range(n) if categories[i] in members]
        per_group[gname] = {"n": len(idx),
                            "acc": _acc([gold[i] for i in idx], [pred[i] for i in idx]) if idx else None}

    res = {"system": system_name, "n": n, "overall_acc": overall,
           "always_contradiction_baseline": base,
           "per_label": per_label, "pred_distribution": pred_dist,
           "per_group": per_group, "per_category": per_cat}

    lines = []
    lines.append(f"\n==================== {system_name} ====================")
    lines.append(f"  overall acc            : {overall:.4f}   (always-contradiction = {base:.4f})")
    lines.append(f"  predicted-label dist   : " +
                 "  ".join(f"{lab[:3]}={pred_dist[lab]:.3f}" for lab in LABELS))
    lines.append(f"  per gold-label acc     : " +
                 "  ".join(f"{lab[:3]}({per_label[lab]['n']})="
                          f"{(per_label[lab]['acc'] if per_label[lab]['acc'] is not None else float('nan')):.3f}"
                          for lab in LABELS))
    lines.append("  grouped:")
    for gname, d in per_group.items():
        a = d["acc"]
        lines.append(f"     {gname:24s} n={d['n']:5d}  acc={a:.3f}" if a is not None
                     else f"     {gname:24s} n={d['n']:5d}  acc=  n/a")
    lines.append("  per-category:")
    for c in sorted(per_cat, key=lambda k: -per_cat[k]["n"]):
        lines.append(f"     {c:18s} n={per_cat[c]['n']:5d}  acc={per_cat[c]['acc']:.3f}")
    text = "\n".join(lines)
    print(text)
    if out is not None:
        out.write(text + "\n")
    return res


if __name__ == "__main__":
    # Self-test: validate loader + scoring with two reference predictors.
    import sys
    rows = load_breaking_nli(sys.argv[1])
    gold = [r["gold_label"] for r in rows]
    cats = [r["category"] for r in rows]
    print(f"loaded {len(rows)} rows")

    # Predictor A: always 'contradiction' (the SNLI words-differ prior taken to the limit)
    report("ALWAYS-CONTRADICTION", gold, ["contradiction"] * len(gold), cats)

    # Predictor B: always 'entailment' (pure lexical-overlap shortcut; every pair overlaps maximally)
    report("ALWAYS-ENTAILMENT", gold, ["entailment"] * len(gold), cats)
