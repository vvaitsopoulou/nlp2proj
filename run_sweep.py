"""
NLP1 Project 2 — Step 4 sweep, as a batch-job script (no notebook state).

Rebuilds the Step 0 data and the Step 1 embeddings/tokenizers, then runs the
requested sweep axes and writes one CSV per axis plus the resumable checkpoint.
Everything is deterministic and checkpointed, so resubmitting continues.

Example:
    python -u run_sweep.py --glove_dir /scratch-shared/$USER/glove \
        --results $HOME/nlp2/results_p2 --axes composition pooling --seeds 0
"""
import argparse
import os
import sys
from pathlib import Path


def get_tokenizers(part2, P2, emb_model, sents):
    def base(remove_stopwords):
        return part2.clslib.BasicTokenizer(
            lowercase=True, remove_punct=False, remove_stopwords=remove_stopwords,
            language="english", vocab=emb_model.vocab)

    class HyphenTok:
        def __init__(self, b):
            self.b = b
            self.vocab = b.vocab
        def tokens(self, text):
            out = []
            for t in self.b.tokens(text):
                out.extend(t.split("-") if "-" in t else [t])
            return out
        def ids(self, text):
            return [self.vocab.index(t) for t in self.tokens(text)]

    specs = {"keep": base(False), "nostop": base(True), "hyphen": HyphenTok(base(False))}
    return {k: P2.CachedTokenizer(P2.build_id_cache(sents, bt), bt) for k, bt in specs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove_dir", required=True, help="dir holding glove.6B.{50,100,300}d.txt")
    ap.add_argument("--results", default="results_p2")
    ap.add_argument("--axes", nargs="*", default=[
        "composition", "pooling", "embedding_training", "embedding_size", "stopwords",
        "tokenizer_hyphen", "head_width", "head_dropout", "learning_rate", "batch_size"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--code_root", default=None, help="extra dir to add to sys.path for part1/part2")
    # Step 6 init-seed variance study (fix one fold, vary only the init seed):
    ap.add_argument("--variance", action="store_true",
                    help="run the init-seed variance study on the two best models and exit")
    ap.add_argument("--init_seeds", nargs="*", type=int, default=[1001, 1002, 1003, 1004, 1005])
    ap.add_argument("--var_cvseed", type=int, default=0)
    ap.add_argument("--var_fold", type=int, default=0)
    # Step 7 Tier-B learning curve (nested subsets, scored on official dev):
    ap.add_argument("--scaling", action="store_true", help="run the Tier-B learning curve and exit")
    ap.add_argument("--sizes", nargs="*", type=int, default=[10000, 40000, 160000, -1],
                    help="training subset sizes; -1 means the full pool")
    ap.add_argument("--rnn512", action="store_true",
                    help="also run the exploratory hidden-512 recurrent at the largest size")
    args = ap.parse_args()

    if args.code_root:
        sys.path.insert(0, args.code_root)
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    import pandas as pd
    import nltk
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception as exc:                       # pre-staged NLTK_DATA covers this
            print(f"nltk.download({pkg}) skipped: {exc}", flush=True)

    from datasets import load_dataset
    import part1
    from part1 import experiment as p1_experiment
    import part2
    import project2_lib as P2
    import project2_pairs as PP           # noqa: F401 (imported for side effects / parity)
    import project2_sweep as SW

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    if device == "cpu":
        print("WARNING: no GPU visible — fine-tune/random axes will be very slow.", flush=True)

    Path(args.results).mkdir(parents=True, exist_ok=True)

    # ---- Step 0: data ----
    print("loading SNLI + rebuilding working set...", flush=True)
    snli = load_dataset("stanfordnlp/snli")
    train_pairs_all, stats = P2.clean_training_pool(snli["train"])
    print("clean stats:", stats, flush=True)
    working_set = P2.build_working_set(train_pairs_all)
    cv_sets = {s: p1_experiment.cross_validation_split(working_set, k=P2.NUM_FOLDS, seed=s)
               for s in P2.CV_SEEDS}
    val_pairs = P2.clean_eval_split(snli["validation"])
    test_pairs = P2.clean_eval_split(snli["test"])

    # ---- Step 1: embeddings + tokenizers ----
    print("loading GloVe 6B at 50/100/300d...", flush=True)
    emb_models = {dim: part2.clslib.PretrainedEmbeddings(
                      os.path.join(args.glove_dir, f"glove.6B.{dim}d.txt"), num_dims=dim)
                  for dim in (50, 100, 300)}
    emb_model = emb_models[300]
    sents = P2.sentences_in(working_set, val_pairs, test_pairs)
    tok_variants = get_tokenizers(part2, P2, emb_model, sents)

    # ---- Step 6: init-seed variance study (fix one fold, vary init only) ----
    if args.variance:
        fold_data = cv_sets[args.var_cvseed][args.var_fold]
        print(f"\ninit-variance on cv_seed {args.var_cvseed} fold {args.var_fold}, "
              f"init seeds {args.init_seeds}", flush=True)
        import json as _json
        rows = []
        for name, spec in SW.best_specs().items():
            print(f"\n=== {name} ===", flush=True)
            accs = SW.init_variance(spec, fold_data, emb_models, tok_variants,
                                    args.init_seeds, device=device)
            rows.append({"model": name, "mean_acc": float(accs.mean()),
                         "init_std": float(accs.std()), "accs": list(map(float, accs))})
            print(f"  {name}: mean={accs.mean():.4f} init_std={accs.std():.4f}", flush=True)
        with open(os.path.join(args.results, "variance.json"), "w") as fh:
            _json.dump(rows, fh, indent=2)
        pd.DataFrame([{k: r[k] for k in ("model", "mean_acc", "init_std")} for r in rows]).to_csv(
            os.path.join(args.results, "variance.csv"), index=False)
        print("\nvariance.csv + variance.json written to", args.results, flush=True)
        return

    # ---- Step 7: Tier-B learning curve (nested subsets, scored on official dev) ----
    if args.scaling:
        import json as _json
        sizes = [None if s < 0 else s for s in args.sizes]
        subsets = SW.build_nested_subsets(train_pairs_all, sizes, seed=P2.SUBSAMPLE_SEED)
        print("Tier-B subset sizes:", list(subsets.keys()), "| dev set:", len(val_pairs), flush=True)
        # the tokenizer cache must cover the FULL pool, not just the 40k working set:
        print("building the keep-tokenizer cache over the full pool (one-time)...", flush=True)
        bt = part2.clslib.BasicTokenizer(lowercase=True, remove_punct=False,
                                         remove_stopwords=False, language="english", vocab=emb_model.vocab)
        sents_full = P2.sentences_in(train_pairs_all, val_pairs, test_pairs)
        tok_keep = P2.CachedTokenizer(P2.build_id_cache(sents_full, bt), bt)

        systems = dict(SW.best_specs())                      # pooled + recurrent, Tier-A configs
        if args.rnn512:
            systems["recurrent_h512"] = SW._spec(SW.RNN_ANCHOR,
                                                 **{"enc_cfg.output": "maxpool", "enc_cfg.hidden": 512})
        seeds2 = args.init_seeds[:2]
        biggest = max(subsets)
        rows = []
        for name, spec in systems.items():
            emb = emb_models[spec["emb_key"]]
            for n, sub in subsets.items():
                if name == "recurrent_h512" and n != biggest:   # exploratory: full size only
                    continue
                for seed in seeds2:
                    with P2.Timer() as t:
                        metrics, best = PP.train(sub, val_pairs, tok_keep, emb, spec["enc_cfg"],
                                                 spec["combo"], spec["head_cfg"], spec["opt_cfg"],
                                                 3, device=device, seed=seed)
                    acc = metrics[best]["val_metrics"]["accuracy"]
                    rows.append({"system": name, "size": n, "seed": seed,
                                 "dev_acc": float(acc), "sec": round(t.sec, 1)})
                    print(f"  {name:14s} size={n:6d} seed={seed} dev_acc={acc:.4f} ({t.sec:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(args.results, "scaling.csv"), index=False)
        with open(os.path.join(args.results, "scaling.json"), "w") as fh:
            _json.dump(rows, fh, indent=2)
        print("\nscaling.csv + scaling.json written to", args.results, flush=True)
        return

    # ---- Step 4: sweep ----
    ck = P2.Checkpoint(os.path.join(args.results, "sweep_ckpt.pkl"))
    for name in args.axes:
        specs = SW.axis(name)
        print(f"\n=== axis: {name} (seeds {args.seeds}) ===", flush=True)
        res = SW.screen(specs, cv_sets, emb_models, tok_variants, ck, args.results,
                        args.seeds, device=device)
        anchor = next(l for l in specs if "anchor" in l)
        df = pd.DataFrame(SW.summarize(res, anchor))
        df.to_csv(os.path.join(args.results, f"axis_{name}.csv"), index=False)
        # raw per-fold accuracies, for paired (interaction) tests after the fact
        import json as _json
        with open(os.path.join(args.results, f"axis_{name}_raw.json"), "w") as fh:
            _json.dump({lbl: list(map(float, accs)) for lbl, accs in res.items()}, fh)
        print(df.to_string(index=False), flush=True)

    print("\nDONE. CSVs + checkpoint in", args.results, flush=True)


if __name__ == "__main__":
    main()
