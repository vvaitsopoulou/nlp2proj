"""
Extension 2 — hypothesis-only per encoder + the hard subset (after Gururangan 2018,
Poliak 2018). Two things on the TEST set:

  (A) Per-encoder premise-blind ceiling: train a hypothesis-only version of each
      new encoder (pooled, recurrent) and compare to the full model -> how much the
      premise actually adds for each architecture.
  (B) Hard subset: the items the hypothesis-only model gets WRONG (the artifact
      cannot solve them). Score every full system on hard vs easy -> a clean
      "genuine inference" number, stripped of the shortcut.

Full embed systems are reused from Step 9 (results_p2/final_*_full.pt); sparse is
retrained. Hypothesis-only models are trained here (--tier full by default; use
--tier 40k for a fast ~2-min pass).

    python -u run_hyponly_hardsubset.py --glove_dir ./glove --results ./results_p2 \
        --bpe part1/results/BPE_snli_vs_2000_mf_2.json
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove_dir", required=True)
    ap.add_argument("--results", default="results_p2")
    ap.add_argument("--bpe", default="part1/results/BPE_snli_vs_2000_mf_2.json")
    ap.add_argument("--tier", default="full", choices=["full", "40k"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from scipy.sparse import csr_matrix
    from sklearn.model_selection import train_test_split
    from datasets import load_dataset
    from tokenizers import Tokenizer
    import part1  # noqa: F401
    from part1.smlp import (SparseMLP, SparseDataset, sparse_collate,
                            train_one_epoch, evaluate as sparse_evaluate)
    import part2
    import project2_lib as P2
    import project2_pairs as PP
    import project2_sweep as SW

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| tier:", args.tier, flush=True)

    snli = load_dataset("stanfordnlp/snli")
    train_pairs_all, _ = P2.clean_training_pool(snli["train"])
    pool = train_pairs_all if args.tier == "full" else P2.build_working_set(train_pairs_all)
    test_pairs = P2.clean_eval_split(snli["test"])
    y = np.array([int(l) for _, l in test_pairs])
    print(f"train pool: {len(pool)} | test: {len(test_pairs)}", flush=True)

    emb = part2.clslib.PretrainedEmbeddings(
        os.path.join(args.glove_dir, "glove.6B.300d.txt"), num_dims=300)
    bt = part2.clslib.BasicTokenizer(lowercase=True, remove_punct=False,
                                     remove_stopwords=False, language="english", vocab=emb.vocab)
    pad = emb.vocab.index(emb.pad_token); unk = emb.vocab.default_index
    bpe = Tokenizer.from_file(args.bpe); V = len(bpe.get_vocab())

    def split_pool(p):
        X = [ph for ph, _ in p]; yy = [lab for _, lab in p]
        Xa, Xb, ya, yb = train_test_split(X, yy, test_size=0.1, stratify=yy, random_state=args.seed)
        return list(zip(Xa, ya)), list(zip(Xb, yb))

    def embed_preds(model, combo):
        ds = PP.PairCorpus(test_pairs, bt, unk, joined=(combo == "joined"), sep_id=unk)
        dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=PP.make_collate(pad))
        out = PP.evaluate(model, dl, nn.NLLLoss(), device)
        return np.array([int(p) for b in out["predictions"] for p in b])

    def train_embed(spec, tag):
        tr, vs = split_pool(pool)
        save_to = os.path.join(args.results, f"ext2_{tag}")
        print(f"[{tag}] training on {len(tr)}...", flush=True)
        PP.train(tr, vs, bt, emb, spec["enc_cfg"], spec["combo"], spec["head_cfg"],
                 spec["opt_cfg"], 3, device=device, save_to=save_to, seed=args.seed)
        return embed_preds(PP.load(save_to, emb, device), spec["combo"])

    def load_or_train_full(name, spec):
        path = os.path.join(args.results, f"final_{name}_{args.tier}")
        if os.path.exists(path + ".pt"):
            print(f"[{name}_{args.tier}] loading {path}", flush=True)
            return embed_preds(PP.load(path, emb, device), spec["combo"])
        return train_embed(spec, f"{name}_{args.tier}")

    def diffprod_csr(pairs):
        data, idx, indptr = [], [], [0]
        for (p, h), _ in pairs:
            pi = set(bpe.encode(p).ids); hi = set(bpe.encode(h).ids)
            row = {}
            for i in pi:        row[i] = 1.0
            for i in hi:        row[V + i] = 1.0
            for i in pi ^ hi:   row[2 * V + i] = 1.0
            for i in pi & hi:   row[3 * V + i] = 1.0
            for k, v in row.items():
                idx.append(k); data.append(v)
            indptr.append(len(idx))
        return csr_matrix((np.asarray(data, np.float32), np.asarray(idx, np.int64),
                           np.asarray(indptr, np.int64)), shape=(len(pairs), 4 * V))

    def sparse_full_preds():
        tr, vs = split_pool(pool)
        print(f"[sparse_full] training on {len(tr)}...", flush=True)
        Xtr, ytr = diffprod_csr(tr), [int(l) for _, l in tr]
        Xvs, yvs = diffprod_csr(vs), [int(l) for _, l in vs]
        Xte = diffprod_csr(test_pairs)
        trl = DataLoader(SparseDataset(Xtr, ytr), batch_size=64, shuffle=True, collate_fn=sparse_collate)
        vsl = DataLoader(SparseDataset(Xvs, yvs), batch_size=64, shuffle=False, collate_fn=sparse_collate)
        tel = DataLoader(SparseDataset(Xte, list(y)), batch_size=64, shuffle=False, collate_fn=sparse_collate)
        P2.set_all_seeds(args.seed)
        m = SparseMLP(4 * V, 128, 3, dropout=0.1).to(device)
        crit = nn.NLLLoss(); opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
        bl, bs = None, None
        for _ in range(10):
            train_one_epoch(m, trl, opt, crit, device)
            va, _ = sparse_evaluate(m, vsl, crit, device)
            if bl is None or va["loss"] <= bl:
                bl = va["loss"]; bs = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        m.load_state_dict(bs)
        _, pr = sparse_evaluate(m, tel, crit, device)
        return np.array([int(p) for p in pr])

    pooled = SW.best_specs()["pooled"]; recur = SW.best_specs()["recurrent"]
    hyp_pooled = copy.deepcopy(pooled); hyp_pooled["combo"] = "hypothesis_only"
    hyp_recur = copy.deepcopy(recur);  hyp_recur["combo"] = "hypothesis_only"

    preds = {
        "pooled_full":    load_or_train_full("pooled", pooled),
        "recurrent_full": load_or_train_full("recurrent", recur),
        "sparse_full":    sparse_full_preds(),
        "pooled_hyponly": train_embed(hyp_pooled, "pooled_hyponly"),
        "recurrent_hyponly": train_embed(hyp_recur, "recurrent_hyponly"),
    }
    acc = {k: float((v == y).mean()) for k, v in preds.items()}

    # (A) per-encoder premise-blind ceiling and the premise contribution
    print("\n=== (A) hypothesis-only ceiling per encoder (TEST) ===")
    print(f"  pooled     full {acc['pooled_full']:.4f}  hyp-only {acc['pooled_hyponly']:.4f}  "
          f"premise adds {acc['pooled_full']-acc['pooled_hyponly']:+.4f}")
    print(f"  recurrent  full {acc['recurrent_full']:.4f}  hyp-only {acc['recurrent_hyponly']:.4f}  "
          f"premise adds {acc['recurrent_full']-acc['recurrent_hyponly']:+.4f}")

    # (B) hard subset = items the (pooled) hypothesis-only model gets WRONG
    hard = preds["pooled_hyponly"] != y
    print(f"\n=== (B) hard subset = hypothesis-only WRONG: {int(hard.sum())}/{len(y)} "
          f"({hard.mean():.1%}) ===")
    print(f"{'system':16s} {'overall':>8s} {'easy':>8s} {'HARD':>8s}")
    for s in ["sparse_full", "pooled_full", "recurrent_full"]:
        p = preds[s]
        print(f"{s:16s} {(p==y).mean():>8.4f} {(p[~hard]==y[~hard]).mean():>8.4f} "
              f"{(p[hard]==y[hard]).mean():>8.4f}")

    out = {"accuracy": acc, "hard_n": int(hard.sum()), "hard_frac": float(hard.mean()),
           "hard_subset_acc": {s: float((preds[s][hard] == y[hard]).mean())
                               for s in ["sparse_full", "pooled_full", "recurrent_full"]}}
    with open(os.path.join(args.results, "ext2_hyponly_hardsubset.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote ext2_hyponly_hardsubset.json", flush=True)


if __name__ == "__main__":
    main()
