"""
Step 7 sparse arm — the Project-1 `mlp1_diffprod` system on the Tier-B learning
curve (nested subsets, scored on official SNLI dev). Reproduces the sparse model
EXACTLY (tutorial SparseMLP on token-indicator + diff/product features, BPE vocab
2000, MLP_ARGS) but builds the feature matrix as a scipy CSR directly from the
sparse structure, so it runs at 549k without the dense-list OOM that Project 1's
encoder would hit above ~40k.

Run on the vast box:
    python -u run_sparse_scaling.py --results ./results_p2 \
        --bpe part1/results/BPE_snli_vs_2000_mf_2.json
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


# mlp1_diffprod's training recipe (verbatim from Project 1's MLP_ARGS).
MLP_ARGS = {"hidden_dim": 128, "dropout": 0.1, "batch_size": 64,
            "learning_rate": 1e-4, "weight_decay": 1e-4, "num_epochs": 10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_p2")
    ap.add_argument("--bpe", default="part1/results/BPE_snli_vs_2000_mf_2.json",
                    help="the saved BPE tokenizer from Project 1")
    ap.add_argument("--sizes", nargs="*", type=int, default=[10000, 40000, 160000, -1],
                    help="training subset sizes; -1 = full pool")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1001, 1002])
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    import torch.nn as nn
    import pandas as pd
    from scipy.sparse import csr_matrix
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from tokenizers import Tokenizer

    import part1  # noqa: F401  (ensures the package imports cleanly)
    from part1.smlp import SparseMLP, SparseDataset, sparse_collate, train_one_epoch, evaluate
    import project2_lib as P2
    import project2_sweep as SW

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    # ---- data: SAME deterministic cleaning as the embed pipeline, SAME dev set ----
    print("loading SNLI + rebuilding the clean pool...", flush=True)
    snli = load_dataset("stanfordnlp/snli")
    train_pairs_all, stats = P2.clean_training_pool(snli["train"])
    print("clean stats:", stats, flush=True)
    dev = P2.clean_eval_split(snli["validation"])

    tok = Tokenizer.from_file(args.bpe)
    V = len(tok.get_vocab())
    print(f"BPE vocab: {V} | dev set: {len(dev)}", flush=True)

    sizes = [None if s < 0 else s for s in args.sizes]
    subsets = SW.build_nested_subsets(train_pairs_all, sizes, seed=P2.SUBSAMPLE_SEED)
    print("Tier-B subset sizes:", list(subsets.keys()), flush=True)

    # ---- diff/product indicator features -> CSR, built sparsely (no dense list) ----
    # blocks of width V: [ind_p | ind_h | |ind_p - ind_h| (XOR) | ind_p * ind_h (AND)]
    def diffprod_csr(pairs):
        data, indices, indptr = [], [], [0]
        for (p, h), _ in pairs:
            pids = set(tok.encode(p).ids)
            hids = set(tok.encode(h).ids)
            row = {}
            for i in pids:          row[i] = 1.0
            for i in hids:          row[V + i] = 1.0
            for i in pids ^ hids:   row[2 * V + i] = 1.0     # |p - h|
            for i in pids & hids:   row[3 * V + i] = 1.0     # p * h
            for idx, val in row.items():
                indices.append(idx); data.append(val)
            indptr.append(len(indices))
        return csr_matrix((np.asarray(data, dtype=np.float32),
                           np.asarray(indices, dtype=np.int64),
                           np.asarray(indptr, dtype=np.int64)),
                          shape=(len(pairs), 4 * V))

    print("encoding dev set once...", flush=True)
    Xdv = diffprod_csr(dev)
    ydv = [int(l) for _, l in dev]

    rows = []
    for n, sub in subsets.items():
        print(f"encoding train subset size={n}...", flush=True)
        Xtr = diffprod_csr(sub)                              # once per size, reused across seeds
        ytr = [int(l) for _, l in sub]
        for seed in args.seeds:
            P2.set_all_seeds(seed)                           # weight init + shuffling
            trl = DataLoader(SparseDataset(Xtr, ytr), batch_size=MLP_ARGS["batch_size"],
                             shuffle=True, collate_fn=sparse_collate)
            dvl = DataLoader(SparseDataset(Xdv, ydv), batch_size=MLP_ARGS["batch_size"],
                             shuffle=False, collate_fn=sparse_collate)
            model = SparseMLP(4 * V, MLP_ARGS["hidden_dim"], 3, dropout=MLP_ARGS["dropout"]).to(device)
            crit = nn.NLLLoss()
            opt = torch.optim.AdamW(model.parameters(), lr=MLP_ARGS["learning_rate"],
                                    weight_decay=MLP_ARGS["weight_decay"])
            best_loss, best_acc = None, 0.0
            t = time.time()
            for _ in range(MLP_ARGS["num_epochs"]):
                train_one_epoch(model, trl, opt, crit, device)
                va, _ = evaluate(model, dvl, crit, device)
                if best_loss is None or va["loss"] <= best_loss:   # best-by-dev-loss, matching Tier B
                    best_loss, best_acc = va["loss"], va["accuracy"]
            dt = time.time() - t
            rows.append({"system": "sparse_mlp1_diffprod", "size": n, "seed": seed,
                         "dev_acc": float(best_acc), "sec": round(dt, 1)})
            print(f"  sparse size={n:6d} seed={seed} dev_acc={best_acc:.4f} ({dt:.0f}s)", flush=True)

    Path(args.results).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.results, "sparse_scaling.csv"), index=False)
    print("\nsparse_scaling.csv written to", args.results, flush=True)


if __name__ == "__main__":
    main()
