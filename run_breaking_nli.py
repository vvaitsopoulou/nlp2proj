"""
Extension 3 — Breaking NLI (Glockner, Shwartz, Goldberg, ACL 2018).

An 8193-pair diagnostic test set where premise and hypothesis differ by ONE
word/phrase, so lexical overlap is ~maximal for every pair. Labels track the
swap category: synonyms -> entailment; antonyms / co-hyponyms / number swaps ->
contradiction. 87.4% of golds are contradiction, so overall accuracy is a trap
(always-contradiction = 0.874). The signal is per-category accuracy (esp.
synonyms vs antonyms) and each system's predicted-label distribution.

This closes the Step-2 -> Step-8 thread at the system level: do our systems,
which lean on lexical overlap, collapse exactly where distributional vectors
cannot separate opposites / co-hyponyms?

Reuses the Step-9 full models (results_p2/final_*_{tier}.pt) and retrains sparse,
with the SAME load/predict/featurize code as run_hyponly_hardsubset.py so numbers
are directly comparable. Scoring lives in bnli_data.py (unit-tested separately).

    # fetch the data first (on the box, open internet):
    #   cd ~/snli && mkdir -p data_breaking && cd data_breaking
    #   curl -fsSL -o bn.zip https://raw.githubusercontent.com/BIU-NLP/Breaking_NLI/master/breaking_nli_dataset.zip
    #   unzip -o bn.zip   ->  data/dataset.jsonl
    python -u run_breaking_nli.py --glove_dir ./glove --results ./results_p2 \
        --bpe part1/results/BPE_snli_vs_2000_mf_2.json \
        --bnli data_breaking/data/dataset.jsonl --tier full
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

S2I = {"entailment": 0, "neutral": 1, "contradiction": 2}   # stanfordnlp/snli coding
I2S = {v: k for k, v in S2I.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove_dir", required=True)
    ap.add_argument("--results", default="results_p2")
    ap.add_argument("--bpe", default="part1/results/BPE_snli_vs_2000_mf_2.json")
    ap.add_argument("--bnli", required=True, help="path to Breaking NLI dataset.jsonl")
    ap.add_argument("--tier", default="full", choices=["full", "40k"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_selfcheck", action="store_true",
                    help="skip the SNLI-test integrity gate")
    args = ap.parse_args()
    here = str(Path(__file__).resolve().parent)
    sys.path.insert(0, here)

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
    import bnli_data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| tier:", args.tier, flush=True)

    # --- SNLI (train pool for sparse retrain; test set for the integrity gate) ---
    snli = load_dataset("stanfordnlp/snli")
    train_pairs_all, _ = P2.clean_training_pool(snli["train"])
    pool = train_pairs_all if args.tier == "full" else P2.build_working_set(train_pairs_all)
    test_pairs = P2.clean_eval_split(snli["test"])
    y = np.array([int(l) for _, l in test_pairs])
    print(f"train pool: {len(pool)} | snli test: {len(test_pairs)}", flush=True)

    # --- embeddings / tokenizers (identical setup to run_hyponly_hardsubset.py) ---
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

    # --- embed predict over an arbitrary pairs list -> int class ids ---
    def embed_preds(model, combo, pairs):
        ds = PP.PairCorpus(pairs, bt, unk, joined=(combo == "joined"), sep_id=unk)
        dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=PP.make_collate(pad))
        out = PP.evaluate(model, dl, nn.NLLLoss(), device)
        return np.array([int(p) for b in out["predictions"] for p in b])

    def load_or_train_full(name, spec):
        path = os.path.join(args.results, f"final_{name}_{args.tier}")
        if os.path.exists(path + ".pt"):
            print(f"[{name}_{args.tier}] loading {path}", flush=True)
            return PP.load(path, emb, device), spec["combo"]
        # fallback: train on pool (mirrors Ext-2) if the Step-9 model is missing
        tr, vs = split_pool(pool)
        save_to = os.path.join(args.results, f"final_{name}_{args.tier}")
        print(f"[{name}_{args.tier}] MISSING -> training on {len(tr)}...", flush=True)
        PP.train(tr, vs, bt, emb, spec["enc_cfg"], spec["combo"], spec["head_cfg"],
                 spec["opt_cfg"], 3, device=device, save_to=save_to, seed=args.seed)
        return PP.load(save_to, emb, device), spec["combo"]

    # --- sparse (Project-1 mlp1_diffprod): CSR diff/prod features ---
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

    def train_sparse():
        tr, vs = split_pool(pool)
        print(f"[sparse] training on {len(tr)}...", flush=True)
        Xtr, ytr = diffprod_csr(tr), [int(l) for _, l in tr]
        Xvs, yvs = diffprod_csr(vs), [int(l) for _, l in vs]
        trl = DataLoader(SparseDataset(Xtr, ytr), batch_size=64, shuffle=True, collate_fn=sparse_collate)
        vsl = DataLoader(SparseDataset(Xvs, yvs), batch_size=64, shuffle=False, collate_fn=sparse_collate)
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
        return m, crit

    def sparse_preds(m, crit, pairs):
        Xte = diffprod_csr(pairs)
        tel = DataLoader(SparseDataset(Xte, [0] * len(pairs)), batch_size=64,
                         shuffle=False, collate_fn=sparse_collate)
        _, pr = sparse_evaluate(m, tel, crit, device)
        return np.array([int(p) for p in pr])

    # --- build the three systems ---
    pooled_spec = SW.best_specs()["pooled"]; recur_spec = SW.best_specs()["recurrent"]
    pooled_model, pooled_combo = load_or_train_full("pooled", pooled_spec)
    recur_model, recur_combo = load_or_train_full("recurrent", recur_spec)
    sparse_m, sparse_crit = train_sparse()

    # --- integrity gate: reproduce the SNLI-test numbers from Step 9 / Ext 2 ---
    if not args.no_selfcheck:
        scp = float((embed_preds(pooled_model, pooled_combo, test_pairs) == y).mean())
        scr = float((embed_preds(recur_model, recur_combo, test_pairs) == y).mean())
        scs = float((sparse_preds(sparse_m, sparse_crit, test_pairs) == y).mean())
        print(f"\n[selfcheck] SNLI test acc  pooled={scp:.4f}  recurrent={scr:.4f}  sparse={scs:.4f}")
        print("  (tier=full should reproduce Step-9/Ext-2: pooled~0.7912 recurrent~0.8291 sparse~0.7541)")
        if args.tier == "full" and not (abs(scp - 0.7912) < 0.02 and abs(scr - 0.8291) < 0.02):
            print("  !! WARNING: full-tier self-check drifted >2pt from the known finals — "
                  "check label mapping / model load BEFORE trusting Breaking NLI numbers.", flush=True)

    # --- Breaking NLI ---
    rows = bnli_data.load_breaking_nli(args.bnli)
    bnli_pairs = [((r["premise"], r["hypothesis"]), S2I[r["gold_label"]]) for r in rows]
    gold_str = [r["gold_label"] for r in rows]
    cats = [r["category"] for r in rows]
    print(f"\nBreaking NLI: {len(rows)} pairs loaded", flush=True)

    pred_ids = {
        "sparse_full":    sparse_preds(sparse_m, sparse_crit, bnli_pairs),
        "pooled_full":    embed_preds(pooled_model, pooled_combo, bnli_pairs),
        "recurrent_full": embed_preds(recur_model, recur_combo, bnli_pairs),
    }

    results = {}
    for name in ["sparse_full", "pooled_full", "recurrent_full"]:
        pred_str = [I2S[int(i)] for i in pred_ids[name]]
        results[name] = bnli_data.report(name, gold_str, pred_str, cats)

    # --- save summary JSON + per-item CSV (for the grouped-bar figure / qualitative read) ---
    os.makedirs(args.results, exist_ok=True)
    jpath = os.path.join(args.results, "breaking_nli_results.json")
    with open(jpath, "w") as f:
        json.dump(results, f, indent=2)

    cpath = os.path.join(args.results, "breaking_nli_predictions.csv")
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "category", "gold", "premise", "hypothesis",
                    "pred_sparse", "pred_pooled", "pred_recurrent"])
        for i, r in enumerate(rows):
            w.writerow([i, r["category"], r["gold_label"], r["premise"], r["hypothesis"],
                        I2S[int(pred_ids['sparse_full'][i])],
                        I2S[int(pred_ids['pooled_full'][i])],
                        I2S[int(pred_ids['recurrent_full'][i])]])

    print(f"\nwrote {jpath}\nwrote {cpath}", flush=True)


if __name__ == "__main__":
    main()
