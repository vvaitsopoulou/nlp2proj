"""
Step 8 — error analysis data builder. Trains the four systems on the 40k working
set (early-stopped on a held-out slice, TEST untouched), predicts each on a fixed
stratified ~1,500-item DEV subsample, computes per-item NLI features, and writes
one wide CSV (error_analysis.csv) for the Bayesian correctness model + qualitative
read done in the notebook.

Systems: sparse mlp1_diffprod | pooled (meanmax fine-tuned) | recurrent (LSTM-maxpool)
         | hypothesis-only pooled (same encoder, premise discarded) = the shortcut baseline.

Run on the vast box:
    python -u run_error_analysis.py --glove_dir ./glove --results ./results_p2 \
        --bpe part1/results/BPE_snli_vs_2000_mf_2.json
"""
import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np

# Negation cues (Gururangan 2018): their presence in the hypothesis correlates with contradiction.
NEG_WORDS = {"not", "no", "never", "n't", "nothing", "nobody", "none", "nowhere",
             "neither", "nor", "without", "cannot", "can't", "won't", "don't",
             "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "ain't"}


def nli_features(premise, hypothesis, gold, hyp_only_solvable):
    pw = premise.lower().split()
    hw = hypothesis.lower().split()
    ps, hs = set(pw), set(hw)
    overlap = len(hs & ps) / max(1, len(hs))
    neg = int(any(w.strip(".,!?;:") in NEG_WORDS for w in hw))
    return {
        "premise_len": len(pw),
        "hyp_len": len(hw),
        "overlap": round(overlap, 4),
        "negation": neg,
        "is_neutral": int(gold == 1),
        "is_contradiction": int(gold == 2),
        "hyp_only_solvable": int(hyp_only_solvable),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove_dir", required=True)
    ap.add_argument("--results", default="results_p2")
    ap.add_argument("--bpe", default="part1/results/BPE_snli_vs_2000_mf_2.json")
    ap.add_argument("--n_dev", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    import torch.nn as nn
    import pandas as pd
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
    from run_sweep import get_tokenizers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    # ---- data ----
    snli = load_dataset("stanfordnlp/snli")
    train_pairs_all, stats = P2.clean_training_pool(snli["train"])
    print("clean stats:", stats, flush=True)
    working_set = P2.build_working_set(train_pairs_all)
    dev_all = P2.clean_eval_split(snli["validation"])

    # fixed stratified dev subsample (~n_dev), TEST never touched
    Xd = [ph for ph, _ in dev_all]
    yd = [lab for _, lab in dev_all]
    Xsub, _, ysub, _ = train_test_split(Xd, yd, train_size=args.n_dev,
                                        stratify=yd, random_state=args.seed)
    dev_sub = list(zip(Xsub, ysub))
    gold = [int(l) for _, l in dev_sub]
    print(f"dev subsample: {len(dev_sub)} (stratified)", flush=True)

    # early-stopping split of the 40k working set (train on 36k, select on 4k)
    Xw = [ph for ph, _ in working_set]
    yw = [lab for _, lab in working_set]
    Xa, Xb, ya, yb = train_test_split(Xw, yw, test_size=0.1, stratify=yw, random_state=args.seed)
    train_set = list(zip(Xa, ya))
    valsel = list(zip(Xb, yb))
    print(f"train/valsel: {len(train_set)}/{len(valsel)}", flush=True)

    # ---- embeddings + keep-tokenizer for the embed systems ----
    emb = part2.clslib.PretrainedEmbeddings(
        os.path.join(args.glove_dir, "glove.6B.300d.txt"), num_dims=300)
    emb_models = {300: emb}
    sents = P2.sentences_in(working_set, dev_sub)
    tok = get_tokenizers(part2, P2, emb, sents)["keep"]

    # ---- the four systems ----
    pooled_spec = SW.best_specs()["pooled"]
    recur_spec = SW.best_specs()["recurrent"]
    hyp_spec = copy.deepcopy(pooled_spec); hyp_spec["combo"] = "hypothesis_only"
    embed_systems = {"pooled": pooled_spec, "recurrent": recur_spec, "hyponly": hyp_spec}

    preds = {}

    def predict_embed(name, spec):
        save_to = os.path.join(args.results, f"erra_{name}")
        print(f"[{name}] training on {len(train_set)}...", flush=True)
        PP.train(train_set, valsel, tok, emb, spec["enc_cfg"], spec["combo"],
                 spec["head_cfg"], spec["opt_cfg"], 3, device=device, save_to=save_to, seed=args.seed)
        model = PP.load(save_to, emb, device)
        pad = emb.vocab.index(emb.pad_token)
        unk = emb.vocab.default_index
        ds = PP.PairCorpus(dev_sub, tok, unk, joined=(spec["combo"] == "joined"), sep_id=unk)
        dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=PP.make_collate(pad))
        out = PP.evaluate(model, dl, nn.NLLLoss(), device)
        flat = [int(p) for batch in out["predictions"] for p in batch]
        print(f"[{name}] dev-subsample acc = {np.mean([p == g for p, g in zip(flat, gold)]):.4f}", flush=True)
        return flat

    for name, spec in embed_systems.items():
        preds[name] = predict_embed(name, spec)

    # ---- sparse mlp1_diffprod (BPE indicator + diff/product), same recipe as Project 1 ----
    bpe = Tokenizer.from_file(args.bpe)
    V = len(bpe.get_vocab())

    def diffprod_csr(pairs):
        data, indices, indptr = [], [], [0]
        for (p, h), _ in pairs:
            pids = set(bpe.encode(p).ids)
            hids = set(bpe.encode(h).ids)
            row = {}
            for i in pids:        row[i] = 1.0
            for i in hids:        row[V + i] = 1.0
            for i in pids ^ hids: row[2 * V + i] = 1.0
            for i in pids & hids: row[3 * V + i] = 1.0
            for idx, val in row.items():
                indices.append(idx); data.append(val)
            indptr.append(len(indices))
        return csr_matrix((np.asarray(data, dtype=np.float32),
                           np.asarray(indices, dtype=np.int64),
                           np.asarray(indptr, dtype=np.int64)), shape=(len(pairs), 4 * V))

    print("[sparse] encoding + training...", flush=True)
    Xtr, ytr = diffprod_csr(train_set), [int(l) for _, l in train_set]
    Xvs, yvs = diffprod_csr(valsel),   [int(l) for _, l in valsel]
    Xdv, ydv = diffprod_csr(dev_sub),  gold
    trl = DataLoader(SparseDataset(Xtr, ytr), batch_size=64, shuffle=True, collate_fn=sparse_collate)
    vsl = DataLoader(SparseDataset(Xvs, yvs), batch_size=64, shuffle=False, collate_fn=sparse_collate)
    dvl = DataLoader(SparseDataset(Xdv, ydv), batch_size=64, shuffle=False, collate_fn=sparse_collate)
    P2.set_all_seeds(args.seed)
    model = SparseMLP(4 * V, 128, 3, dropout=0.1).to(device)
    crit = nn.NLLLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_loss, best_state = None, None
    for _ in range(10):
        train_one_epoch(model, trl, opt, crit, device)
        va, _ = sparse_evaluate(model, vsl, crit, device)
        if best_loss is None or va["loss"] <= best_loss:
            best_loss = va["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    _, sparse_preds = sparse_evaluate(model, dvl, crit, device)
    preds["sparse"] = [int(p) for p in sparse_preds]
    print(f"[sparse] dev-subsample acc = {np.mean([p == g for p, g in zip(preds['sparse'], gold)]):.4f}", flush=True)

    # ---- assemble the wide CSV ----
    rows = []
    for i, ((p, h), g) in enumerate(dev_sub):
        feats = nli_features(p, h, g, hyp_only_solvable=(preds["hyponly"][i] == g))
        row = {"item_id": i, "premise": p, "hypothesis": h, "gold": g, **feats}
        for s in ["sparse", "pooled", "recurrent", "hyponly"]:
            row[f"{s}_pred"] = preds[s][i]
            row[f"{s}_correct"] = int(preds[s][i] == g)
        rows.append(row)
    out_path = os.path.join(args.results, "error_analysis.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("\nwrote", out_path, flush=True)
    import pandas as _pd
    df = _pd.DataFrame(rows)
    print("per-system dev-subsample accuracy:\n",
          {s: round(df[f"{s}_correct"].mean(), 4) for s in ["sparse", "pooled", "recurrent", "hyponly"]}, flush=True)


if __name__ == "__main__":
    main()
