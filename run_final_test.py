"""
Step 9 — FINAL TEST EVALUATION. The SNLI test split is used ONCE, here, in a
single pass. No tuning, no iteration against it. Mirrors Project 1's final-test
cell (train on all of a training pool, early-stop on a held-out slice of THAT
pool, evaluate once on the cleaned test split) and reports, per system:
accuracy, macro-F1, per-class precision/recall/F1, and the confusion matrix.

Two numbers / two questions (the plan):
  - Tier A: every system trained on the 40k working set  -> does the ladder hold on test?
  - Tier B: systems trained on the full ~549k pool       -> full-data numbers vs published.

Run on the vast box (ONE time):
    python -u run_final_test.py --glove_dir ./glove --results ./results_p2 \
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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from scipy.sparse import csr_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
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
    LABELS = ["entailment", "neutral", "contradiction"]

    # ---- data: SAME cleaning as training; TEST cleaned the same way, used ONCE ----
    snli = load_dataset("stanfordnlp/snli")
    train_pairs_all, stats = P2.clean_training_pool(snli["train"])
    working_set = P2.build_working_set(train_pairs_all)
    test_pairs = P2.clean_eval_split(snli["test"])
    print("clean stats:", stats, "| test:", len(test_pairs), flush=True)
    y_test = [int(l) for _, l in test_pairs]

    # ---- embeddings + keep-tokenizer covering the full pool + test ----
    emb = part2.clslib.PretrainedEmbeddings(
        os.path.join(args.glove_dir, "glove.6B.300d.txt"), num_dims=300)
    print("building keep-tokenizer over full pool + test (one-time)...", flush=True)
    sents = P2.sentences_in(train_pairs_all, test_pairs)
    tok = get_tokenizers(part2, P2, emb, sents)["keep"]

    bpe = Tokenizer.from_file(args.bpe)
    V = len(bpe.get_vocab())

    def split_pool(pool):
        X = [ph for ph, _ in pool]; y = [lab for _, lab in pool]
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.1, stratify=y, random_state=args.seed)
        return list(zip(Xa, ya)), list(zip(Xb, yb))

    def diffprod_csr(pairs):
        data, indices, indptr = [], [], [0]
        for (p, h), _ in pairs:
            pids = set(bpe.encode(p).ids); hids = set(bpe.encode(h).ids)
            row = {}
            for i in pids:        row[i] = 1.0
            for i in hids:        row[V + i] = 1.0
            for i in pids ^ hids: row[2 * V + i] = 1.0
            for i in pids & hids: row[3 * V + i] = 1.0
            for idx, val in row.items():
                indices.append(idx); data.append(val)
            indptr.append(len(indices))
        return csr_matrix((np.asarray(data, np.float32), np.asarray(indices, np.int64),
                           np.asarray(indptr, np.int64)), shape=(len(pairs), 4 * V))

    def test_preds_embed(spec, pool, tag):
        train_set, valsel = split_pool(pool)
        save_to = os.path.join(args.results, f"final_{tag}")
        print(f"[{tag}] training embed on {len(train_set)}...", flush=True)
        PP.train(train_set, valsel, tok, emb, spec["enc_cfg"], spec["combo"],
                 spec["head_cfg"], spec["opt_cfg"], 3, device=device, save_to=save_to, seed=args.seed)
        model = PP.load(save_to, emb, device)
        pad = emb.vocab.index(emb.pad_token); unk = emb.vocab.default_index
        ds = PP.PairCorpus(test_pairs, tok, unk, joined=(spec["combo"] == "joined"), sep_id=unk)
        dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=PP.make_collate(pad))
        out = PP.evaluate(model, dl, nn.NLLLoss(), device)
        return [int(p) for b in out["predictions"] for p in b]

    def test_preds_sparse(pool, tag):
        train_set, valsel = split_pool(pool)
        print(f"[{tag}] encoding + training sparse on {len(train_set)}...", flush=True)
        Xtr, ytr = diffprod_csr(train_set), [int(l) for _, l in train_set]
        Xvs, yvs = diffprod_csr(valsel),   [int(l) for _, l in valsel]
        Xte = diffprod_csr(test_pairs)
        trl = DataLoader(SparseDataset(Xtr, ytr), batch_size=64, shuffle=True, collate_fn=sparse_collate)
        vsl = DataLoader(SparseDataset(Xvs, yvs), batch_size=64, shuffle=False, collate_fn=sparse_collate)
        tel = DataLoader(SparseDataset(Xte, y_test), batch_size=64, shuffle=False, collate_fn=sparse_collate)
        P2.set_all_seeds(args.seed)
        model = SparseMLP(4 * V, 128, 3, dropout=0.1).to(device)
        crit = nn.NLLLoss(); opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        best_loss, best_state = None, None
        for _ in range(10):
            train_one_epoch(model, trl, opt, crit, device)
            va, _ = sparse_evaluate(model, vsl, crit, device)
            if best_loss is None or va["loss"] <= best_loss:
                best_loss = va["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        _, preds = sparse_evaluate(model, tel, crit, device)
        return [int(p) for p in preds]

    def metrics(preds, tag, train_size):
        acc = accuracy_score(y_test, preds)
        mf1 = f1_score(y_test, preds, average="macro")
        rep = classification_report(y_test, preds, labels=[0, 1, 2], target_names=LABELS,
                                    digits=4, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=[0, 1, 2]).tolist()
        print(f"  {tag:22s} train={train_size:>6d}  TEST acc={acc:.4f}  macroF1={mf1:.4f}", flush=True)
        return {"system": tag, "train_size": train_size, "test_acc": acc, "macro_f1": mf1,
                "per_class": {k: rep[k] for k in LABELS}, "confusion": cm, "label_order": LABELS}

    pooled = SW.best_specs()["pooled"]
    recur = SW.best_specs()["recurrent"]
    hyp = copy.deepcopy(pooled); hyp["combo"] = "hypothesis_only"

    results = []
    # ---- Tier A: trained on the 40k working set ----
    print("\n=== Tier A (40k working set) ===", flush=True)
    results.append(metrics(test_preds_embed(pooled, working_set, "pooled_40k"), "pooled_40k", len(working_set)))
    results.append(metrics(test_preds_embed(recur, working_set, "recurrent_40k"), "recurrent_40k", len(working_set)))
    results.append(metrics(test_preds_embed(hyp, working_set, "hyponly_40k"), "hyponly_40k", len(working_set)))
    results.append(metrics(test_preds_sparse(working_set, "sparse_40k"), "sparse_40k", len(working_set)))

    # ---- Tier B: trained on the full pool (numbers next to published) ----
    print("\n=== Tier B (full ~549k pool) ===", flush=True)
    results.append(metrics(test_preds_embed(recur, train_pairs_all, "recurrent_full"), "recurrent_full", len(train_pairs_all)))
    results.append(metrics(test_preds_embed(pooled, train_pairs_all, "pooled_full"), "pooled_full", len(train_pairs_all)))
    results.append(metrics(test_preds_sparse(train_pairs_all, "sparse_full"), "sparse_full", len(train_pairs_all)))

    out_path = os.path.join(args.results, "final_test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", out_path, flush=True)
    print("\nSUMMARY (test accuracy):")
    for r in results:
        print(f"  {r['system']:16s} {r['test_acc']:.4f}  (macroF1 {r['macro_f1']:.4f})")


if __name__ == "__main__":
    main()
