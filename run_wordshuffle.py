"""
Extension 1 — word-shuffle test (after Sinha et al. 2021). Scramble the HYPOTHESIS
word order at evaluation and measure each system's accuracy drop on the test set.

Control: pooling is order-invariant (a bag is a bag), so its accuracy must stay
flat — that proves the shuffle is applied and isolates the order effect. Only the
recurrent model can move; its drop = how much it actually uses word order.

Reuses the Step-9 trained models (results_p2/final_*.pt). Inference only, minutes.

    python -u run_wordshuffle.py --glove_dir ./glove --results ./results_p2
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove_dir", required=True)
    ap.add_argument("--results", default="results_p2")
    ap.add_argument("--tier", default="full", choices=["full", "40k"],
                    help="which saved Step-9 models to use (full = 549k-trained)")
    ap.add_argument("--n_shuffle", type=int, default=3, help="shuffle seeds to average over")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    from datasets import load_dataset
    import part2
    import project2_lib as P2
    import project2_pairs as PP
    import project2_sweep as SW

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    snli = load_dataset("stanfordnlp/snli")
    test_pairs = P2.clean_eval_split(snli["test"])
    y_test = [int(l) for _, l in test_pairs]
    print("test:", len(test_pairs), flush=True)

    emb = part2.clslib.PretrainedEmbeddings(
        os.path.join(args.glove_dir, "glove.6B.300d.txt"), num_dims=300)
    # plain (live) tokenizer so shuffled hypotheses tokenize correctly; identical ids to the cached one
    bt = part2.clslib.BasicTokenizer(lowercase=True, remove_punct=False,
                                     remove_stopwords=False, language="english", vocab=emb.vocab)
    pad = emb.vocab.index(emb.pad_token)
    unk = emb.vocab.default_index

    def shuffled_hyp(pairs, seed):
        rng = random.Random(seed)
        out = []
        for (p, h), y in pairs:
            w = h.split()
            rng.shuffle(w)
            out.append(((p, " ".join(w)), y))
        return out

    def accuracy(model, pairs, combo):
        ds = PP.PairCorpus(pairs, bt, unk, joined=(combo == "joined"), sep_id=unk)
        dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=PP.make_collate(pad))
        out = PP.evaluate(model, dl, nn.NLLLoss(), device)
        preds = [int(p) for b in out["predictions"] for p in b]
        return float(np.mean([p == g for p, g in zip(preds, y_test)]))

    def load_or_train(name, spec):
        path = os.path.join(args.results, f"final_{name}_{args.tier}")
        if os.path.exists(path + ".pt"):
            print(f"[{name}] loading {path}", flush=True)
            return PP.load(path, emb, device)
        print(f"[{name}] no saved model at {path}; training a quick 40k fallback...", flush=True)
        ws = P2.build_working_set(P2.clean_training_pool(snli["train"])[0])
        X = [ph for ph, _ in ws]; y = [l for _, l in ws]
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.1, stratify=y, random_state=0)
        PP.train(list(zip(Xa, ya)), list(zip(Xb, yb)), bt, emb, spec["enc_cfg"], spec["combo"],
                 spec["head_cfg"], spec["opt_cfg"], 3, device=device, save_to=path + "_fb", seed=0)
        return PP.load(path + "_fb", emb, device)

    systems = {"recurrent": SW.best_specs()["recurrent"], "pooled": SW.best_specs()["pooled"]}
    print(f"\n{'system':12s} {'normal':>8s} {'shuffled':>10s} {'drop':>8s}")
    for name, spec in systems.items():
        model = load_or_train(name, spec)
        base = accuracy(model, test_pairs, spec["combo"])
        sh = [accuracy(model, shuffled_hyp(test_pairs, s), spec["combo"]) for s in range(args.n_shuffle)]
        sh_mean = float(np.mean(sh))
        print(f"{name:12s} {base:>8.4f} {sh_mean:>10.4f} {base - sh_mean:>+8.4f}   "
              f"(shuffle runs: {[round(x,4) for x in sh]})", flush=True)

    print("\nReading: pooled drop should be ~0.000 (order-invariant control). "
          "The recurrent drop measures how much it actually uses hypothesis word order.", flush=True)


if __name__ == "__main__":
    main()
