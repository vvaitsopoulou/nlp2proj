"""
NLP1 Project 2 — Step 3: the sentence-pair pipeline (the training engine).

A shared encoder is applied to the premise and to the hypothesis; a combination
rule fuses the two vectors; the tutorial-style MLP head classifies. Everything
mirrors part2.experiment so Project 1's analysis code and our Step 0 logging /
checkpoint / seeding drop in unchanged.

Encoders (one interface, enc_dim + forward([B,L]) -> [B, enc_dim]):
  PoolEncoder : mean / sum / max   (mean & sum are numerically the tutorial's)
  RNNEncoder  : lstm / gru, uni / bi, output last / maxpool, word dropout

Combination rules:
  concat           : [u, v]                     (2 * enc_dim)
  nli              : [u, v, |u-v|, u*v]         (4 * enc_dim)
  hypothesis_only  : [v]                        (1 * enc_dim)  -- ignores premise
"""
from __future__ import annotations

import itertools
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


# ==========================================================================
# Encoders
# ==========================================================================
class PoolEncoder(nn.Module):
    def __init__(self, embedding_matrix, pool="mean", padding_idx=0, freeze=True):
        super().__init__()
        assert pool in ("mean", "sum", "max", "meanmax")
        self.pool = pool
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze, padding_idx=padding_idx)

    @property
    def enc_dim(self):
        d = self.embedding.embedding_dim
        return 2 * d if self.pool == "meanmax" else d

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def _mean(self, e, mask):
        den = mask.sum(-1, keepdim=True).clamp_min(1)
        return e.sum(-2) / den

    def _max(self, e, mask):
        e = e.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        out = e.max(dim=-2).values
        return torch.where(torch.isinf(out), torch.zeros_like(out), out)

    def forward(self, x):
        e = self.embedding(x)                      # [B, L, D]; pad rows are zero
        mask = x.ne(self.pad_idx)                  # [B, L]
        if self.pool == "sum":
            return e.sum(-2)
        if self.pool == "mean":
            return self._mean(e, mask)
        if self.pool == "max":
            return self._max(e, mask)
        return torch.cat([self._mean(e, mask), self._max(e, mask)], dim=-1)   # meanmax


class RNNEncoder(nn.Module):
    def __init__(self, embedding_matrix, cell="lstm", hidden=128, bidirectional=False,
                 output="last", padding_idx=0, freeze=True, word_dropout=0.0):
        super().__init__()
        assert cell in ("lstm", "gru") and output in ("last", "maxpool")
        _, dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze, padding_idx=padding_idx)
        self.cell = cell
        self.bi = bidirectional
        self.output = output
        self.hidden = hidden
        self.word_dropout = word_dropout
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(dim, hidden, num_layers=1, batch_first=True,
                           bidirectional=bidirectional)

    @property
    def enc_dim(self):
        return self.hidden * (2 if self.bi else 1)

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):
        emb = self.embedding(x)
        mask = x.ne(self.pad_idx)
        if self.training and self.word_dropout > 0:
            drop = (torch.rand(x.shape, device=x.device) < self.word_dropout) & mask
            emb = emb.masked_fill(drop.unsqueeze(-1), 0.0)
        lengths = mask.sum(-1)                                  # [B], guaranteed >= 1
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out_packed, hidden = self.rnn(packed)
        if self.output == "last":
            h = hidden[0] if self.cell == "lstm" else hidden    # [num_dir, B, H]
            if self.bi:
                return torch.cat([h[-2], h[-1]], dim=-1)        # [B, 2H]
            return h[-1]                                        # [B, H]
        seq, lens = pad_packed_sequence(out_packed, batch_first=True)   # [B, L, enc_dim]
        m = torch.arange(seq.size(1), device=x.device)[None, :] < lens.to(x.device)[:, None]
        seq = seq.masked_fill(~m.unsqueeze(-1), float("-inf"))
        out = seq.max(dim=1).values
        return torch.where(torch.isinf(out), torch.zeros_like(out), out)


# ==========================================================================
# Head + pair model
# ==========================================================================
COMBO_MULT = {"concat": 2, "nli": 4, "hypothesis_only": 1, "joined": 1}


class MLPHead(nn.Module):
    """Tutorial-style log-prob MLP, with an input BatchNorm added.

    Why the BatchNorm: pooled GloVe vectors have large, uneven norms that
    saturate the first Tanh and stall learning (the head could not even reach a
    linear probe's accuracy without it). Normalising the combined feature — the
    learnable equivalent of StandardScaler — removes the saturation. Set
    input_norm=False to recover the exact tutorial head; activation='relu' is
    also available."""

    def __init__(self, in_dim, hidden1, hidden2, num_classes, dropout=0.0,
                 input_norm=True, activation="tanh"):
        super().__init__()
        act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        layers = []
        if input_norm:
            layers.append(nn.BatchNorm1d(in_dim))
        layers += [
            nn.Dropout(dropout), nn.Linear(in_dim, hidden1), act(),
            nn.Dropout(dropout), nn.Linear(hidden1, hidden2), act(),
            nn.Dropout(dropout), nn.Linear(hidden2, num_classes),
            nn.LogSoftmax(dim=-1)]
        self.net = nn.Sequential(*layers)

    def forward(self, f):
        return self.net(f)


class PairClassifier(nn.Module):
    def __init__(self, encoder, combo, num_classes, hidden1=256, hidden2=256, dropout=0.1):
        super().__init__()
        assert combo in COMBO_MULT
        self.encoder = encoder
        self.combo = combo
        self.head = MLPHead(encoder.enc_dim * COMBO_MULT[combo],
                            hidden1, hidden2, num_classes, dropout)

    def forward(self, xp, xh):
        # 'joined' feeds the premise+SEP+hypothesis sequence through the hypothesis
        # slot (see PairCorpus), so it is encoded exactly like hypothesis_only.
        if self.combo in ("hypothesis_only", "joined"):
            feat = self.encoder(xh)                 # premise slot is never read
        else:
            u = self.encoder(xp)
            v = self.encoder(xh)
            if self.combo == "concat":
                feat = torch.cat([u, v], dim=-1)
            else:  # nli
                feat = torch.cat([u, v, (u - v).abs(), u * v], dim=-1)
        return self.head(feat)


def make_encoder(emb_model, enc_cfg):
    V, D = emb_model.vectors.shape
    pad = emb_model.vocab.index(emb_model.pad_token)
    if enc_cfg.get("random_init"):
        # trainable embeddings initialised from scratch (global RNG -> per-run seed controls it)
        mat = torch.randn(V, D) * 0.1
        mat[pad] = 0.0
        freeze = False
    else:
        mat = torch.tensor(emb_model.vectors, dtype=torch.float32)
        freeze = enc_cfg.get("freeze", True)
    t = enc_cfg["type"]
    if t == "pool":
        return PoolEncoder(mat, pool=enc_cfg.get("pool", "mean"), padding_idx=pad, freeze=freeze)
    if t == "rnn":
        return RNNEncoder(mat, cell=enc_cfg.get("cell", "lstm"),
                          hidden=enc_cfg.get("hidden", 128),
                          bidirectional=enc_cfg.get("bi", False),
                          output=enc_cfg.get("output", "last"),
                          padding_idx=pad, freeze=freeze,
                          word_dropout=enc_cfg.get("word_dropout", 0.0))
    raise ValueError(f"unknown encoder type: {t}")


def build_model(emb_model, enc_cfg, combo, head_cfg, num_classes, device="cpu"):
    model = PairClassifier(make_encoder(emb_model, enc_cfg), combo, num_classes, **head_cfg)
    return model.to(torch.device(device) if isinstance(device, str) else device)


# ==========================================================================
# Data
# ==========================================================================
class PairCorpus(Dataset):
    """Premise ids + hypothesis ids + label. Empty token lists are replaced by
    [unk_id] so pack_padded_sequence never sees a zero-length sequence."""

    def __init__(self, pairs, tokenizer, unk_id, joined=False, sep_id=None):
        self.P, self.H, ys = [], [], []
        sep = unk_id if sep_id is None else sep_id
        for (p, h), lab in pairs:
            pi = tokenizer.ids(p) or [unk_id]
            hi = tokenizer.ids(h) or [unk_id]
            if joined:                                   # premise + SEP + hypothesis -> hyp slot
                self.P.append(torch.tensor([unk_id], dtype=torch.long))   # dummy, unused
                self.H.append(torch.tensor(pi + [sep] + hi, dtype=torch.long))
            else:
                self.P.append(torch.tensor(pi, dtype=torch.long))
                self.H.append(torch.tensor(hi, dtype=torch.long))
            ys.append(int(lab))
        self.y = torch.tensor(ys, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.P[i], self.H[i], self.y[i]


def make_collate(pad_id):
    def collate(batch):
        P, H, Y = zip(*batch)
        xp = pad_sequence(P, batch_first=True, padding_value=pad_id)
        xh = pad_sequence(H, batch_first=True, padding_value=pad_id)
        return xp, xh, torch.stack(Y)
    return collate


# ==========================================================================
# Train / evaluate (mirror part2.experiment)
# ==========================================================================
def training_one_epoch(model, dl, opt, crit, device):
    model.train()
    tot_loss = tot_correct = n = 0
    for xp, xh, y in dl:
        xp, xh, y = xp.to(device), xh.to(device), y.to(device)
        opt.zero_grad()
        logp = model(xp, xh)
        loss = crit(logp, y)
        loss.backward()
        opt.step()
        tot_loss += loss.item() * y.size(0)
        tot_correct += (logp.argmax(1) == y).sum().item()
        n += y.size(0)
    return {"loss": tot_loss / n, "accuracy": tot_correct / n}


def evaluate(model, dl, crit, device):
    model.eval()
    tot_loss = tot_correct = n = 0
    preds, labels = [], []
    with torch.no_grad():
        for xp, xh, y in dl:
            xp, xh, y = xp.to(device), xh.to(device), y.to(device)
            logp = model(xp, xh)
            loss = crit(logp, y)
            tot_loss += loss.item() * y.size(0)
            p = logp.argmax(1)
            tot_correct += (p == y).sum().item()
            n += y.size(0)
            preds.append(p.tolist())
            labels.append(y.tolist())
    return {"predictions": preds, "labels": labels,
            "loss": tot_loss / n, "accuracy": tot_correct / n}


def train(training_set, validation_set, tokenizer, emb_model,
          enc_cfg, combo, head_cfg, opt_cfg, num_classes,
          device="cpu", save_to=None, seed=None):
    if seed is not None:
        import project2_lib as P2
        P2.set_all_seeds(seed)

    device = torch.device(device) if isinstance(device, str) else device
    pad = emb_model.vocab.index(emb_model.pad_token)
    unk = emb_model.vocab.default_index
    collate = make_collate(pad)
    joined = (combo == "joined")

    tr = DataLoader(PairCorpus(training_set, tokenizer, unk, joined=joined, sep_id=unk),
                    batch_size=opt_cfg["batch_size"], shuffle=True, collate_fn=collate,
                    drop_last=True)   # drop the size-1 tail so BatchNorm never sees it
    va = DataLoader(PairCorpus(validation_set, tokenizer, unk, joined=joined, sep_id=unk),
                    batch_size=opt_cfg["batch_size"], shuffle=False, collate_fn=collate)

    model = build_model(emb_model, enc_cfg, combo, head_cfg, num_classes, device)
    crit = nn.NLLLoss()
    # Trainable embeddings (fine-tune / random init) get their OWN group: no weight
    # decay (so pretrained vectors aren't pulled to zero) and an optional emb_lr.
    # Frozen embeddings have requires_grad=False and never reach the optimizer.
    emb_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and "encoder.embedding" in n]
    main_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "encoder.embedding" not in n]
    groups = [{"params": main_params, "weight_decay": opt_cfg["weight_decay"]}]
    if emb_params:
        groups.append({"params": emb_params, "weight_decay": 0.0,
                       "lr": opt_cfg.get("emb_lr", opt_cfg["learning_rate"])})
    opt = torch.optim.AdamW(groups, lr=opt_cfg["learning_rate"])

    if save_to:
        with open(f"{save_to}.json", "w", encoding="utf-8") as f:
            json.dump({"enc_cfg": enc_cfg, "combo": combo, "head_cfg": head_cfg,
                       "num_classes": num_classes}, f, indent=2)

    metrics, best_epoch, best_loss = [], 0, None
    for epoch in range(opt_cfg["num_epochs"]):
        tm = training_one_epoch(model, tr, opt, crit, device)
        vm = evaluate(model, va, crit, device)
        metrics.append({"train_metrics": tm, "val_metrics": vm})
        if best_loss is None or vm["loss"] <= best_loss:        # best-by-val-loss, as part2
            best_loss, best_epoch = vm["loss"], epoch
            if save_to:
                torch.save(model.state_dict(), f"{save_to}.pt")
    return metrics, best_epoch


def load(path, emb_model, device="cpu"):
    with open(f"{path}.json", encoding="utf-8") as f:
        cfg = json.load(f)
    model = build_model(emb_model, cfg["enc_cfg"], cfg["combo"], cfg["head_cfg"],
                        cfg["num_classes"], device)
    model.load_state_dict(torch.load(f"{path}.pt", map_location=device))
    return model


# ==========================================================================
# Cross-validation with per-fold seeding, logging, and resumable checkpoint.
# Returns a list over seeds of {'predictions': {fold: np.array}, 'accuracies':
# {fold: acc}, 'models': {fold: path}} — the Project 1 grid structure.
# ==========================================================================
def cross_validation(system_key, cv_sets, tokenizer, emb_model,
                     enc_cfg, combo, head_cfg, opt_cfg, num_classes,
                     results_folder, checkpoint, seeds, device="cpu"):
    import project2_lib as P2

    per_seed = []
    for seed in seeds:
        preds, accs, models = OrderedDict(), OrderedDict(), OrderedDict()
        for fold in cv_sets[seed]:
            key = f"{system_key}|seed{seed}|fold{fold}"
            folder = f"{results_folder}/{system_key}/seed-{seed}/fold-{fold}"
            if key in checkpoint:
                rec = checkpoint.get(key)
            else:
                P2.set_all_seeds(P2.run_seed(seed, fold))
                Path(folder).mkdir(parents=True, exist_ok=True)
                with P2.Timer() as timer:
                    metrics, best = train(
                        cv_sets[seed][fold]["training"],
                        cv_sets[seed][fold]["validation"],
                        tokenizer, emb_model, enc_cfg, combo, head_cfg, opt_cfg,
                        num_classes, device=device, save_to=f"{folder}/model", seed=None)
                vm = metrics[best]["val_metrics"]
                flat = np.array(list(itertools.chain.from_iterable(vm["predictions"])))
                rec = {"accuracy": vm["accuracy"], "predictions": flat, "path": f"{folder}/model"}
                P2.log_run(f"{folder}/model",
                           {"system": system_key, "cv_seed": seed, "fold": fold,
                            "run_seed": P2.run_seed(seed, fold), "enc_cfg": enc_cfg,
                            "combo": combo, "head_cfg": head_cfg, "opt_cfg": opt_cfg},
                           metrics, best, timer.sec)
                checkpoint.set(key, rec)
                print(f"    {system_key[:30]:30s} seed{seed} fold{fold}  "
                      f"{timer.sec:6.0f}s  val_acc={rec['accuracy']:.4f}", flush=True)
            preds[fold] = rec["predictions"]
            accs[fold] = rec["accuracy"]
            models[fold] = rec["path"]
        per_seed.append({"predictions": preds, "accuracies": accs, "models": models})
    return per_seed


# ==========================================================================
# Smoke tests — the correctness checks the plan lists. Works on any emb_model
# with .vectors, .vocab.index, .vocab.default_index, .pad_token.
# ==========================================================================
def run_smoke_tests(emb_model, device="cpu", tol=1e-5, verbose=True):
    device = torch.device(device) if isinstance(device, str) else device
    pad = emb_model.vocab.index(emb_model.pad_token)
    results = {}

    def log(name, ok, detail=""):
        results[name] = ok
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    head_cfg = dict(hidden1=32, hidden2=32, dropout=0.0)
    # two synthetic id sequences of different lengths (ids >= 2, i.e. real words)
    xp = torch.tensor([[2, 3, 4, 5], [6, 7, pad, pad]], device=device)
    xh = torch.tensor([[8, 9, pad, pad], [3, 4, 5, 6]], device=device)

    # 1) batched forward == per-example forward (eval mode, no dropout)
    import project2_lib as P2
    P2.set_all_seeds(0)
    m = build_model(emb_model, {"type": "pool", "pool": "mean"}, "nli", head_cfg, 3, device).eval()
    batched = m(xp, xh)
    singles = torch.cat([m(xp[i:i+1], xh[i:i+1]) for i in range(xp.size(0))], 0)
    log("batched == single (pool)", torch.allclose(batched, singles, atol=1e-4),
        f"max|diff|={(batched-singles).abs().max().item():.2e}")

    # 2) hypothesis-only ignores the premise
    P2.set_all_seeds(0)
    mh = build_model(emb_model, {"type": "pool", "pool": "mean"}, "hypothesis_only",
                     head_cfg, 3, device).eval()
    out_a = mh(xp, xh)
    out_b = mh(torch.full_like(xp, 2), xh)          # completely different premise
    log("hypothesis_only ignores premise", torch.allclose(out_a, out_b, atol=tol),
        f"max|diff|={(out_a-out_b).abs().max().item():.2e}")

    # 3) pooling is order-invariant; the RNN is not
    seq = torch.tensor([[2, 3, 4, 5]], device=device)
    perm = torch.tensor([[5, 4, 3, 2]], device=device)
    pool_enc = make_encoder(emb_model, {"type": "pool", "pool": "mean"}).to(device).eval()
    same = torch.allclose(pool_enc(seq), pool_enc(perm), atol=1e-5)
    P2.set_all_seeds(0)
    rnn_enc = make_encoder(emb_model, {"type": "rnn", "cell": "lstm", "hidden": 16}).to(device).eval()
    differ = not torch.allclose(rnn_enc(seq), rnn_enc(perm), atol=1e-4)
    log("pool order-invariant, RNN order-sensitive", same and differ,
        f"pool_same={same} rnn_differs={differ}")

    # 4) same seed reproduces the model
    P2.set_all_seeds(123)
    m1 = build_model(emb_model, {"type": "rnn", "cell": "gru", "hidden": 16}, "nli", head_cfg, 3, device).eval()
    P2.set_all_seeds(123)
    m2 = build_model(emb_model, {"type": "rnn", "cell": "gru", "hidden": 16}, "nli", head_cfg, 3, device).eval()
    log("seed reproduces model", torch.allclose(m1(xp, xh), m2(xp, xh), atol=tol))

    # 5) empty-sequence guard: a sentence that tokenizes to [] becomes [unk]
    class StubTok:
        def __init__(self, empty):
            self.empty = empty
        def ids(self, text):
            return [] if text in self.empty else [2, 3]
    unk = emb_model.vocab.default_index
    ds = PairCorpus([(("", "hi"), 0)], StubTok(empty={""}), unk)
    xp2, xh2, y2 = make_collate(pad)([ds[0]])
    renc = make_encoder(emb_model, {"type": "rnn", "cell": "lstm", "hidden": 8}).to(device).eval()
    ok_guard = True
    try:
        renc(xp2.to(device))                        # would crash if length 0
    except Exception as exc:                          # noqa
        ok_guard = False
        log("empty-sequence guard", False, str(exc))
    if ok_guard:
        log("empty-sequence guard", (xp2 == unk).all().item(), "empty -> [unk]")

    # 6) all three combos give the expected feature width
    widths_ok = True
    for combo, mult in COMBO_MULT.items():
        enc = make_encoder(emb_model, {"type": "pool", "pool": "max"})
        mm = PairClassifier(enc, combo, 3, **head_cfg).to(device).eval()
        first_linear = next(m for m in mm.head.net if isinstance(m, nn.Linear))
        widths_ok &= (first_linear.in_features == enc.enc_dim * mult)
    log("combo feature widths", widths_ok)

    passed = sum(results.values())
    if verbose:
        print(f"\n{passed}/{len(results)} smoke tests passed")
    return results
