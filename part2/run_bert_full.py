"""
BERT token embeddings + LSTM on the full cleaned SNLI training pool.

Steps: clean pool -> embed with frozen BERT (saved in shards, resumable) -> train one
model on 90% (best epoch picked on the other 10%) -> evaluate on SNLI test and Breaking NLI.

Compute nodes may not have internet, so download everything first on the login node:
    python run_bert_full.py --prepare
Then submit job_bert_full.sh.
"""
import argparse
import json
import re
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

BERT_NAME = "google-bert/bert-base-uncased"
BNLI_URL = "https://raw.githubusercontent.com/BIU-NLP/Breaking_NLI/master/breaking_nli_dataset.zip"
LABEL_NAMES = ["entailment", "neutral", "contradiction"]


class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size, embedding_dim, word_dropout=0.):
        super().__init__()
        self.hidden_size = hidden_size
        self.word_dropout = word_dropout
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)

    @property
    def enc_dim(self):
        return self.hidden_size

    def forward(self, x, mask):
        if self.training and self.word_dropout > 0:
            drop_mask = (torch.rand(mask.shape, device=x.device) < self.word_dropout) & mask
            x = x.masked_fill(drop_mask.unsqueeze(-1), 0.0)
        lengths = torch.sum(mask, dim=-1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(packed)
        return hidden[-1]


class SoftmaxRegressor(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, num_classes, encoder):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(encoder.enc_dim, hidden_size1),
            nn.Tanh(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Tanh(),
            nn.Linear(hidden_size2, num_classes),
        )

    def forward(self, x, mask):
        out = self.mlp(self.encoder(x, mask))
        return torch.nn.functional.log_softmax(out, dim=1)


def collate(batch):
    seqs, ys = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    x = pad_sequence(seqs, batch_first=True).float()
    mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]
    return x, mask, torch.tensor(ys)


def clean_training_pool(snli_train):
    """Same filter as cell 6 of project.ipynb."""
    url_pattern = re.compile(r"https?://|www\.", re.IGNORECASE)
    pair_labels = defaultdict(set)
    for ex in snli_train:
        if ex["label"] != -1:
            pair_labels[(ex["premise"], ex["hypothesis"])].add(ex["label"])

    pairs = []
    for ex in snli_train:
        if ex["label"] == -1:
            continue
        if ex["premise"] == ex["hypothesis"] and ex["label"] != 0:
            continue
        if len(pair_labels[(ex["premise"], ex["hypothesis"])]) > 1:
            continue
        if len(ex["hypothesis"].split()) <= 1:
            continue
        if url_pattern.search(ex["premise"]) or url_pattern.search(ex["hypothesis"]):
            continue
        pairs.append(((ex["premise"], ex["hypothesis"]), ex["label"]))
    return pairs


def clean_eval_split(snli_split):
    """Only removes label noise, like the team's test-set cleaning."""
    return [((ex["premise"], ex["hypothesis"]), ex["label"]) for ex in snli_split
            if ex["label"] != -1 and not (ex["premise"] == ex["hypothesis"] and ex["label"] != 0)]


def embed_pairs(pairs, bert_tokenizer, bert_model, batch_size):
    result = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            encoding = bert_tokenizer(
                [p for (p, h), _ in batch],
                [h for (p, h), _ in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(bert_model.device)
            out = bert_model(**encoding)
            lengths = encoding["attention_mask"].sum(-1)
            for j in range(len(lengths)):
                result.append(out.last_hidden_state[j, :lengths[j]].half().cpu())
    return result


def embed_in_shards(pairs, shard_dir, shard_size, bert_tokenizer, bert_model, batch_size):
    """Shards that already exist are skipped, so a resubmitted job continues where it stopped."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    embeddings = []
    for start in range(0, len(pairs), shard_size):
        path = shard_dir / f"shard_{start // shard_size:03d}.pt"
        if not path.exists():
            shard = embed_pairs(pairs[start:start + shard_size], bert_tokenizer, bert_model, batch_size)
            torch.save(shard, path)
            print(f"embedded {start + len(shard)}/{len(pairs)}", flush=True)
        embeddings.extend(torch.load(path))
    return embeddings


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total = 0., 0, 0
    predictions = []
    with torch.set_grad_enabled(training):
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logprobs = model(x, mask)
            loss = criterion(logprobs, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_correct += (logprobs.argmax(1) == y).sum().item()
            total += y.size(0)
            predictions.extend(logprobs.argmax(1).tolist())
    return {"loss": total_loss / total, "accuracy": total_correct / total, "predictions": predictions}


def breaking_nli_report(rows, preds):
    groups = {
        "synonyms": ["synonyms"],
        "antonyms": ["antonyms", "antonyms_wordnet"],
        "co-hyponyms": ["countries", "nationalities", "colors", "materials", "planets",
                        "drinks", "vegetables", "instruments", "rooms"],
        "numbers": ["cardinals", "ordinals"],
    }
    category_to_group = {c: g for g, cats in groups.items() for c in cats}
    df = pd.DataFrame({
        "premise": [r["sentence1"] for r in rows],
        "hypothesis": [r["sentence2"] for r in rows],
        "category": [r["category"] for r in rows],
        "gold": [r["gold_label"] for r in rows],
        "pred": [LABEL_NAMES[p] for p in preds],
    })
    df["group"] = df["category"].map(category_to_group)
    df["correct"] = df["gold"] == df["pred"]

    summary = {
        "overall_acc": float(df["correct"].mean()),
        "always_contradiction": float((df["gold"] == "contradiction").mean()),
        "pred_distribution": df["pred"].value_counts(normalize=True).round(4).to_dict(),
        "per_group": df.groupby("group")["correct"].mean().round(4).to_dict(),
        "per_category": df.groupby("category")["correct"].mean().round(4).to_dict(),
    }
    return df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="only download SNLI, BERT and Breaking NLI, then exit")
    ap.add_argument("--embeddings_dir", default="bert_full_embeddings")
    ap.add_argument("--results", default="results/bert_lstm_full")
    ap.add_argument("--bnli_dir", default="results/breaking_nli")
    ap.add_argument("--shard_size", type=int, default=50_000)
    ap.add_argument("--bert_batch_size", type=int, default=256)
    ap.add_argument("--num_epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel

    snli = load_dataset("stanfordnlp/snli")
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    bert_model = AutoModel.from_pretrained(BERT_NAME)

    bnli_dir = Path(args.bnli_dir)
    bnli_path = bnli_dir / "data" / "dataset.jsonl"
    if not bnli_path.exists():
        bnli_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BNLI_URL, bnli_dir / "bn.zip")
        with zipfile.ZipFile(bnli_dir / "bn.zip") as z:
            z.extractall(bnli_dir)
    if args.prepare:
        print("SNLI, BERT and Breaking NLI are downloaded")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    bert_model.to(device)
    bert_model.eval()

    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)

    pool = clean_training_pool(snli["train"])
    test_pairs = clean_eval_split(snli["test"])
    print(f"training pool: {len(pool)}  test: {len(test_pairs)}", flush=True)

    embeddings = embed_in_shards(pool, Path(args.embeddings_dir), args.shard_size,
                                 bert_tokenizer, bert_model, args.bert_batch_size)
    labels = [label for _, label in pool]
    data = list(zip(embeddings, labels))

    train_data, val_data = train_test_split(data, test_size=0.1, stratify=labels, random_state=args.seed)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False, collate_fn=collate)

    enc_kwargs = {"hidden_size": 256, "embedding_dim": 768, "word_dropout": 0.1}
    regressor_kwargs = {"hidden_size1": 256, "hidden_size2": 256, "num_classes": 3}
    opt_kwargs = {"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 128, "num_epochs": args.num_epochs}

    torch.manual_seed(args.seed)
    model = SoftmaxRegressor(encoder=LSTMEncoder(**enc_kwargs), **regressor_kwargs).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_kwargs["learning_rate"], weight_decay=opt_kwargs["weight_decay"])

    metrics = []
    best_val_loss, best_epoch = float("inf"), 0
    for epoch in range(args.num_epochs):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        train_metrics.pop("predictions")
        val_metrics.pop("predictions")
        metrics.append({"train_metrics": train_metrics, "val_metrics": val_metrics})
        print(f"Epoch {epoch + 1:02d} | train loss {train_metrics['loss']:.4f} | train acc {train_metrics['accuracy']:.4f} | "
              f"val loss {val_metrics['loss']:.4f} | val acc {val_metrics['accuracy']:.4f}", flush=True)
        if val_metrics["loss"] <= best_val_loss:
            best_val_loss, best_epoch = val_metrics["loss"], epoch
            torch.save(model.state_dict(), results / "model.pt")

    model.load_state_dict(torch.load(results / "model.pt", map_location=device))

    test_embeddings = embed_pairs(test_pairs, bert_tokenizer, bert_model, args.bert_batch_size)
    test_loader = DataLoader(list(zip(test_embeddings, [label for _, label in test_pairs])),
                             batch_size=128, shuffle=False, collate_fn=collate)
    test_acc = run_epoch(model, test_loader, criterion, device)["accuracy"]
    print(f"\nSNLI test acc: {test_acc:.4f}", flush=True)

    bnli_rows = []
    with open(bnli_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["gold_label"] in LABEL_NAMES:
                bnli_rows.append(d)
    bnli_pairs = [((r["sentence1"], r["sentence2"]), LABEL_NAMES.index(r["gold_label"])) for r in bnli_rows]
    bnli_embeddings = embed_pairs(bnli_pairs, bert_tokenizer, bert_model, args.bert_batch_size)
    bnli_loader = DataLoader(list(zip(bnli_embeddings, [label for _, label in bnli_pairs])),
                             batch_size=128, shuffle=False, collate_fn=collate)
    bnli_preds = run_epoch(model, bnli_loader, criterion, device)["predictions"]
    bnli_df, bnli_summary = breaking_nli_report(bnli_rows, bnli_preds)
    bnli_df.to_csv(results / "breaking_nli_predictions.csv", index=False)
    print(json.dumps(bnli_summary, indent=2), flush=True)

    with open(results / "results.json", "w") as f:
        json.dump({"train_size": len(train_data), "val_size": len(val_data), "best_epoch": best_epoch,
                   "metrics": metrics, "snli_test_acc": test_acc, "breaking_nli": bnli_summary,
                   "enc_kwargs": enc_kwargs, "regressor_kwargs": regressor_kwargs, "opt_kwargs": opt_kwargs}, f, indent=2)
    print("saved to", results)


if __name__ == "__main__":
    main()
