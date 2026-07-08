"""Score a prediction file against gold labels and explanations.

Reads a JSONL file with a ``response`` column (model output) and a ``labels``
column (gold reference), as written by ``swift infer``. Two formats are
handled automatically:

  text     both columns contain "Label: ...\\nExplanation: ..." blocks,
           optionally preceded by a <think> trace
  seq_cls  both columns contain integer class indices

Outputs metrics.json (accuracy, macro/weighted precision/recall/F1 and, with
--has_explanation, BERTScore / ROUGE / BLEU / METEOR over the explanations)
plus a confusion-matrix plot.

Predictions with a missing or invalid label are replaced by a random valid
label (seeded, so results are reproducible) — this penalizes malformed output
instead of silently dropping it.

Example:
    python evaluation/compute_metrics.py \
        --data results/hateful/grpo_think.jsonl \
        --out_dir scores/hateful/grpo_think --has_explanation
"""

import argparse
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from bert_score import score as bertscore
from nltk import download as nltk_download
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from transformers import AutoTokenizer

for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
    nltk_download(pkg, quiet=True)

LABEL_RE = re.compile(r"Label:\s*([^\n]+)", re.IGNORECASE)
EXPL_RE = re.compile(r"Explanation:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
MAX_WORDS = 1024


def normalize_label(label):
    if label is None or pd.isna(label):
        return None
    label = str(label).strip().lower()
    label = re.sub(r"\*+", "", label)
    label = re.sub(r"^[-•·●▪▫◦◘◙○◌]\s*", "", label)
    label = re.sub(r"\s+", " ", label).strip().replace("_", "-")
    if label.startswith("non-"):
        label = "not-" + label[4:]
    return label


def extract_label_and_explanation(text):
    if pd.isna(text):
        return None, None
    text = str(text)
    label = LABEL_RE.search(text)
    explanation = EXPL_RE.search(text)
    return (normalize_label(label.group(1)) if label else None,
            explanation.group(1).strip() if explanation else None)


def classification_metrics(y_true, y_pred, cm_path=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if cm_path:
        labels = sorted(set(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="g", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.close()
    return metrics


def explanation_metrics(preds, refs, arabic=False):
    preds = [" ".join(str(p).split()[:MAX_WORDS]) or "[empty]" for p in preds]
    refs = [" ".join(str(r).split()[:MAX_WORDS]) or "[empty]" for r in refs]

    bert_model = "aubmindlab/bert-base-arabertv2" if arabic else "bert-base-multilingual-uncased"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P, R, F = bertscore(cands=preds, refs=refs, model_type=bert_model,
                        device=device, num_layers=12, batch_size=32, verbose=False)

    rouge_tokenizer = AutoTokenizer.from_pretrained(
        "aubmindlab/bert-base-arabertv2" if arabic else "bert-base-uncased")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=rouge_tokenizer)
    rouge = [scorer.score(r, p) for p, r in zip(preds, refs)]

    bleu = corpus_bleu([[word_tokenize(r)] for r in refs],
                       [word_tokenize(p) for p in preds],
                       smoothing_function=SmoothingFunction().method1)
    meteor = [meteor_score([word_tokenize(r)], word_tokenize(p)) for p, r in zip(preds, refs)]

    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F.mean().item(),
        "rouge1": float(np.mean([s["rouge1"].fmeasure for s in rouge])),
        "rouge2": float(np.mean([s["rouge2"].fmeasure for s in rouge])),
        "rougeL": float(np.mean([s["rougeL"].fmeasure for s in rouge])),
        "bleu": bleu,
        "meteor": float(np.mean(meteor)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="prediction JSONL file")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--has_explanation", action="store_true",
                        help="also score explanations (BERTScore/ROUGE/BLEU/METEOR)")
    parser.add_argument("--is_arabic", action="store_true",
                        help="use AraBERT for BERTScore/ROUGE tokenization")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for the random fallback on invalid predictions")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(args.data, lines=True)[["response", "labels"]]
    seq_cls = isinstance(df.iloc[0]["response"], (int, np.integer))

    if seq_cls:
        gold = df["labels"].astype(str)
        pred = df["response"].astype(str)
        has_explanation = False
    else:
        gold, gold_expl = zip(*df["labels"].map(extract_label_and_explanation))
        pred, pred_expl = zip(*df["response"].map(extract_label_and_explanation))
        gold, pred = pd.Series(gold), pd.Series(pred)
        has_explanation = args.has_explanation

    valid_labels = sorted(set(gold.dropna()))
    if not valid_labels:
        raise SystemExit("no gold labels found")
    invalid = (~pred.isin(valid_labels)).sum()
    if invalid and not seq_cls:
        print(f"{invalid}/{len(pred)} predictions had no valid label; assigning random labels")
        pred = pred.map(lambda x: x if x in valid_labels else random.choice(valid_labels))

    metrics = classification_metrics(gold, pred, cm_path=out_dir / "confusion_matrix.png")
    metrics["invalid_predictions"] = int(invalid)

    if has_explanation:
        metrics.update(explanation_metrics(
            [e or "" for e in pred_expl], [e or "" for e in gold_expl], arabic=args.is_arabic))

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
