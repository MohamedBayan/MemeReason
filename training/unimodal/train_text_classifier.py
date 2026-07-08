"""Fine-tune a text-only classifier on meme text (paper Sec. 5.3).

Reads {train,dev,test}.jsonl with a text field and a label field, fine-tunes an
AutoModelForSequenceClassification, and writes per-split metrics and per-sample
class probabilities under --result_dir.

Example:
    python train_text_classifier.py \
        --data_dir data/hateful --text_field text --label_field class_label \
        --model_name bert-base-multilingual-cased \
        --output_dir checkpoints/unimodal/text/mbert \
        --result_dir results/unimodal/text/mbert
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class JsonlTextDataset(Dataset):
    def __init__(self, path, text_field, label_field, tokenizer, max_length, label2id):
        with open(path, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.text_field = text_field
        self.label_field = label_field
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item[self.text_field],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": self.label2id[item[self.label_field]],
        }


def collate(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def predict_and_save(trainer, dataset, id2label, path):
    predictions = trainer.predict(dataset)
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    rows = [
        {
            "text": item[dataset.text_field],
            "true_label": item[dataset.label_field],
            "predicted_label": id2label[int(np.argmax(p))],
            "probabilities": {id2label[i]: float(p[i]) for i in range(len(p))},
        }
        for item, p in zip(dataset.data, probs)
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="directory with train/dev/test.jsonl")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--label_field", default="class_label")
    parser.add_argument("--model_name", default="bert-base-multilingual-cased")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--metric_for_best_model", default="accuracy")
    args = parser.parse_args()

    splits = {s: os.path.join(args.data_dir, f"{s}.jsonl") for s in ("train", "dev", "test")}
    labels = sorted({
        json.loads(line)[args.label_field]
        for line in open(splits["train"], encoding="utf-8") if line.strip()
    })
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    print(f"labels: {labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    datasets = {
        split: JsonlTextDataset(path, args.text_field, args.label_field,
                                tokenizer, args.max_seq_length, label2id)
        for split, path in splits.items()
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=True,
            remove_unused_columns=False,
            report_to="none",
        ),
        data_collator=collate,
        compute_metrics=compute_metrics,
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        processing_class=tokenizer,
    )
    trainer.train()

    os.makedirs(args.result_dir, exist_ok=True)
    dev_metrics = trainer.evaluate(datasets["dev"])
    predict_and_save(trainer, datasets["dev"], id2label,
                     os.path.join(args.result_dir, "dev_probabilities.json"))
    test_predictions = predict_and_save(trainer, datasets["test"], id2label,
                                        os.path.join(args.result_dir, "test_probabilities.json"))
    test_metrics = compute_metrics((test_predictions.predictions, test_predictions.label_ids))

    summary = {
        "model_name": args.model_name,
        "labels": labels,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
    }
    with open(os.path.join(args.result_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
