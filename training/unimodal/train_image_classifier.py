"""Fine-tune an image-only classifier on meme images (paper Sec. 5.3).

Reads {train,dev,test}.jsonl with an image-path field and a label field,
fine-tunes an AutoModelForImageClassification, and writes per-split metrics
and per-sample class probabilities under --result_dir.

Example:
    python train_image_classifier.py \
        --data_dir data/hateful --image_field img_path --label_field class_label \
        --model_name google/vit-base-patch16-224 \
        --output_dir checkpoints/unimodal/image/vit \
        --result_dir results/unimodal/image/vit
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)


class JsonlImageDataset(Dataset):
    def __init__(self, path, image_field, label_field, processor, label2id):
        with open(path, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.image_field = image_field
        self.label_field = label_field
        self.processor = processor
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item[self.image_field]).convert("RGB")
        except Exception as e:
            print(f"failed to load {item[self.image_field]}: {e}")
            image = Image.new("RGB", (224, 224))
        inputs = self.processor(image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": self.label2id[item[self.label_field]],
        }


def collate(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
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
            "image_path": item[dataset.image_field],
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
    parser.add_argument("--image_field", default="img_path")
    parser.add_argument("--label_field", default="class_label")
    parser.add_argument("--model_name", default="google/vit-base-patch16-224")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
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

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    datasets = {
        split: JsonlImageDataset(path, args.image_field, args.label_field, processor, label2id)
        for split, path in splits.items()
    }
    model = AutoModelForImageClassification.from_pretrained(
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
        processing_class=processor,
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
