"""Build data for the multimodal sequence-classification baselines.

Converts the dataset to the integer-label format expected by
``swift sft --task_type seq_cls``:

  {"messages": [{"role": "user", "content": "<image> <meme text>"}],
   "label": <int>, "images": [<path>]}

Negative classes (not-*) are assigned the lowest indices so that binary tasks
map not-hateful -> 0, hateful -> 1. The label mapping is written next to the
splits as label_mapping.json.

Example:
    python data_prep/prepare_seq_cls_data.py --task hateful \
        --from_hub QCRI/MemeReason --image_root data/hateful_memes/img \
        --output_dir data/formatted/hateful/seq_cls
"""

import argparse
import json
from pathlib import Path

from prepare_training_data import TASKS, load_split_hub, load_split_local, normalize, resolve_image


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from_hub", metavar="REPO")
    source.add_argument("--source_dir")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    cfg = TASKS[args.task]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = sorted(cfg["labels"], key=lambda x: (not x.startswith("not-"), x))
    label2id = {label: i for i, label in enumerate(labels)}
    with open(out / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)
    print(f"label mapping: {label2id}")

    for split in ("train", "dev", "test"):
        if args.from_hub:
            rows = load_split_hub(args.from_hub, cfg["hub_config"], split)
        else:
            rows = load_split_local(args.source_dir, split)
        if rows is None:
            print(f"split {split!r} not found, skipping")
            continue

        records = []
        for row in (normalize(r, args.task) for r in rows):
            text = row["text"].strip()
            records.append({
                "messages": [{"role": "user", "content": f"<image> {text}" if text else "<image>"}],
                "label": label2id[row["label"]],
                "images": [resolve_image(row["image"], args.image_root)],
            })
        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(records):>6} records  {path}")


if __name__ == "__main__":
    main()
