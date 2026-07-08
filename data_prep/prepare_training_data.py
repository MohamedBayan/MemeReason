"""Build the ms-swift training/evaluation files from the MemeReason dataset.

Produces, for a given task, the JSONL files consumed by the training and
inference scripts:

  classification_only/   label only                          (SFT warm-up)
  with_explanation/      label + explanation                 (SFT warm-up)
  fine_grained/          fine-grained labels [+ explanation] (SFT warm-up)
  thinking/              distilled <think> + label + expl.   (SFT CoTD)
  cot_explanation/       CoT prompt, no reasoning target     (zero-shot CoT eval)
  grpo/                  CoT prompt + gold "solution"        (GRPO)

Every record is a chat sample:
  {"messages": [system, user, assistant], "images": [path]}
GRPO records replace the assistant turn with a "solution" string used by the
reward functions.

The source data is the extended dataset released at
https://huggingface.co/datasets/QCRI/MemeReason (or local JSONL files with the
same fields). Images are not distributed with the dataset; download them from
the original benchmarks (see data/README.md) and pass --image_root.

The system prompt is shared across all stages (the files used for the paper
differed only in a trailing period between warm-up and GRPO).

Examples:
    python data_prep/prepare_training_data.py --task hateful \
        --from_hub QCRI/MemeReason --image_root data/hateful_memes/img \
        --output_dir data/formatted/hateful

    python data_prep/prepare_training_data.py --task armeme \
        --source_dir data/armeme_raw --image_root data/armeme_images \
        --output_dir data/formatted/armeme
"""

import argparse
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Message templates differ slightly between the two tasks; these mirror the
# files the paper models were trained on.
TASKS = {
    "hateful": {
        "hub_config": "hateful_memes",
        "user_template": "<image> {instruction} Text extracted: {text}",
        "cotd_user_template": "<image>\n{instruction}\n\nMeme text: {text}",
        "labels": ["hateful", "not-hateful"],
        "prompts": {
            "system": "hateful/system.txt",
            "classification": "hateful/classification.txt",
            "explanation_train": "hateful/explanation_with_guideline.txt",
            "explanation_eval": "hateful/explanation.txt",
            "fine_grained": "hateful/fine_grained.txt",
            "cot": "hateful/cot.txt",
        },
    },
    "armeme": {
        "hub_config": "armeme",
        "user_template": "<image> {instruction}\n\nText extracted: {text}",
        "cotd_user_template": "<image> {instruction}\n\nText extracted: {text}",
        "labels": ["propaganda", "not-propaganda", "not-meme", "other"],
        "prompts": {
            "system": "armeme/system.txt",
            "classification": "armeme/classification.txt",
            "explanation_train": "armeme/explanation.txt",
            "explanation_eval": "armeme/explanation.txt",
            "fine_grained": "armeme/fine_grained.txt",
            "cot": "armeme/cot.txt",
        },
    },
}

EMPTY_THINK = "<think>\n\n</think>\n\n"


def load_prompt(prompts_dir, relpath):
    return (prompts_dir / relpath).read_text(encoding="utf-8").strip()


def load_split_local(source_dir, split):
    path = Path(source_dir) / f"{split}.jsonl"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split_hub(repo, config, split):
    from datasets import Image, load_dataset

    try:
        ds = load_dataset(repo, config, split=split)
    except ValueError:
        return None
    # The armeme config embeds the images; the training files reference them
    # on disk instead (via --image_root and the id field), so skip decoding.
    if isinstance(ds.features.get("image"), Image):
        ds = ds.remove_columns(["image"])
    return [dict(row) for row in ds]


def normalize(record, task):
    """Map either the released hub schema or the raw local schema to one dict."""
    label = record.get("label") or record.get("class_label")
    explanation = record.get("explanation") or record.get("explanation_en") or ""
    techniques = record.get("techniques", record.get("fine-grained_techniques"))
    if isinstance(techniques, str) and techniques:
        techniques = json.loads(techniques)
    return {
        "image": record.get("image") or record.get("img_path") or record.get("id"),
        "text": record.get("text") or "",
        "label": label,
        "explanation": explanation,
        "protected_category": record.get("protected_category", record.get("gold_pc")),
        "attack_type": record.get("attack_type", record.get("gold_attack")),
        "techniques": techniques,
        "think": record.get("think") or "",
    }


def resolve_image(path, image_root):
    if os.path.isabs(path):
        return path
    return str(Path(image_root).resolve() / path)


def chat_record(system, user, assistant, image):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "images": [image],
    }


def think_block(raw):
    """Re-wrap the distilled reasoning trace as a normalized <think> block."""
    m = re.search(r"<think>\s*(.*?)\s*</think>", raw, flags=re.DOTALL)
    if not m or not m.group(1):
        return ""
    return f"<think>\n{m.group(1)}\n</think>"


def fine_grained_target(row, task):
    if task == "hateful":
        return (f"Protected_category: {row['protected_category']}\n"
                f"Attack_type: {row['attack_type']}\n"
                f"Label: {row['label']}")
    techniques = row["techniques"] or {}
    entries = techniques.get("techniques", []) if isinstance(techniques, dict) else []
    if not entries:
        return (f'Techniques: ["none"]\n\n'
                f"Label: {row['label']}\nExplanation: {row['explanation']}")
    names = [t.get("label", "") for t in entries if isinstance(t, dict)]
    lines = f"Techniques: {json.dumps(names)}\n"
    for t in entries:
        if isinstance(t, dict) and t.get("label") and t.get("rationale"):
            lines += f"{t['label']}: {t['rationale']}\n"
    return f"{lines}\nLabel: {row['label']}\nExplanation: {row['explanation']}"


def save(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records):>6} records  {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from_hub", metavar="REPO",
                        help="Hugging Face dataset repo, e.g. QCRI/MemeReason")
    source.add_argument("--source_dir", help="directory with local {train,dev,test}.jsonl")
    parser.add_argument("--image_root", required=True,
                        help="directory containing the meme images")
    parser.add_argument("--prompts_dir", default=str(REPO_ROOT / "prompts"))
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    cfg = TASKS[args.task]
    prompts_dir = Path(args.prompts_dir)
    out = Path(args.output_dir)

    system = load_prompt(prompts_dir, cfg["prompts"]["system"])
    instructions = {k: load_prompt(prompts_dir, v)
                    for k, v in cfg["prompts"].items() if k != "system"}

    for split in ("train", "dev", "test"):
        if args.from_hub:
            rows = load_split_hub(args.from_hub, cfg["hub_config"], split)
        else:
            rows = load_split_local(args.source_dir, split)
        if rows is None:
            print(f"split {split!r} not found, skipping")
            continue
        rows = [normalize(r, args.task) for r in rows]
        is_test = split == "test"

        def user(instruction_key, row, template=cfg["user_template"]):
            return template.format(instruction=instructions[instruction_key], text=row["text"])

        # 1. classification only
        save([chat_record(system, user("classification", r),
                          f"{EMPTY_THINK}Label: {r['label']}",
                          resolve_image(r["image"], args.image_root))
              for r in rows],
             out / "classification_only" / f"{split}.jsonl")

        # 2. label + explanation (guideline prompt for training, plain for eval;
        #    the eval reference carries no placeholder think block)
        expl_key = "explanation_eval" if is_test else "explanation_train"
        expl_prefix = "" if is_test else EMPTY_THINK
        save([chat_record(system, user(expl_key, r),
                          f"{expl_prefix}Label: {r['label']}\nExplanation: {r['explanation']}",
                          resolve_image(r["image"], args.image_root))
              for r in rows],
             out / "with_explanation" / f"{split}.jsonl")

        # 3. fine-grained supervision (test keeps the plain label+explanation target)
        fg_records = []
        for r in rows:
            target = (f"Label: {r['label']}\nExplanation: {r['explanation']}" if is_test
                      else fine_grained_target(r, args.task))
            fg_records.append(chat_record(system, user("fine_grained", r),
                                          f"{EMPTY_THINK}{target}",
                                          resolve_image(r["image"], args.image_root)))
        save(fg_records, out / "fine_grained" / f"{split}.jsonl")

        # 4. distilled chain-of-thought (CoTD); test carries no reasoning trace
        cot_records = []
        for r in rows:
            image = resolve_image(r["image"], args.image_root)
            u = cfg["cotd_user_template"].format(instruction=instructions["cot"], text=r["text"])
            trace = "" if is_test else think_block(r["think"])
            if trace:
                assistant = f"{trace}\n\nLabel: {r['label']}\nExplanation: {r['explanation']}"
            else:
                assistant = f"{EMPTY_THINK}Label: {r['label']}\nExplanation: {r['explanation']}"
            cot_records.append(chat_record(system, u, assistant, image))
        save(cot_records, out / "thinking" / f"{split}.jsonl")

        # 5. CoT evaluation file for zero-shot inference (assistant = gold reference)
        if is_test:
            save([chat_record(system, user("cot", r),
                              f"Label: {r['label']}\nExplanation: {r['explanation']}",
                              resolve_image(r["image"], args.image_root))
                  for r in rows],
                 out / "cot_explanation" / f"{split}.jsonl")

        # 6. GRPO: prompt only, gold answer in "solution"
        grpo_records = []
        for r in rows:
            grpo_records.append({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user("cot", r)},
                ],
                "solution": f"Label: {r['label']}\nExplanation: {r['explanation']}",
                "images": [resolve_image(r["image"], args.image_root)],
            })
        save(grpo_records, out / "grpo" / f"{split}.jsonl")


if __name__ == "__main__":
    main()
