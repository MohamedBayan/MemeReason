"""Disagreement-based sampling of unlabeled memes for self-supervised GRPO.

Implements the hard-sample selection of paper Sec. 4.5: the unlabeled pool is
run through the model at five temperatures (see inference/multi_temperature.sh),
labels are extracted from each run, and samples are kept according to how many
of the five predictions agree:

  3_4  partial agreement (3/5 or 4/5) - genuine model uncertainty; this is the
       strategy used for the paper's self-supervised experiments
  5_5  full agreement - high-confidence pseudo-labels, kept for comparison

The selected samples are written in GRPO format with the majority-vote label
as "solution" (the majority-vote reward recomputes consensus online during
training; the stored label is informational).

Example:
    python data_prep/build_unlabeled_training_set.py --task hateful \
        --predictions_dir results/multi_temperature/hateful \
        --pool data/formatted/hateful/unlabeled/pool.jsonl \
        --strategy 3_4 --num_samples 2000 \
        --output data/formatted/hateful/unlabeled/train_2000.jsonl
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

VALID_LABELS = {
    "hateful": ["not-hateful", "hateful"],
    "armeme": ["not-propaganda", "not-meme", "propaganda", "other"],
}


def extract_label(response, valid_labels):
    """Take the last 'Label:' line of a response, mapped onto the label set."""
    idx = response.lower().rfind("label:")
    if idx == -1:
        return None
    line = response[idx + len("label:"):].split("\n", 1)[0].strip().lower()
    if not line:
        return None
    if line in valid_labels:
        return line
    for label in valid_labels:  # ordered so that "not-*" is matched first
        if line.startswith(label):
            return label
    return None


def stratified_sample(indices_by_class, n_total, rng):
    """Proportional class-stratified sampling of up to n_total indices."""
    available = sum(len(v) for v in indices_by_class.values())
    if available == 0:
        return []
    classes = sorted(indices_by_class)
    allocations = {c: min(int(len(indices_by_class[c]) / available * n_total),
                          len(indices_by_class[c])) for c in classes}
    remaining = n_total - sum(allocations.values())
    for _, c in sorted(((len(indices_by_class[c]) - allocations[c], c) for c in classes),
                       reverse=True):
        if remaining <= 0:
            break
        add = min(remaining, len(indices_by_class[c]) - allocations[c])
        allocations[c] += add
        remaining -= add
    selected = []
    for c in classes:
        pool = sorted(indices_by_class[c])
        rng.shuffle(pool)
        selected.extend(pool[:allocations[c]])
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=sorted(VALID_LABELS))
    parser.add_argument("--predictions_dir", required=True,
                        help="directory with one JSONL per temperature run")
    parser.add_argument("--pool", required=True,
                        help="unlabeled pool in GRPO format, aligned with the prediction files")
    parser.add_argument("--strategy", default="3_4", choices=["3_4", "5_5"])
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    valid_labels = VALID_LABELS[args.task]

    prediction_files = sorted(Path(args.predictions_dir).glob("*.jsonl"))
    if len(prediction_files) < 2:
        raise SystemExit(f"expected multiple temperature runs in {args.predictions_dir}, "
                         f"found {len(prediction_files)}")
    runs = []
    for path in prediction_files:
        with open(path, encoding="utf-8") as f:
            runs.append([json.loads(line) for line in f if line.strip()])
        print(f"loaded {len(runs[-1]):>6} predictions  {path.name}")
    n_runs = len(runs)

    with open(args.pool, encoding="utf-8") as f:
        pool = [json.loads(line) for line in f if line.strip()]
    if any(len(r) != len(pool) for r in runs):
        raise SystemExit("prediction files and pool are not aligned")

    # Majority vote per sample across the temperature runs.
    samples = []
    for i in range(len(pool)):
        votes, responses = [], []
        for run in runs:
            response = run[i]["response"]
            votes.append(extract_label(response, valid_labels))
            responses.append(response)
        valid_votes = [v for v in votes if v is not None]
        if not valid_votes:
            continue
        majority, count = Counter(valid_votes).most_common(1)[0]
        majority_response = next(r for v, r in zip(votes, responses) if v == majority)
        samples.append({"idx": i, "agreement": count, "label": majority,
                        "response": majority_response})

    agreement_hist = Counter(s["agreement"] for s in samples)
    print("agreement distribution:",
          {f"{k}/{n_runs}": agreement_hist[k] for k in sorted(agreement_hist, reverse=True)})

    wanted = {"3_4": (3, 4), "5_5": (n_runs,)}[args.strategy]
    by_class = defaultdict(list)
    index = {s["idx"]: s for s in samples}
    for s in samples:
        if s["agreement"] in wanted:
            by_class[s["label"]].append(s["idx"])
    selected = stratified_sample(by_class, args.num_samples, rng)

    # Fill from the full-agreement pool if the disagreement pool is too small.
    if args.strategy == "3_4" and len(selected) < args.num_samples:
        backfill = defaultdict(list)
        for s in samples:
            if s["agreement"] == n_runs:
                backfill[s["label"]].append(s["idx"])
        extra = stratified_sample(backfill, args.num_samples - len(selected), rng)
        print(f"filling {len(extra)} samples from the {n_runs}/{n_runs} pool")
        selected.extend(extra)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for i in selected:
            s = index[i]
            record = {
                "messages": [m for m in pool[i]["messages"] if m["role"] != "assistant"]
                            + [{"role": "assistant", "content": s["response"]}],
                "solution": f"Label: {s['label']}",
                "images": pool[i]["images"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    label_hist = Counter(index[i]["label"] for i in selected)
    print(f"wrote {len(selected)} samples to {output}")
    print("class distribution:", dict(sorted(label_hist.items())))


if __name__ == "__main__":
    main()
