#!/usr/bin/env python3
"""
Paired bootstrap test on macro-F1 for two model prediction files (JSONL).

Compares model A (e.g. GRPO) vs. model B (e.g. SFT) on the SAME test set.
Both files must contain the same items in the same order; the gold label
is taken from `labels` field of file A.

Reports:
  - macro-F1 of A and B (point estimates)
  - paired bootstrap delta = F1(A) - F1(B), 95% CI, one-sided p-value
    (H1: A > B)
  - McNemar's exact test on per-instance correctness
  - per-class F1 for context
  - count of invalid (unparseable) predictions per model

Designed to be fast: parses labels with simple string ops (no regex),
and runs a fully vectorised bootstrap over class-wise TP/FP/FN counts.

Usage
-----
    python paired_bootstrap_macro_f1.py \
        --pred_a path/to/grpo.jsonl \
        --pred_b path/to/sft.jsonl  \
        --name_a "GRPO" --name_b "SFT" \
        --dataset "ArMeme_4class" \
        --out_dir scores/sig/armeme \
        --n_resamples 10000 --seed 42

Output
------
    {out_dir}/significance.json   # full results
    {out_dir}/significance.md     # human-readable report
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Fast label parsing (no regex).
# --------------------------------------------------------------------------- #
_BULLET_CHARS = "*-•·●▪▫◦◘◙○◌\t "
_INVALID = "__INVALID__"


def extract_label(text) -> str | None:
    """Return the lower-cased, normalised label, or None if no `Label:` found.

    Looks at the LAST `Label:` occurrence (model output may discuss the word
    label in its CoT before the final answer). Cuts at the next newline.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)

    idx = text.rfind("Label:")
    if idx == -1:
        return None
    rest = text[idx + len("Label:"):]
    nl = rest.find("\n")
    chunk = rest if nl == -1 else rest[:nl]

    label = chunk.strip().lower()
    label = label.replace("_", "-")
    label = label.strip(_BULLET_CHARS)
    if not label:
        return None
    if label.startswith("non-"):
        label = "not-" + label[4:]
    return label


def load_predictions(path: Path) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# Vectorised bootstrap on macro-F1.
# --------------------------------------------------------------------------- #
def _macro_f1_batch(gold_2d: np.ndarray,
                    pred_2d: np.ndarray,
                    num_classes: int) -> np.ndarray:
    """gold_2d, pred_2d: (B, n) int arrays of class indices in [0, num_classes).

    Predictions equal to `num_classes` are treated as a sentinel "invalid"
    class that never matches any gold class -> always counted wrong.
    Returns macro-F1 per row, shape (B,).
    """
    B = gold_2d.shape[0]
    f1s = np.zeros((B, num_classes), dtype=np.float64)
    for c in range(num_classes):
        gc = gold_2d == c
        pc = pred_2d == c
        tp = np.logical_and(gc, pc).sum(axis=1)
        fp = np.logical_and(~gc, pc).sum(axis=1)
        fn = np.logical_and(gc, ~pc).sum(axis=1)
        denom = 2 * tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_c = np.where(denom > 0, 2 * tp / denom, 0.0)
        f1s[:, c] = f1_c
    return f1s.mean(axis=1)


def macro_f1(gold: np.ndarray, pred: np.ndarray, num_classes: int) -> float:
    return float(
        _macro_f1_batch(gold[None, :], pred[None, :], num_classes)[0]
    )


def per_class_f1(gold: np.ndarray, pred: np.ndarray,
                 num_classes: int) -> np.ndarray:
    out = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        gc = gold == c
        pc = pred == c
        tp = int(np.logical_and(gc, pc).sum())
        fp = int(np.logical_and(~gc, pc).sum())
        fn = int(np.logical_and(gc, ~pc).sum())
        denom = 2 * tp + fp + fn
        out[c] = (2 * tp / denom) if denom > 0 else 0.0
    return out


def paired_bootstrap(
    gold: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    num_classes: int,
    n_resamples: int = 10_000,
    seed: int = 42,
    chunk: int = 1000,
) -> dict:
    """Paired bootstrap over identical test set.

    p-value is one-sided: P(F1_A* - F1_B* <= 0) under resampling, i.e. the
    fraction of resamples where B does at least as well as A. Smaller is
    stronger evidence that A > B.
    """
    n = gold.shape[0]
    rng = np.random.default_rng(seed)

    f1_a = macro_f1(gold, pred_a, num_classes)
    f1_b = macro_f1(gold, pred_b, num_classes)
    delta_obs = f1_a - f1_b

    deltas = np.empty(n_resamples, dtype=np.float64)
    done = 0
    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        idx = rng.integers(0, n, size=(m, n))
        g = gold[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]
        deltas[done:done + m] = (
            _macro_f1_batch(g, pa, num_classes)
            - _macro_f1_batch(g, pb, num_classes)
        )
        done += m

    p_one_sided = float((deltas <= 0).mean())
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return {
        "f1_a": float(f1_a),
        "f1_b": float(f1_b),
        "delta": float(delta_obs),
        "p_value_one_sided": p_one_sided,
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "n_resamples": int(n_resamples),
        "n_test": int(n),
    }


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar's test on paired binary correctness.

    Returns one-sided p-value for H1: A is more often correct than B,
    plus the discordant counts.
    """
    only_a = int(np.logical_and(correct_a, ~correct_b).sum())
    only_b = int(np.logical_and(~correct_a, correct_b).sum())
    both = int(np.logical_and(correct_a, correct_b).sum())
    neither = int(np.logical_and(~correct_a, ~correct_b).sum())
    n_disc = only_a + only_b
    if n_disc == 0:
        p_one = 1.0
        p_two = 1.0
    else:
        # Under H0, only_a ~ Binomial(n_disc, 0.5)
        # One-sided H1: A more often correct -> P(X >= only_a)
        p_one = float(stats.binom.sf(only_a - 1, n_disc, 0.5))
        # Two-sided
        p_two = float(stats.binomtest(only_a, n_disc, 0.5,
                                      alternative="two-sided").pvalue)
    return {
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "both_correct": both,
        "neither_correct": neither,
        "p_value_one_sided": p_one,
        "p_value_two_sided": p_two,
    }


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
@dataclass
class CompareResult:
    dataset: str
    name_a: str
    name_b: str
    path_a: str
    path_b: str
    n_test: int
    classes: list[str]
    invalid_a: int
    invalid_b: int
    accuracy_a: float
    accuracy_b: float
    macro_f1_a: float
    macro_f1_b: float
    per_class_f1_a: dict
    per_class_f1_b: dict
    bootstrap: dict
    mcnemar: dict


def _build_class_index(gold_labels: list[str | None],
                       pred_labels: Iterable[Iterable[str | None]]
                       ) -> tuple[dict, list]:
    """Class index excludes None / unparseable; gold must be parseable."""
    classes = sorted({g for g in gold_labels if g is not None})
    if not classes:
        raise ValueError("No parseable gold labels found.")
    return {c: i for i, c in enumerate(classes)}, classes


def _to_int_arrays(gold_labels, pred_labels, cls_to_idx, num_classes):
    n = len(gold_labels)
    gold = np.empty(n, dtype=np.int32)
    pred = np.empty(n, dtype=np.int32)
    invalid = 0
    for i, (g, p) in enumerate(zip(gold_labels, pred_labels)):
        if g is None or g not in cls_to_idx:
            raise ValueError(f"Gold label missing/unknown at row {i}: {g!r}")
        gold[i] = cls_to_idx[g]
        if p is None or p not in cls_to_idx:
            pred[i] = num_classes  # sentinel "invalid"
            invalid += 1
        else:
            pred[i] = cls_to_idx[p]
    return gold, pred, invalid


def compare(path_a: Path, path_b: Path,
            name_a: str, name_b: str,
            dataset: str,
            n_resamples: int, seed: int) -> CompareResult:
    rows_a = load_predictions(path_a)
    rows_b = load_predictions(path_b)
    if len(rows_a) != len(rows_b):
        raise ValueError(
            f"Length mismatch: {len(rows_a)} (A) vs {len(rows_b)} (B). "
            "Both files must be the same test set in the same order."
        )

    gold_a = [extract_label(r.get("labels")) for r in rows_a]
    gold_b = [extract_label(r.get("labels")) for r in rows_b]
    pred_a = [extract_label(r.get("response")) for r in rows_a]
    pred_b = [extract_label(r.get("response")) for r in rows_b]

    if gold_a != gold_b:
        n_diff = sum(1 for x, y in zip(gold_a, gold_b) if x != y)
        raise ValueError(
            f"Gold labels disagree between files at {n_diff} positions. "
            "Aborting — these files are not the same test set / order."
        )

    cls_to_idx, classes = _build_class_index(gold_a, [pred_a, pred_b])
    num_classes = len(classes)
    gold_arr, pa_arr, inv_a = _to_int_arrays(gold_a, pred_a, cls_to_idx, num_classes)
    _,        pb_arr, inv_b = _to_int_arrays(gold_a, pred_b, cls_to_idx, num_classes)

    correct_a = (pa_arr == gold_arr)
    correct_b = (pb_arr == gold_arr)
    acc_a = float(correct_a.mean())
    acc_b = float(correct_b.mean())

    pcf_a = per_class_f1(gold_arr, pa_arr, num_classes)
    pcf_b = per_class_f1(gold_arr, pb_arr, num_classes)

    bs = paired_bootstrap(gold_arr, pa_arr, pb_arr,
                          num_classes=num_classes,
                          n_resamples=n_resamples, seed=seed)
    mc = mcnemar_exact(correct_a, correct_b)

    return CompareResult(
        dataset=dataset,
        name_a=name_a,
        name_b=name_b,
        path_a=str(path_a),
        path_b=str(path_b),
        n_test=len(gold_arr),
        classes=classes,
        invalid_a=inv_a,
        invalid_b=inv_b,
        accuracy_a=acc_a,
        accuracy_b=acc_b,
        macro_f1_a=bs["f1_a"],
        macro_f1_b=bs["f1_b"],
        per_class_f1_a={c: float(pcf_a[i]) for i, c in enumerate(classes)},
        per_class_f1_b={c: float(pcf_b[i]) for i, c in enumerate(classes)},
        bootstrap=bs,
        mcnemar=mc,
    )


def render_markdown(r: CompareResult) -> str:
    bs = r.bootstrap
    mc = r.mcnemar
    lines = []
    lines.append(f"# Paired bootstrap macro-F1: {r.dataset}")
    lines.append("")
    lines.append(f"- A = **{r.name_a}**  `{Path(r.path_a).name}`")
    lines.append(f"- B = **{r.name_b}**  `{Path(r.path_b).name}`")
    lines.append(f"- n = {r.n_test} examples, {len(r.classes)} classes "
                 f"({', '.join(r.classes)})")
    lines.append(f"- invalid predictions: A={r.invalid_a}, B={r.invalid_b} "
                 f"(treated as wrong)")
    lines.append("")
    lines.append("## Point estimates")
    lines.append("")
    lines.append("| metric | A ({}) | B ({}) | Δ (A − B) |".format(r.name_a, r.name_b))
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| accuracy | {r.accuracy_a*100:.2f} | {r.accuracy_b*100:.2f} "
                 f"| {(r.accuracy_a-r.accuracy_b)*100:+.2f} |")
    lines.append(f"| macro-F1 | {r.macro_f1_a*100:.2f} | {r.macro_f1_b*100:.2f} "
                 f"| {bs['delta']*100:+.2f} |")
    lines.append("")
    lines.append("Per-class F1:")
    lines.append("")
    lines.append("| class | A | B | Δ |")
    lines.append("|---|---:|---:|---:|")
    for c in r.classes:
        a = r.per_class_f1_a[c]; b = r.per_class_f1_b[c]
        lines.append(f"| {c} | {a*100:.2f} | {b*100:.2f} | {(a-b)*100:+.2f} |")
    lines.append("")
    lines.append("## Paired bootstrap on macro-F1")
    lines.append("")
    lines.append(f"- B = {bs['n_resamples']:,} resamples, paired over the same "
                 f"test indices.")
    lines.append(f"- 95% bootstrap CI on Δ macro-F1: "
                 f"[{bs['ci_95_low']*100:+.2f}, {bs['ci_95_high']*100:+.2f}] (pp)")
    lines.append(f"- One-sided p-value (H1: A > B): "
                 f"**p = {bs['p_value_one_sided']:.4f}**")
    lines.append("")
    lines.append("## McNemar's exact test (per-instance correctness)")
    lines.append("")
    lines.append(f"- only A correct: {mc['only_a_correct']}, "
                 f"only B correct: {mc['only_b_correct']}, "
                 f"both: {mc['both_correct']}, "
                 f"neither: {mc['neither_correct']}")
    lines.append(f"- One-sided p-value (H1: A more often correct): "
                 f"**p = {mc['p_value_one_sided']:.4g}**")
    lines.append(f"- Two-sided p-value: p = {mc['p_value_two_sided']:.4g}")
    return "\n".join(lines) + "\n"


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_a", required=True, help="JSONL for model A (the proposed / better-hypothesised one, e.g. GRPO)")
    p.add_argument("--pred_b", required=True, help="JSONL for model B (the baseline, e.g. SFT)")
    p.add_argument("--name_a", default="A")
    p.add_argument("--name_b", default="B")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_resamples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = cli()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = compare(
        path_a=Path(args.pred_a),
        path_b=Path(args.pred_b),
        name_a=args.name_a,
        name_b=args.name_b,
        dataset=args.dataset,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )

    md = render_markdown(res)
    (out_dir / "significance.md").write_text(md, encoding="utf-8")
    with open(out_dir / "significance.json", "w", encoding="utf-8") as f:
        json.dump(asdict(res), f, indent=2, ensure_ascii=False)

    print(md)
    print(f"[saved] {out_dir / 'significance.md'}")
    print(f"[saved] {out_dir / 'significance.json'}")


if __name__ == "__main__":
    main()
