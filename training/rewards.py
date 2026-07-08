"""GRPO reward functions for MemeReason.

This module is loaded by ms-swift through ``--external_plugins`` and registers
the reward functions used in the paper:

  format_check         structural compliance of <think>/Label/Explanation
  label_accuracy       exact match against the gold label (supervised)
  explanation_length   Gaussian reward centred on ~100-word explanations
  explanation_meteor   METEOR similarity to the gold explanation (supervised)
  think_length         saturated length reward on the <think> block (R_think)
  majority_vote_label  consensus pseudo-reward over rollouts (self-supervised)
  label_entropy        label-diversity bonus (optional anti-collapse term)

The expected completion format is:

  <think> ... </think>
  Label: <class>
  Explanation: <free text>

Set the environment variable MEME_TASK to "hateful" or "armeme" before
training so the label set matches the dataset. The training scripts in this
repository do this for you.
"""

import math
import os
import re
import sys
from collections import Counter

from swift.plugin import ORM, orms

LABEL_SETS = {
    "hateful": {
        "hateful": "hateful",
        "hate": "hateful",
        "not hateful": "nothateful",
        "not-hateful": "nothateful",
        "nothateful": "nothateful",
        "non hateful": "nothateful",
        "non-hateful": "nothateful",
        "benign": "nothateful",
        "not hate": "nothateful",
    },
    "armeme": {
        "propaganda": "propaganda",
        "propganda": "propaganda",
        "not propaganda": "notpropaganda",
        "not-propaganda": "notpropaganda",
        "non propaganda": "notpropaganda",
        "non-propaganda": "notpropaganda",
        "notpropaganda": "notpropaganda",
        "not meme": "notmeme",
        "not-meme": "notmeme",
        "non meme": "notmeme",
        "non-meme": "notmeme",
        "notmeme": "notmeme",
        "other": "other",
    },
}

_task = os.environ.get("MEME_TASK")
if _task not in LABEL_SETS:
    raise RuntimeError(
        f"MEME_TASK must be one of {sorted(LABEL_SETS)}, got {_task!r}. "
        "Export it before launching swift rlhf, e.g. MEME_TASK=hateful."
    )

LABEL_CANON = LABEL_SETS[_task]
VALID_LABELS = set(LABEL_CANON.values())

DEBUG = os.environ.get("MAJORITY_VOTE_DEBUG", "0") == "1"


def first_think_block(text):
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else "").strip()


def first_label_line(text):
    m = re.search(r"^\s*Label\s*:\s*([^\n\r]*)", text, flags=re.IGNORECASE | re.MULTILINE)
    return m.group(1) if m else ""


def first_explanation_block(text):
    m = re.search(r"^\s*Explanation\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    return (m.group(1) if m else "").strip()


def canonicalize_label(raw):
    x = raw.strip().lower()
    x = re.sub(r"\s+", " ", x.replace("_", " ").replace("-", " "))
    return LABEL_CANON.get(x, x.replace(" ", ""))


def count_words(s):
    return len(re.findall(r"\b\w+\b", s))


def plateau_gaussian(length, center, tol, sigma):
    """1.0 on [center-tol, center+tol], Gaussian decay outside."""
    if center - tol <= length <= center + tol:
        return 1.0
    return math.exp(-((length - center) ** 2) / (2.0 * sigma ** 2))


def saturated_length(length, lo, hi, sigma):
    """1.0 on [lo, hi], Gaussian decay below lo and above hi."""
    if lo <= length <= hi:
        return 1.0
    edge = lo if length < lo else hi
    return math.exp(-((length - edge) ** 2) / (2.0 * sigma ** 2))


class FormatCheckReward(ORM):
    """Additive structural reward, clamped to [0, 1].

    +0.45 non-empty <think> block, +0.35 valid label, +0.20 non-empty
    explanation. Keeping the components additive lets partially formatted
    completions receive gradient signal instead of a flat zero.
    """

    def __call__(self, completions, **kwargs):
        rewards = []
        for c in completions:
            score = 0.0
            if first_think_block(c):
                score += 0.45
            if canonicalize_label(first_label_line(c)) in VALID_LABELS:
                score += 0.35
            if first_explanation_block(c):
                score += 0.20
            rewards.append(min(1.0, score))
        return rewards


class LabelAccuracyReward(ORM):
    """1.0 iff the first predicted Label line matches the gold label."""

    def __call__(self, completions, solution, **kwargs):
        rewards = []
        for c, s in zip(completions, solution):
            pred = canonicalize_label(first_label_line(c))
            gold = canonicalize_label(first_label_line(s))
            rewards.append(1.0 if pred and pred == gold and pred in VALID_LABELS else 0.0)
        return rewards


class ExplanationLengthReward(ORM):
    """Softly regularizes explanation length towards ~100 words.

    1.0 for 80-120 words, Gaussian decay (sigma=20) outside; 0.0 when no
    explanation is produced.
    """

    def __call__(self, completions, **kwargs):
        rewards = []
        for c in completions:
            expl = first_explanation_block(c)
            rewards.append(plateau_gaussian(count_words(expl), 100, 20, 20) if expl else 0.0)
        return rewards


class ExplanationMETEORReward(ORM):
    """METEOR similarity between the predicted and gold explanations."""

    MAX_WORDS = 1024

    def __call__(self, completions, solution, **kwargs):
        from nltk.tokenize import word_tokenize
        from nltk.translate.meteor_score import meteor_score

        _ensure_nltk_data()
        rewards = []
        for c, s in zip(completions, solution):
            pred = " ".join(first_explanation_block(c).split()[: self.MAX_WORDS])
            ref = " ".join(first_explanation_block(s).split()[: self.MAX_WORDS])
            if not pred or not ref:
                rewards.append(0.0)
                continue
            score = meteor_score([word_tokenize(ref.lower())], word_tokenize(pred.lower()))
            rewards.append(min(1.0, max(0.0, float(score))))
        return rewards


class ThinkLengthReward(ORM):
    """R_think: penalizes reasoning traces shorter than 150 words.

    1.0 for 150-400 words, Gaussian decay (sigma=50) outside the plateau,
    0.0 for an empty <think> block. Discourages the reward hacking pattern
    where GRPO collapses the reasoning trace to nothing (see paper, Sec. 6.4).
    """

    def __call__(self, completions, **kwargs):
        rewards = []
        for c in completions:
            think = first_think_block(c)
            rewards.append(saturated_length(len(think.split()), 150, 400, 50) if think else 0.0)
        return rewards


class MajorityVoteLabelReward(ORM):
    """Consensus pseudo-reward for self-supervised GRPO (no gold labels).

    Labels are extracted from the completions in the local batch (rollouts of
    the same prompt). If a supermajority (>= 75% of valid votes) agrees on one
    label, completions predicting it get 1.0 and the rest 0.0; otherwise every
    completion gets 0.0 so ties contribute no gradient signal.

    Note: ms-swift calls reward functions per GPU with the local slice of the
    rollout group, so with num_generations=16 on 4 GPUs the vote runs over 4
    completions. Set MAJORITY_VOTE_DEBUG=1 to log votes to stderr.
    """

    SUPERMAJORITY = 0.75

    def __call__(self, completions, **kwargs):
        labels = []
        for c in completions:
            canon = canonicalize_label(first_label_line(c))
            labels.append(canon if canon in VALID_LABELS else None)

        valid = [l for l in labels if l is not None]
        if not valid:
            return [0.0] * len(completions)

        majority_label, majority_count = Counter(valid).most_common(1)[0]
        if DEBUG:
            print(f"[majority_vote] votes={labels} majority={majority_label} "
                  f"({majority_count}/{len(valid)})", file=sys.stderr, flush=True)

        if majority_count < len(valid) * self.SUPERMAJORITY:
            return [0.0] * len(completions)
        return [1.0 if l == majority_label else 0.0 for l in labels]


class LabelEntropyReward(ORM):
    """Normalized entropy of the predicted label distribution.

    Broadcast to every completion in the group. Optional anti-collapse term
    for self-supervised training on binary tasks, where near-unanimous groups
    otherwise make the consensus reward self-reinforcing.
    """

    def __call__(self, completions, **kwargs):
        labels = [canonicalize_label(first_label_line(c)) for c in completions]
        valid = [l for l in labels if l in VALID_LABELS]
        if len(valid) < 2:
            return [0.0] * len(completions)
        counts = Counter(valid)
        total = len(valid)
        entropy = -sum((n / total) * math.log(n / total) for n in counts.values())
        entropy /= math.log(len(VALID_LABELS))
        return [min(1.0, entropy)] * len(completions)


def _ensure_nltk_data():
    import nltk

    for resource, path in [("wordnet", "corpora/wordnet.zip"), ("punkt", "tokenizers/punkt")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


orms["format_check"] = FormatCheckReward
orms["label_accuracy"] = LabelAccuracyReward
orms["explanation_length"] = ExplanationLengthReward
orms["explanation_meteor"] = ExplanationMETEORReward
orms["think_length"] = ThinkLengthReward
orms["majority_vote_label"] = MajorityVoteLabelReward
orms["label_entropy"] = LabelEntropyReward
