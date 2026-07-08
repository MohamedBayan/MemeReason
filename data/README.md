# Getting the data

The extended annotations (labels, explanations, fine-grained labels, distilled
CoT traces) are released at
[QCRI/MemeReason](https://huggingface.co/datasets/QCRI/MemeReason) and are
loaded directly by the `data_prep/` scripts via `--from_hub QCRI/MemeReason`.

The meme **images** are not redistributed; download them from the original
benchmarks and point `--image_root` at them:

## Hateful Memes (English, binary)

1. Request the dataset from the
   [Hateful Memes Challenge](https://ai.meta.com/tools/hatefulmemes/)
   (Meta license agreement required).
2. Extract the `img/` directory, e.g. to `data/hateful_memes/img/`.
3. The `image` field in the released dataset (`img/12345.png`) resolves
   against that directory.

Fine-grained labels (protected categories, attack types) originate from
[Mathias et al. (2021)](https://github.com/facebookresearch/fine_grained_hateful_memes).

## ArMeme (Arabic, 4-class)

The images are embedded in the `armeme` config of QCRI/MemeReason (and also
distributed in [QCRI/ArMeme](https://huggingface.co/datasets/QCRI/ArMeme)).
Export them to disk once so the training files can reference them by path:

```python
from pathlib import Path
from datasets import load_dataset

for split, ds in load_dataset("QCRI/MemeReason", "armeme").items():
    for row in ds:
        path = Path("data/armeme") / row["id"]
        path.parent.mkdir(parents=True, exist_ok=True)
        row["image"].save(path)
```

Then pass `--image_root data/armeme` to the data_prep scripts.

## Unlabeled pools (self-supervised GRPO)

The unlabeled pools are built from public meme datasets (MAMI, Memotion,
MET-Meme, labels discarded) for the English task, and from a new social-media
crawl following the ArMeme methodology for the Arabic task (paper Sec. 3.6).
Format them as GRPO records without a gold solution (one JSONL per pool, same
message format as `data/formatted/<task>/grpo/train.jsonl`) and pass them to
`inference/multi_temperature.sh`.

## samples/

`samples/` contains a handful of formatted records of each type so you can see
the exact training format without downloading anything.
