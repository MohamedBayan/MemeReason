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

1. Download [QCRI/ArMeme](https://huggingface.co/datasets/QCRI/ArMeme) — it
   contains the images.
2. Place / export the images so that the `image` field of the released dataset
   (the original ArMeme relative path) resolves against `--image_root`.

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
