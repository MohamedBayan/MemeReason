# LLM data-construction pipelines

These scripts produced the dataset extensions released with the paper. They
are included for transparency and reuse; to reproduce the paper's training
runs you do **not** need to rerun them — the outputs are already in
[QCRI/MemeReason](https://huggingface.co/datasets/QCRI/MemeReason).

Credentials: the Azure OpenAI steps read `AZURE_API_KEY`, `AZURE_API_URL`,
`AZURE_API_VERSION` (or a `.env` file via `--env`); the Gemini consolidation
reads `VERTEX_PROJECT` / `VERTEX_LOCATION` with application-default gcloud
credentials.

## Chain-of-thought distillation (paper Sec. 3.5)

GPT-4.1 generates a private step-by-step reasoning trace conditioned on the
meme image, extracted text, annotation guidelines, and the gold label and
explanation. The prompt explicitly forbids copying the reference explanation
or naming the label inside the trace.

```bash
python annotation/azure_batch.py build \
    --dataset data/hateful_raw/train.jsonl \
    --prompt prompts/hateful/think_generation.txt \
    --workdir runs/cot_hateful
python annotation/azure_batch.py submit   --workdir runs/cot_hateful
python annotation/azure_batch.py status   --workdir runs/cot_hateful
python annotation/azure_batch.py retrieve --workdir runs/cot_hateful
python annotation/azure_batch.py merge \
    --dataset data/hateful_raw/train.jsonl --workdir runs/cot_hateful \
    --field think --output data/hateful_raw/train_with_think.jsonl
```

Same flow for ArMeme with `prompts/armeme/think_generation.txt`.

## Fine-grained propaganda annotation (paper Sec. 3.4)

Two independent annotators over the 23 propaganda techniques defined in
`prompts/fine_grained_annotation/system.txt`, then consolidation:

```bash
# annotator A: GPT-4.1 (batch API)
python annotation/azure_batch.py build \
    --dataset data/armeme_raw/train.jsonl \
    --system prompts/fine_grained_annotation/system.txt \
    --prompt prompts/fine_grained_annotation/annotator.txt \
    --workdir runs/fg_gpt41
# ... submit / retrieve / merge --field fine_grained_labels_gpt41

# annotator B: Llama-4-Scout (local, swift infer)
python annotation/prepare_annotator_input.py \
    --dataset data/armeme_raw/train.jsonl \
    --output data/annotation/annotator_input/train.jsonl
annotation/annotate_techniques_llama.sh \
    data/annotation/annotator_input/train.jsonl \
    results/annotation/llama4_scout_train.jsonl
# join the responses back as fine_grained_labels_llama4

# consolidation: Gemini adjudicator
python annotation/consolidate_annotations.py \
    --dataset data/annotation/armeme_with_both_annotations.jsonl \
    --image_root data/armeme \
    --output data/annotation/armeme_consolidated.jsonl
```

Reliability: on ~150 doubly human-annotated memes, human-LLM agreement was
Gwet's AC1 = 0.77 (paper Sec. 3.4).

## LLM-as-judge evaluation of distilled CoT (paper Sec. 3.5)

`prompts/llm_judge/` contains the prompts used to score the distilled traces
(faithfulness, plausibility, clarity, informativeness on a 1-5 Likert scale)
with InternVL3.5-8B and Phi-3.5-Vision via swift infer.
