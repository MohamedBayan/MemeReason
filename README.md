# MemeReason 

Code and data for two papers on explainable meme understanding with
thinking-based multimodal LLMs:

- **"Can Thinking Models Think to Detect Hateful Memes?"**
  (WWW Companion 2026, [doi:10.1145/3774905.3795465](https://doi.org/10.1145/3774905.3795465)) —
  introduces GRPO-based post-training with task-specific rewards for
  thinking-based MLLMs on the English Hateful Memes benchmark, showing that
  reinforcement learning improves both classification and explanation quality.
- **"Adapting Reinforcement Learning with Chain-of-Thought Supervision for
  Explainable Detection of Hateful and Propagandistic Memes"**
  ([arXiv:2606.15307](https://arxiv.org/abs/2606.15307)) — the journal
  extension: cross-lingual generalization to Arabic propagandistic memes
  (ArMeme, 4-class), a multi-LLM fine-grained annotation pipeline, distilled
  chain-of-thought supervision, thinking-length regularization (R_think), and
  self-supervised GRPO with consensus pseudo-labels.

We post-train thinking-based multimodal LLMs (Qwen3-VL-8B-Thinking) to jointly
classify memes and explain their decisions, using SFT warm-up followed by GRPO
with task-specific rewards, and extend both benchmarks with distilled
chain-of-thought rationales and fine-grained annotations. On Hateful Memes the
approach reaches **82.0 accuracy / 0.80 macro-F1**; on ArMeme, self-supervised
GRPO reaches **0.612 macro-F1**, +7.6 points over prior work, while also
generating natural-language explanations.

> **Warning:** this repository and the associated datasets contain memes and
> model outputs that may be disturbing or offensive.

## Resources

| Resource | Link |
| --- | --- |
| Journal paper | [arXiv:2606.15307](https://arxiv.org/abs/2606.15307) |
| Conference paper (WWW Companion 2026) | [doi:10.1145/3774905.3795465](https://doi.org/10.1145/3774905.3795465) |
| Extended datasets (explanations, fine-grained labels, distilled CoT) | [QCRI/MemeReason](https://huggingface.co/datasets/QCRI/MemeReason) |
| Explanation-augmented benchmarks (prior work) | [QCRI/MemeXplain](https://huggingface.co/datasets/QCRI/MemeXplain) |
| Original benchmarks | [Hateful Memes](https://ai.meta.com/tools/hatefulmemes/), [QCRI/ArMeme](https://huggingface.co/datasets/QCRI/ArMeme) |

## Repository layout

```
prompts/       all prompts: task instructions, CoT distillation, fine-grained
               annotation, LLM-as-judge
data_prep/     build training/eval files from the released dataset
annotation/    LLM data-construction pipelines (CoT distillation, multi-LLM
               fine-grained annotation)
training/      reward functions + SFT / GRPO / baseline training scripts
inference/     zero-shot and fine-tuned inference (ms-swift + vLLM)
evaluation/    classification & explanation metrics, significance tests
notebooks/     dataset tour and results analysis
results/       sample model outputs (5 examples from 3 key experiments)
scores/        full test-set metrics for those experiments
slurm/         cluster job template
```

## Installation

```bash
conda create -n memereason python=3.10 -y
conda activate memereason
pip install -r requirements.txt
```

Training and inference are built on [ms-swift](https://github.com/modelscope/ms-swift)
with vLLM rollouts and DeepSpeed ZeRO-3. The paper's training runs used
4x NVIDIA H200 GPUs; inference fits on a single GPU.

## Data

The released dataset contains the text fields, labels, explanations,
fine-grained annotations, and distilled CoT traces. Meme images must be
obtained from the original benchmarks (see [data/README.md](data/README.md)),
then:

```bash
# chat-format training/eval files for both tasks
python data_prep/prepare_training_data.py --task hateful \
    --from_hub QCRI/MemeReason --image_root data/hateful_memes \
    --output_dir data/formatted/hateful
python data_prep/prepare_training_data.py --task armeme \
    --from_hub QCRI/MemeReason --image_root data/armeme \
    --output_dir data/formatted/armeme

# integer-label files for the sequence-classification baselines
python data_prep/prepare_seq_cls_data.py --task hateful \
    --from_hub QCRI/MemeReason --image_root data/hateful_memes \
    --output_dir data/formatted/hateful/seq_cls
```

## Reproducing the papers

Steps 1-3 on the `hateful` task reproduce the WWW Companion paper; the journal
extension adds the `armeme` task, the fine-grained annotation pipeline, the
R_think reward, and steps 4-5. Section numbers below refer to the journal
paper.

### 1. Zero-shot and CoT baselines (Table 8)

```bash
inference/run_all_zero_shot.sh hateful
inference/run_all_zero_shot.sh armeme
evaluation/evaluate.sh results/zero_shot/hateful scores/zero_shot/hateful '*.jsonl' --has_explanation
```

### 2. SFT warm-up (Sec. 4.3)

Three variants, each adding supervision (labels+explanations, +fine-grained
labels, +distilled CoT). The CoTD stage continues from the cls-fg-exp
checkpoint:

```bash
training/sft_warmup.sh hateful cls-exp
training/sft_warmup.sh hateful cls-fg-exp
training/sft_warmup.sh hateful cotd checkpoints/sft/hateful/cls-fg-exp/<checkpoint>
```

### 3. Supervised GRPO (Sec. 4.4)

Composite reward: format compliance, label correctness, explanation length,
METEOR similarity to the gold explanation, and the thinking-length reward
R_think that prevents reasoning collapse. Reward functions live in
[training/rewards.py](training/rewards.py); weights follow the paper
(0.35 / 0.35 / 0.08 / 0.12 / 0.10).

```bash
training/grpo.sh hateful checkpoints/sft/hateful/cotd/<checkpoint>
training/grpo.sh hateful checkpoints/sft/hateful/cotd/<checkpoint> --no-think-reward  # ablation
training/grpo.sh hateful Qwen/Qwen3-VL-8B-Thinking --no-think-reward                  # cold start
```

### 4. Self-supervised GRPO (Sec. 4.5)

Selects 2,000 unlabeled memes where multi-temperature predictions disagree,
then trains with a majority-vote consensus reward instead of gold labels:

```bash
inference/multi_temperature.sh checkpoints/grpo/hateful/<best> \
    data/formatted/hateful/unlabeled/pool.jsonl results/multi_temperature/hateful
python data_prep/build_unlabeled_training_set.py --task hateful \
    --predictions_dir results/multi_temperature/hateful \
    --pool data/formatted/hateful/unlabeled/pool.jsonl \
    --output data/formatted/hateful/unlabeled/train_2000.jsonl
training/grpo_self_supervised.sh hateful checkpoints/grpo/hateful/<best>
```

### 5. Baselines (Tables 5-7)

```bash
training/seq_cls_baseline.sh hateful Qwen/Qwen3-VL-8B-Instruct   # seq-cls heads
training/unimodal/run_all.sh hateful data/hateful_raw            # text/image-only
```

### 6. Evaluation and significance

```bash
inference/finetuned.sh checkpoints/grpo/hateful/<best>/<checkpoint> \
    data/formatted/hateful/cot_explanation/test.jsonl results/hateful/grpo_think.jsonl
python evaluation/compute_metrics.py --data results/hateful/grpo_think.jsonl \
    --out_dir scores/hateful/grpo_think --has_explanation
python evaluation/significance_test.py --pred_a results/hateful/grpo_think.jsonl \
    --pred_b results/hateful/sft_cotd.jsonl --name_a GRPO --name_b SFT \
    --dataset hateful --out_dir scores/significance/hateful
```

Use `--is_arabic` when scoring ArMeme explanations. All numbers use macro-F1
as the primary metric; significance is a Holm-corrected one-sided paired
bootstrap (B=10,000).

## Data construction pipelines

The dataset extensions were produced with the pipelines in `annotation/`
(details in [annotation/README.md](annotation/README.md)):

- **CoT distillation** - GPT-4.1 generates step-by-step reasoning conditioned
  on the meme, gold label, and guidelines (never exposed at inference).
- **Fine-grained propaganda annotation** - GPT-4.1 and Llama-4-Scout annotate
  each ArMeme meme with propaganda techniques independently; Gemini
  consolidates the two annotations. Human agreement: Gwet's AC1 = 0.77.

## Results at a glance

| Setting | FHM Acc | FHM M-F1 | ArMeme Acc | ArMeme M-F1 |
| --- | --- | --- | --- | --- |
| Best zero-shot | 77.0 | .719 | 69.7 | .350 |
| Best SFT warm-up | 79.2 | .78 | 70.3 | .54 |
| GRPO | 81.2 | .79 | 72.9 | .577 |
| GRPO + R_think | **82.0** | **.80** | 72.6 | .597 |
| + self-supervised GRPO | 81.8 | .79 | 72.8 | **.612** |

Sample predictions from the key experiments are in `results/samples/`, with
the corresponding full-test-set metrics in `scores/`.

## Citation

If you use this code or the datasets, please cite both papers — the WWW
Companion paper that introduced the method, and the journal extension that
this repository fully reproduces:

```bibtex
@inproceedings{kmainasi2026can,
  title     = {Can Thinking Models Think to Detect Hateful Memes?},
  author    = {Kmainasi, Mohamed Bayan and Kutlu, Mucahid and Ezzat Shahroor, Ali
               and Hasnat, Abul and Alam, Firoj},
  booktitle = {Companion Proceedings of the ACM Web Conference 2026},
  pages     = {935--944},
  year      = {2026}
}

@article{kmainasi2026memereason,
  title   = {Adapting Reinforcement Learning with Chain-of-Thought Supervision
             for Explainable Detection of Hateful and Propagandistic Memes},
  author  = {Kmainasi, Mohamed Bayan and Kutlu, Mucahid and Shahroor, Ali Ezzat
             and Hasnat, Abul and Alam, Firoj},
  journal = {arXiv preprint arXiv:2606.15307},
  year    = {2026}
}
```

## Acknowledgments

This work was supported by NPRP grant 14C-0916-210015 from the Qatar National
Research Fund, part of the Qatar Research Development and Innovation Council
(QRDI).
