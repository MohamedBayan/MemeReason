"""Azure OpenAI batch pipeline for LLM-based data construction.

One tool for every GPT-4.1 batch job in the paper:
  - chain-of-thought distillation (prompts/*/think_generation.txt)
  - fine-grained propaganda annotation (prompts/fine_grained_annotation/)
  - zero-shot GPT-4.1 inference on the test sets

Subcommands mirror the batch API workflow:

  build     render one request per dataset row (image + filled prompt
            template) into <=180MB JSONL batch files
  submit    upload the batch files and start the jobs
  status    print the status of every submitted job
  retrieve  download the outputs of completed jobs
  merge     join the responses back onto the dataset by id, storing the
            response text under --field (e.g. "think")

Credentials are read from the environment (or a .env file passed with --env):
  AZURE_API_KEY, AZURE_API_URL, AZURE_API_VERSION

Example (CoT distillation for the hateful memes training set):
  python annotation/azure_batch.py build \
      --dataset data/hateful_raw/train.jsonl \
      --prompt prompts/hateful/think_generation.txt \
      --workdir runs/cot_hateful --deployment gpt-4.1
  python annotation/azure_batch.py submit   --workdir runs/cot_hateful
  python annotation/azure_batch.py status   --workdir runs/cot_hateful
  python annotation/azure_batch.py retrieve --workdir runs/cot_hateful
  python annotation/azure_batch.py merge \
      --dataset data/hateful_raw/train.jsonl \
      --workdir runs/cot_hateful --field think \
      --output data/hateful_raw/train_with_think.jsonl
"""

import argparse
import base64
import json
import os
from pathlib import Path

BATCH_FILE_LIMIT = 180 * 1024 * 1024
IMAGE_SIZE_LIMIT = 10 * 1024 * 1024


class PromptFields(dict):
    """Missing placeholders render as empty strings instead of raising."""

    def __missing__(self, key):
        return ""


def prompt_fields(item):
    """Aliases so one prompt template vocabulary covers all datasets."""
    fields = PromptFields(item)
    fields.setdefault("label", item.get("class_label", ""))
    fields.setdefault("class_label", item.get("class_label", ""))
    fields.setdefault("explanation", item.get("explanation", item.get("explanation_en", "")))
    fields.setdefault("explanation_en", item.get("explanation_en", item.get("explanation", "")))
    fields.setdefault("protected_category", item.get("gold_pc", "None"))
    fields.setdefault("attack", item.get("gold_attack", "None"))
    fields.setdefault("techniques", json.dumps(item.get("fine-grained_techniques", {}),
                                               ensure_ascii=False))
    return fields


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def client_from_env(env_file=None):
    from openai import AzureOpenAI

    if env_file:
        for line in Path(env_file).read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"'))
    return AzureOpenAI(
        api_key=os.environ["AZURE_API_KEY"],
        azure_endpoint=os.environ["AZURE_API_URL"],
        api_version=os.environ["AZURE_API_VERSION"],
    )


def cmd_build(args):
    prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    system = Path(args.system).read_text(encoding="utf-8").strip() if args.system else None
    batch_dir = Path(args.workdir) / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch, size, counter, skipped = [], 0, 1, 0

    def flush():
        nonlocal batch, size, counter
        if not batch:
            return
        path = batch_dir / f"batch_{counter}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in batch:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(batch):>5} requests  {path}")
        batch, size, counter = [], 0, counter + 1

    for item in load_jsonl(args.dataset):
        image_path = item.get("img_path", "")
        if args.image_root and not os.path.isabs(image_path):
            image_path = os.path.join(args.image_root, image_path)
        if not os.path.exists(image_path) or os.path.getsize(image_path) > IMAGE_SIZE_LIMIT:
            skipped += 1
            continue
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt.format_map(prompt_fields(item))},
            ],
        })
        payload = {
            "custom_id": str(item.get("id", os.path.basename(image_path))),
            "method": "POST",
            "url": "/chat/completions",
            "body": {
                "model": args.deployment,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            },
        }
        payload_size = len(json.dumps(payload).encode())
        if size + payload_size > BATCH_FILE_LIMIT:
            flush()
        batch.append(payload)
        size += payload_size
    flush()
    if skipped:
        print(f"skipped {skipped} items (missing or oversized image)")


def tracking_file(workdir):
    return Path(workdir) / "batch_ids.csv"


def cmd_submit(args):
    client = client_from_env(args.env)
    batch_dir = Path(args.workdir) / "batches"
    with open(tracking_file(args.workdir), "a", encoding="utf-8") as track:
        for path in sorted(batch_dir.glob("*.jsonl")):
            with open(path, "rb") as f:
                uploaded = client.files.create(file=f, purpose="batch")
            job = client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/chat/completions",
                completion_window="24h",
                metadata={"description": path.name},
            )
            track.write(f"{job.id},{path}\n")
            print(f"submitted {job.id}  ({path.name})")


def iter_jobs(workdir):
    path = tracking_file(workdir)
    if not path.exists():
        raise SystemExit(f"no tracking file at {path}; run submit first")
    for line in path.read_text().splitlines():
        job_id, _, batch_path = line.partition(",")
        yield job_id, batch_path


def cmd_status(args):
    client = client_from_env(args.env)
    for job_id, batch_path in iter_jobs(args.workdir):
        job = client.batches.retrieve(job_id)
        print(f"{job_id}  {job.status:<12} {os.path.basename(batch_path)}")


def cmd_retrieve(args):
    client = client_from_env(args.env)
    results_dir = Path(args.workdir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for job_id, _ in iter_jobs(args.workdir):
        job = client.batches.retrieve(job_id)
        if job.status != "completed":
            print(f"{job_id} is {job.status}, skipping")
            continue
        output = results_dir / f"batch_output_{job_id}.jsonl"
        output.write_text(client.files.content(job.output_file_id).text, encoding="utf-8")
        print(f"retrieved {output}")


def cmd_merge(args):
    responses = {}
    for path in sorted((Path(args.workdir) / "results").glob("batch_output_*.jsonl")):
        for row in load_jsonl(path):
            try:
                content = row["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                continue
            responses[str(row["custom_id"])] = content

    merged = missing = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for item in load_jsonl(args.dataset):
            key = str(item.get("id", ""))
            if key in responses:
                item[args.field] = responses[key]
                merged += 1
            else:
                item[args.field] = None
                missing += 1
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"merged {merged} responses into {args.output}"
          + (f" ({missing} rows without a response)" if missing else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="render batch request files")
    build.add_argument("--dataset", required=True)
    build.add_argument("--prompt", required=True, help="prompt template with {placeholders}")
    build.add_argument("--system", help="optional system prompt file")
    build.add_argument("--image_root", help="prefix for relative image paths")
    build.add_argument("--workdir", required=True)
    build.add_argument("--deployment", default="gpt-4.1")
    build.add_argument("--max_tokens", type=int, default=4096)
    build.add_argument("--temperature", type=float, default=0.0)
    build.set_defaults(func=cmd_build)

    for name, func in (("submit", cmd_submit), ("status", cmd_status), ("retrieve", cmd_retrieve)):
        p = sub.add_parser(name)
        p.add_argument("--workdir", required=True)
        p.add_argument("--env", help=".env file with the Azure credentials")
        p.set_defaults(func=func)

    merge = sub.add_parser("merge", help="join responses back onto the dataset")
    merge.add_argument("--dataset", required=True)
    merge.add_argument("--workdir", required=True)
    merge.add_argument("--field", required=True, help="field name for the response text")
    merge.add_argument("--output", required=True)
    merge.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
