"""Build the swift-infer input for the open-weight fine-grained annotator.

The ArMeme fine-grained annotation pipeline uses two independent annotators
(paper Sec. 3.4): GPT-4.1 through annotation/azure_batch.py, and
Llama-4-Scout locally through swift infer. This script renders the annotator
prompt for the local model:

  {"messages": [system, user(<image> + filled prompt), assistant("")],
   "images": [path]}

Example:
    python annotation/prepare_annotator_input.py \
        --dataset data/armeme_raw/train.jsonl \
        --image_root data/armeme_images \
        --output data/annotation/armeme_annotator_input/train.jsonl
"""

import argparse
import json
import os
from pathlib import Path

from azure_batch import load_jsonl, prompt_fields

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--system",
                        default=str(REPO_ROOT / "prompts/fine_grained_annotation/system.txt"))
    parser.add_argument("--prompt",
                        default=str(REPO_ROOT / "prompts/fine_grained_annotation/annotator.txt"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    system = Path(args.system).read_text(encoding="utf-8").strip()
    template = Path(args.prompt).read_text(encoding="utf-8").strip()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for item in load_jsonl(args.dataset):
            image = item.get("img_path", "")
            if args.image_root and not os.path.isabs(image):
                image = os.path.join(args.image_root, image)
            record = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",
                     "content": f"<image>\n\n{template.format_map(prompt_fields(item))}"},
                    {"role": "assistant", "content": ""},
                ],
                "images": [image],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} records to {output}")


if __name__ == "__main__":
    main()
