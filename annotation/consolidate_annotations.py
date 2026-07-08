"""Consolidate the two fine-grained annotations with a Gemini adjudicator.

Stage 2 of the ArMeme fine-grained annotation pipeline (paper Sec. 3.4):
Gemini merges the independent GPT-4.1 and Llama-4-Scout technique annotations
into one final annotation per meme - validating overlaps, resolving
disagreements, and adding missing techniques when justified.

Requires Vertex AI credentials (gcloud auth application-default login) and the
environment variables VERTEX_PROJECT and VERTEX_LOCATION.

Input rows must carry the two annotations under
``fine_grained_labels_gpt41`` and ``fine_grained_labels_llama4``; the result
is written to ``fine_grained_labels_final``.

Example:
    python annotation/consolidate_annotations.py \
        --dataset data/annotation/armeme_with_both_annotations.jsonl \
        --image_root data/armeme_images \
        --output data/annotation/armeme_consolidated.jsonl
"""

import argparse
import json
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_json_response(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(cleaned)


def as_compact_json(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {}
    return json.dumps(value or {}, ensure_ascii=False, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--system",
                        default=str(REPO_ROOT / "prompts/fine_grained_annotation/system.txt"))
    parser.add_argument("--prompt",
                        default=str(REPO_ROOT / "prompts/fine_grained_annotation/adjudicator.txt"))
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    import vertexai
    from vertexai.generative_models import GenerativeModel, Part

    vertexai.init(project=os.environ["VERTEX_PROJECT"],
                  location=os.environ.get("VERTEX_LOCATION", "us-central1"))

    system = Path(args.system).read_text(encoding="utf-8").strip()
    template = Path(args.prompt).read_text(encoding="utf-8").strip()
    model = GenerativeModel(args.model, system_instruction=[system])

    with open(args.dataset, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with open(output, "w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            image = row.get("img_path", "")
            if args.image_root and not os.path.isabs(image):
                image = os.path.join(args.image_root, image)

            prompt = template.format(
                text=row.get("text", ""),
                class_label=row.get("class_label", ""),
                explanation=row.get("explanation_en", row.get("explanation", "")),
                annotation_a=as_compact_json(row.get("fine_grained_labels_gpt41")),
                annotation_b=as_compact_json(row.get("fine_grained_labels_llama4")),
            )
            image_part = Part.from_data(Path(image).read_bytes(), mime_type="image/jpeg")

            result = None
            for attempt in range(args.max_retries):
                try:
                    response = model.generate_content(
                        [image_part, prompt],
                        generation_config={"temperature": 0.0, "max_output_tokens": 4096},
                    )
                    result = parse_json_response(response.text)
                    break
                except Exception as e:
                    print(f"row {i}: attempt {attempt + 1} failed: {e}")
                    time.sleep(2 ** attempt)

            row["fine_grained_labels_final"] = result
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            done += result is not None
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(rows)} rows processed")

    print(f"consolidated {done}/{len(rows)} rows into {output}")


if __name__ == "__main__":
    main()
