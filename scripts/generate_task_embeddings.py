"""
Generate CLIP text embeddings for tasks.json.

Uses openai/clip-vit-base-patch32 (512-dim output) via HuggingFace transformers.
Encodes each task's instruction (optionally prepended with embodiment description).

Usage:
    python scripts/generate_task_embeddings.py
    python scripts/generate_task_embeddings.py --input tasks.json --output tasks.json
    python scripts/generate_task_embeddings.py --use-embodiment   # prepend embodiment to instruction
    python scripts/generate_task_embeddings.py --no-normalize     # keep raw (unnormalized) embeddings
"""

import argparse
import json
import os
import torch
from transformers import CLIPTextModel, CLIPTokenizer

CLIP_MODEL = "openai/clip-vit-base-patch32"  # 512-dim text embeddings


def encode_texts(texts: list[str], normalize: bool = True) -> list[list[float]]:
    print(f"Loading {CLIP_MODEL}...")
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    model = CLIPTextModel.from_pretrained(CLIP_MODEL)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Encoding {len(texts)} texts on {device}...")

    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Use pooled output (CLS token after projection) — shape: (N, 512)
        embeddings = outputs.pooler_output

        if normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    return embeddings.cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tasks.json", help="Input tasks.json path")
    parser.add_argument("--output", default=None, help="Output path (default: overwrite input)")
    parser.add_argument("--use-embodiment", action="store_true",
                        help="Prepend embodiment description to instruction")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip L2 normalization of embeddings")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or args.input

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"tasks.json not found at: {input_path}")

    with open(input_path) as f:
        tasks = json.load(f)

    task_names = list(tasks.keys())
    texts = []
    for name in task_names:
        t = tasks[name]
        instruction = t.get("instruction", name)
        if args.use_embodiment and "embodiment" in t:
            text = t["embodiment"] + ". " + instruction
        else:
            text = instruction
        texts.append(text)
        print(f"  {name}: {text!r}")

    embeddings = encode_texts(texts, normalize=not args.no_normalize)

    for name, emb in zip(task_names, embeddings):
        tasks[name]["text_embedding"] = emb

    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"\nSaved {len(task_names)} embeddings ({len(embeddings[0])}-dim) → {output_path}")

    # Sanity check
    import numpy as np
    e = np.array(embeddings[0])
    print(f"Sample embedding norm: {np.linalg.norm(e):.4f}  (should be ~1.0 if normalized)")


if __name__ == "__main__":
    main()
