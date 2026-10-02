"""
Minimal MLRS inference example.

    python release/mlrs/usage_example.py --model /group/earthx/accv/release/MLRS \
        --image scene.png --prompt "Segment the buildings damaged by the flood."

`--model` can be a local export or a Hub repo id. Pass several `--image` for multi-image
tasks; masks are predicted on the last one.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def overlay(image: Image.Image, masks: torch.Tensor, alpha: float = 0.5) -> Image.Image:
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    colors = np.array([[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180]], np.float32)
    for i, m in enumerate(masks.cpu().numpy()):
        rgb[m] = (1 - alpha) * rgb[m] + alpha * colors[i % len(colors)]
    return Image.fromarray(rgb.astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="mlrs_prediction.png")
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    images = [Image.open(p) for p in args.image]
    inputs = processor(images=images, text=args.prompt)
    outputs = model.generate(**inputs, max_new_tokens=1024)

    print("text:", outputs.text[0])
    print("phrases:", outputs.phrases[0])
    print("boxes:", outputs.boxes[0])

    masks = processor.post_process_masks(outputs, inputs)[0]  # [K, H, W] at the input image size
    print("masks:", tuple(masks.shape))
    if masks.shape[0]:
        overlay(images[-1], masks).save(args.output)
        print("saved", args.output)


if __name__ == "__main__":
    main()
