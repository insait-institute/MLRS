# MLRS

**More with Less: a Large Scale Remote Sensing VLM with a Simple Recipe**

Accepted to **ACCV 2026**. 📄 Paper: https://arxiv.org/abs/2607.15942 🤗 Weights: https://huggingface.co/INSAIT-Institute/MLRS

MLRS is a remote-sensing vision-language model that answers questions and performs reasoning segmentation. 
MLRS supports wide range of inputs: ultra-high-res, SAR, thermal, false-color, multi-temporal, multi-view, visual prompting, multi-image. 
A LoRA-finetuned **InternVL3.5-8B** reasons about the image (or multiple images in rgb format) and either
answers in text or calls a segmentation tool:

```
<think>...</think><segmentation>{"noun phrase": "...", "objects": [{"bbox": [x1, y1, x2, y2]}]}</segmentation>
```

Each noun phrase, its boxes (on a 0–999 grid), and the last rgb image are passed to **SAM3**, whose mask decoder was finetuned
jointly via GRTO. SAM3 returns one mask per phrase, clipped to the predicted boxes. Text-only tasks
answer with `<answer>...</answer>` and return no masks.

This repository only holds the finetuned deltas (a LoRA adapter and SAM3 mask-decoder weights).
The base models are downloaded from the Hub at the revisions pinned in `config.json`.

> `facebook/sam3` is gated. Accept its license on the Hub and log in (`hf auth login` or set `HF_TOKEN` env.variable) before loading.

## Usage

```
python usage_example.py --model INSAIT-Institute/MLRS \
        --image scene.png --prompt "Segment the buildings damaged by the flood."
```
or
```python
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

repo = "INSAIT-Institute/MLRS"
model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)

image = Image.open("scene.png")
inputs = processor(images=image, text="Segment the buildings damaged by the flood.")
out = model.generate(**inputs, max_new_tokens=1024)

out.text[0]      # full generation, including <think> and the tool call / answer
out.phrases[0]   # noun phrases sent to SAM3
out.boxes[0]     # per phrase: pixel boxes [x1, y1, x2, y2]
masks = processor.post_process_masks(out, inputs)[0]  # bool [K, H, W], one mask per phrase
```

**Multiple images** (for example, pre/post-event pairs): `processor(images=[pre, post], text=...)`.
The segmentation tool runs on the last image; use `primary_image_index=` to pick a different one.

**Batching:** `processor(images=[[img_a], [img_b1, img_b2]], text=[q_a, q_b])`.

## Inputs and outputs

| Processor output | Description |
|---|---|
| `input_ids`, `attention_mask`, `pixel_values` | InternVL3.5 chat inputs. The instruction prompt is added automatically. |
| `sam_inputs` | SAM3 image inputs for the primary image |
| `sam_original_sizes` | `(H, W)` of the primary image after downsampling (longest side ≤ 4030, or ≤ 448 with 3 or more images) |
| `image_original_sizes` | `(H, W)` of the primary image as given |

`model.generate` returns `MLRSOutput(text, masks, phrases, boxes)`. `masks[i]` is a bool tensor
of shape `[K, H, W]` at `sam_original_sizes` resolution. `processor.post_process_masks` resizes the masks
back to the size of the input image.

## Notes

- The defaults reproduce the evaluation setup: greedy decoding, `max_new_tokens=1024`, and the
  LoRA adapter kept unmerged. `AutoModel.from_pretrained(..., merge_lora=True)` generates faster, with small numerical differences.
- Tested with `transformers==5.18.0`, `peft==0.18.0`, `torch==2.9.0`.

## Citation

```bibtex
@misc{ailuro2026mlrs,
      title={{More with Less}: a Large Scale Remote Sensing {VLM} with a Simple Recipe}, 
      author={Stefan Maria Ailuro and Mario Markov and Mohammad Mahdi and Luc Van Gool and Danda Pani Paudel},
      year={2026},
      eprint={2607.15942},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.15942}, 
}
```
