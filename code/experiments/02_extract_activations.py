"""forward-pass activation extraction per model.

Loads one of {base, ocr2, ocr_s} via HuggingFace AutoModelForImageTextToText,
runs a forward pass on each of the 24 manifest images, and saves the
decoder's hidden_states + attentions + token IDs to per-image .pt files.

Usage:
    python code/experiments/02_extract_activations.py --model base
    python code/experiments/02_extract_activations.py --model ocr2
    python code/experiments/02_extract_activations.py --model ocr_s

Design notes:
- Image tokens capped at ~256 via min_pixels=max_pixels=256*28*28 so the
  attention tensors stay ~2.5 MB/layer/image (~90 MB/image/model total after
  36 layers). Budget for activations dir: ~10 GB across 72 files.
- Models loaded sequentially; MPS cache emptied between runs.
- Prompt matches each model's intended use — the Nanonets OCR prompt comes
  from the OCR2 model card; the base uses a generic OCR prompt so the
  differential is measured on a task both attempt.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_REPOS = {
    "base": "Qwen/Qwen2.5-VL-3B-Instruct",
    "ocr2": "nanonets/Nanonets-OCR2-3B",
    "ocr_s": "nanonets/Nanonets-OCR-s",
}

# Target LLM image-token count ≈ 256 → 448×448-ish (14 px patch * merge_size=2 → 28 px per LLM token)
TARGET_TOKENS = 256
PIXELS_PER_TOKEN = 28 * 28
MIN_PIXELS = TARGET_TOKENS * PIXELS_PER_TOKEN
MAX_PIXELS = TARGET_TOKENS * PIXELS_PER_TOKEN

# Nanonets model card prompt — what OCR2-3B was fine-tuned to respond to
OCR_PROMPT = (
    "Extract the text from the above document as if you were reading it "
    "naturally. Return the tables in html format. Return the equations in "
    "LaTeX representation. If there is an image in the document and image "
    "caption is not present, add a small description of the image inside the "
    "<img></img> tag; otherwise, add the image caption inside <img></img>. "
    "Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL "
    "COPY</watermark>. Page numbers should be wrapped in brackets. Ex: "
    "<page_number>14</page_number> or <page_number>9/22</page_number>. "
    "Prefer using ☐ and ☑ for check boxes."
)

@dataclass
class ImageRec:
    image_id: str
    category: str
    path: str

def load_manifest(manifest_path: Path) -> list[ImageRec]:
    blob = json.loads(manifest_path.read_text())
    recs = []
    for e in blob["images"]:
        recs.append(ImageRec(image_id=e["image_id"], category=e["category"], path=e["image_path"]))
    return recs

def build_inputs(processor, image: Image.Image, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        videos=None,
        padding=True,
        return_tensors="pt",
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    return inputs

def find_image_token_span(input_ids: torch.Tensor, image_token_id: int) -> tuple[int, int] | None:
    """Return (start, end_exclusive) token indices of the image-token run,
    or None if not found."""
    ids = input_ids.squeeze(0).tolist()
    try:
        start = ids.index(image_token_id)
    except ValueError:
        return None
    end = start
    while end < len(ids) and ids[end] == image_token_id:
        end += 1
    return (start, end)

def peak_mps_mb() -> float:
    if not torch.backends.mps.is_available():
        return 0.0
    try:
        return torch.mps.driver_allocated_memory() / (1024 ** 2)
    except Exception:  # noqa: BLE001
        return 0.0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_REPOS.keys()), required=True)
    ap.add_argument("--manifest", default="data/processed/eval_set_manifest.json")
    ap.add_argument("--out-root", default="code/activations")
    ap.add_argument("--limit", type=int, default=None, help="cap number of images (debug)")
    ap.add_argument("--prompt-for-base", choices=["ocr", "terse"], default="ocr",
                    help="prompt used for the base model; 'ocr' keeps apples-to-apples with ocr2/ocr_s")
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    if args.limit:
        manifest = manifest[: args.limit]
    print(f"[{args.model}] {len(manifest)} images from manifest", flush=True)

    repo = MODEL_REPOS[args.model]
    print(f"[{args.model}] loading {repo}", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(repo)
    # attn_implementation="eager" is required for output_attentions=True to
    # return per-head attention probabilities — sdpa/flash_attention_2 both
    # return empty tuples from that path.
    model = AutoModelForImageTextToText.from_pretrained(
        repo, dtype=torch.bfloat16, device_map="mps",
        attn_implementation="eager",
    )
    model.train(False)
    load_dt = time.time() - t0
    print(f"[{args.model}] loaded in {load_dt:.1f}s  peak_mps_mb={peak_mps_mb():.1f}", flush=True)

    # Resolve image_token_id from config
    cfg = model.config
    image_token_id = getattr(cfg, "image_token_id", None)
    if image_token_id is None and hasattr(cfg, "image_token_index"):
        image_token_id = cfg.image_token_index
    print(f"[{args.model}] image_token_id={image_token_id}", flush=True)

    # Choose prompt
    if args.model == "base":
        prompt = OCR_PROMPT if args.prompt_for_base == "ocr" else "Describe the text in this image."
    else:
        prompt = OCR_PROMPT

    out_dir = Path(args.out_root) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image_log = []
    for i, rec in enumerate(manifest):
        t_i = time.time()
        img = Image.open(rec.path).convert("RGB")
        inputs = build_inputs(processor, img, prompt)
        inputs_on_device = {k: (v.to("mps") if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(
                **inputs_on_device,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

        hidden = [h.detach().to("cpu", dtype=torch.bfloat16) for h in out.hidden_states]
        attn = [a.detach().to("cpu", dtype=torch.bfloat16) for a in out.attentions]

        input_ids = inputs["input_ids"]
        image_span = find_image_token_span(input_ids, image_token_id) if image_token_id is not None else None

        tok_ids_list = input_ids.squeeze(0).tolist()
        tok_strs = processor.tokenizer.convert_ids_to_tokens(tok_ids_list)

        payload = {
            "model_tag": args.model,
            "model_repo": repo,
            "image_id": rec.image_id,
            "category": rec.category,
            "image_path": rec.path,
            "hidden_states": hidden,
            "attentions": attn,
            "input_ids": input_ids.squeeze(0).cpu(),
            "token_strings": tok_strs,
            "image_token_id": image_token_id,
            "image_token_span": image_span,
            "token_count": int(input_ids.shape[1]),
        }
        out_path = out_dir / f"image_{rec.image_id}_{rec.category}.pt"
        torch.save(payload, out_path)

        dt = time.time() - t_i
        mps_mb = peak_mps_mb()
        per_image_log.append({"i": i, "id": rec.image_id, "tokens": payload["token_count"], "elapsed_s": round(dt, 2), "mps_mb_peak": round(mps_mb, 1)})
        print(f"[{args.model} {i + 1:2d}/{len(manifest)}] {rec.image_id}_{rec.category} tokens={payload['token_count']} elapsed={dt:.1f}s mps_mb={mps_mb:.0f}", flush=True)

        del out, hidden, attn, inputs, inputs_on_device
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Soft guardrail
        if mps_mb > 20000:
            print(f"[{args.model}] MPS > 20 GB, halting to protect system", flush=True)
            break

    del model, processor
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Checkpoint
    ck_dir = Path("docs/checkpoints")
    ck_dir.mkdir(parents=True, exist_ok=True)
    suffix = {"base": "a_base", "ocr2": "b_ocr2", "ocr_s": "c_ocr_s"}[args.model]
    ck_path = ck_dir / f"phase_2_2{suffix}_complete.json"
    ck_path.write_text(json.dumps({
        "phase": f"2.2 activations — {args.model}",
        "status": "complete" if len(per_image_log) == len(manifest) else "partial",
        "images_processed": len(per_image_log),
        "images_total": len(manifest),
        "out_dir": str(out_dir),
        "per_image": per_image_log,
        "peak_mps_mb": max((r["mps_mb_peak"] for r in per_image_log), default=0),
        "prompt_used": prompt[:200],
    }, indent=2))
    print(f"[{args.model}] checkpoint: {ck_path}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
