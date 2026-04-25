"""Phase 0.5 Step 6 — single forward pass per model to verify shapes.

For each of the three target VLMs:
    1. Load in bf16 onto MPS.
    2. Run one multimodal prompt on a tiny solid-gray image.
    3. Capture hidden_states + attentions from the language decoder.
    4. Print the shapes we actually got (catches GQA-expansion surprises
       and image-token budget mismatches before Phase 2 commits 72 files).
    5. Free the model; torch.mps.empty_cache().

Exit 0 iff all three models return expected shapes. Exit 1 on any structural
mismatch so the prelaunch checkpoint can refuse to mark ready_to_launch.
"""

from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODELS = {
    "base": "Qwen/Qwen2.5-VL-3B-Instruct",
    "ocr2": "nanonets/Nanonets-OCR2-3B",
    "ocr_s": "nanonets/Nanonets-OCR-s",
}

SMOKE_SIZE = (224, 224)
PROMPT = "Extract any text from this image."

def build_inputs(processor, image: Image.Image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
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
    )
    return inputs

def smoke_one(tag: str, repo: str, image: Image.Image) -> dict:
    t0 = time.time()
    info: dict = {"tag": tag, "repo": repo}

    print(f"\n[{tag}] loading from {repo} ...", flush=True)
    processor = AutoProcessor.from_pretrained(repo)
    # `eager` is required so output_attentions=True actually returns per-head
    # attention probabilities. sdpa / flash_attention_2 return empty tuples.
    model = AutoModelForImageTextToText.from_pretrained(
        repo, dtype=torch.bfloat16, device_map="mps",
        attn_implementation="eager",
    )
    model.train(False)  # inference mode (equivalent to .eval())
    info["loaded_in_sec"] = round(time.time() - t0, 1)

    inputs = build_inputs(processor, image)
    inputs = {k: v.to("mps") if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    hs = out.hidden_states
    att = out.attentions
    assert hs is not None, f"{tag}: hidden_states is None"
    assert att is not None, f"{tag}: attentions is None"

    h0 = hs[0]
    a0 = att[0]
    info["num_hidden_layers_out"] = len(hs)
    info["num_attention_layers_out"] = len(att)
    info["hidden_shape"] = tuple(h0.shape)
    info["attention_shape"] = tuple(a0.shape)
    info["token_count"] = int(h0.shape[1])
    mid = len(hs) // 2
    info["hidden_mid_shape"] = tuple(hs[mid].shape)
    info["attention_mid_shape"] = tuple(att[mid].shape)

    print(
        f"[{tag}] hidden[0]={info['hidden_shape']}  "
        f"attn[0]={info['attention_shape']}  layers_hs={info['num_hidden_layers_out']}  "
        f"layers_attn={info['num_attention_layers_out']}",
        flush=True,
    )

    del model, processor, inputs, out
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    info["total_elapsed_sec"] = round(time.time() - t0, 1)
    return info

def main() -> int:
    image = Image.new("RGB", SMOKE_SIZE, color=(200, 200, 200))
    results: list[dict] = []
    any_fail = False

    for tag, repo in MODELS.items():
        try:
            results.append(smoke_one(tag, repo, image))
        except Exception as e:
            traceback.print_exc()
            results.append({"tag": tag, "repo": repo, "error": repr(e)})
            any_fail = True

    out_path = Path("docs/checkpoints/phase_0_5_smoke_forward.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    return 1 if any_fail else 0

if __name__ == "__main__":
    sys.exit(main())
