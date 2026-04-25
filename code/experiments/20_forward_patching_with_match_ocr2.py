"""forward patching (OCR2 -> base) with match-OCR2 tracking.

this stage's result tracks `top1_eq_base` only (patched top-1 matches base
natural or not). this stage controls showed that for head-level patching
`flip != match_ocr2`. We need to verify the §4.2 claim by re-running the
same layer-level patching but additionally recording whether the patched
top-1 matches OCR2's own top-1 at the same token position.

If match-OCR2 is high at late layers (due to shared lm_head) and non-trivial
at the §4.2 threshold layers, the bimodal-threshold story stays valid as
an OCR-directional claim. If match-OCR2 is zero everywhere, the §4.2
'threshold' is also a destabilization result, and the paper's core claim
downgrades accordingly.

Output:
    code/results/forward_patching_match_ocr2.json
    code/figures/fig4b_forward_patching_match_ocr2.pdf
    docs/checkpoints/phase_2_20_complete.json
"""

from __future__ import annotations

import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForImageTextToText, AutoProcessor

ACT_ROOT = Path("code/activations")
RESULTS_PATH = Path("code/results/forward_patching_match_ocr2.json")
FIG_PDF = Path("code/figures/fig4b_forward_patching_match_ocr2.pdf")
FIG_PNG = Path("code/figures/fig4b_forward_patching_match_ocr2.png")

BASE_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"

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
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 256 * 28 * 28

def load_manifest():
    m = json.loads(Path("data/processed/eval_set_manifest.json").read_text())
    return [(e["image_id"], e["category"]) for e in m["images"]]

def build_inputs(processor, image):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": OCR_PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image], videos=None, padding=True, return_tensors="pt",
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )
    return {k: (v.to("mps") if hasattr(v, "to") else v) for k, v in inputs.items()}

def make_patch_hook(patch_tensor):
    def hook(module, args, output):
        if isinstance(output, tuple):
            return (patch_tensor,) + output[1:]
        return patch_tensor
    return hook

def main():
    pairs = load_manifest()
    print(f"[fwd-match] {len(pairs)} images", flush=True)

    # Load OCR2's own top-1 predictions from this stage (already captured)
    sh = json.load(open("code/results/single_head_controls.json"))
    ocr2_top1 = {r["image_id"]: r["ocr2_top1"] for r in sh["per_image"]}

    processor = AutoProcessor.from_pretrained(BASE_REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_REPO, dtype=torch.bfloat16, device_map="mps", attn_implementation="eager",
    )
    model.train(False)
    decoder_layers = model.model.language_model.layers
    num_layers = len(decoder_layers)
    print(f"[fwd-match] base loaded; {num_layers} decoder layers", flush=True)

    all_results = []
    for img_i, (iid, cat) in enumerate(pairs):
        t_img = time.time()
        img_path = f"data/processed/image_{iid}_{cat}.png"
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            continue
        inputs = build_inputs(processor, img)
        T = inputs["input_ids"].shape[1]

        ocr2_p = torch.load(ACT_ROOT / "ocr2" / f"image_{iid}_{cat}.pt",
                            map_location="cpu", weights_only=False)
        ocr2_hs = ocr2_p["hidden_states"]
        if ocr2_hs[0].shape[1] != T:
            continue

        with torch.no_grad():
            base_nat = model(**inputs, return_dict=True).logits[0, -1].float().cpu()
        base_top1 = int(base_nat.argmax().item())
        o_top1 = ocr2_top1.get(iid)

        per_layer = []
        for L in range(num_layers):
            patch = ocr2_hs[L + 1].to("mps", dtype=torch.bfloat16)
            handle = decoder_layers[L].register_forward_hook(make_patch_hook(patch))
            try:
                with torch.no_grad():
                    out = model(**inputs, return_dict=True)
                patched_last = out.logits[0, -1].float().cpu()
            finally:
                handle.remove()
            del patch

            p_top1 = int(patched_last.argmax().item())
            per_layer.append({
                "layer": L,
                "patched_top1": p_top1,
                "flip_vs_base": int(p_top1 != base_top1),
                "match_ocr2": int(p_top1 == o_top1) if o_top1 is not None else None,
            })
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Per-image thresholds
        fwd_flip_L = next((L for L in range(num_layers) if per_layer[L]["flip_vs_base"] == 1), None)
        match_ocr2_L = next((L for L in range(num_layers) if per_layer[L]["match_ocr2"] == 1), None) if o_top1 is not None else None

        all_results.append({
            "image_id": iid, "category": cat,
            "base_top1": base_top1, "ocr2_top1": o_top1,
            "flip_threshold": fwd_flip_L,
            "match_ocr2_threshold": match_ocr2_L,
            "per_layer": per_layer,
        })
        print(f"  {img_i + 1:2d}/{len(pairs)} {iid}_{cat}: flip_L={fwd_flip_L} match_L={match_ocr2_L} t={time.time()-t_img:.1f}s", flush=True)

    del model, processor
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if not all_results:
        return 1

    num_layers = len(all_results[0]["per_layer"])
    frac_flipped = [float(np.mean([r["per_layer"][L]["flip_vs_base"] for r in all_results])) for L in range(num_layers)]
    frac_match_ocr2 = []
    for L in range(num_layers):
        vals = [r["per_layer"][L]["match_ocr2"] for r in all_results if r["per_layer"][L]["match_ocr2"] is not None]
        frac_match_ocr2.append(float(np.mean(vals)) if vals else 0.0)

    flip_thresh = [r["flip_threshold"] for r in all_results if r["flip_threshold"] is not None]
    match_thresh = [r["match_ocr2_threshold"] for r in all_results if r["match_ocr2_threshold"] is not None]
    flip_mode = Counter(flip_thresh).most_common(1)[0][0] if flip_thresh else None
    match_mode = Counter(match_thresh).most_common(1)[0][0] if match_thresh else None

    # Figure: 2-panel overlay of flip-vs-base and match-OCR2 curves, + threshold-distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    xs = list(range(num_layers))
    axes[0].plot(xs, frac_flipped, marker="o", ms=3, color="#cd4a4a", label="fraction flipped vs base")
    axes[0].plot(xs, frac_match_ocr2, marker="s", ms=3, color="#2b8cbe", label="fraction match OCR2's top-1")
    axes[0].set_xlabel("patching layer L")
    axes[0].set_ylabel("fraction of 26 images")
    axes[0].set_title("A. Forward patching (OCR2 -> base): flip rate vs OCR-match rate")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)

    if match_thresh:
        c = Counter(match_thresh)
        axes[1].bar(sorted(c.keys()), [c[k] for k in sorted(c.keys())], color="#2b8cbe")
        axes[1].set_xlabel("match-OCR2 threshold layer")
        axes[1].set_ylabel("image count")
        axes[1].set_title(f"B. First L at which patched base matches OCR2's top-1\nmedian L={int(np.median(match_thresh))}, mode L={match_mode}, n={len(match_thresh)}/26")
        axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Figure 4b — Forward patching with match-OCR2 verification", fontsize=11)
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_images": len(all_results),
        "num_layers": num_layers,
        "fraction_flipped_vs_base_per_layer": frac_flipped,
        "fraction_match_ocr2_per_layer": frac_match_ocr2,
        "flip_threshold_median": int(np.median(flip_thresh)) if flip_thresh else None,
        "flip_threshold_mode": flip_mode,
        "match_ocr2_threshold_median": int(np.median(match_thresh)) if match_thresh else None,
        "match_ocr2_threshold_mode": match_mode,
        "n_images_ever_match_ocr2": len(match_thresh),
        "peak_match_ocr2_rate": max(frac_match_ocr2),
        "per_image": all_results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    ck = Path("docs/checkpoints/phase_2_20_complete.json")
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps({
        "phase": "2.20 forward patching with match-OCR2 tracking",
        "status": "complete",
        "n_images": len(all_results),
        "peak_match_ocr2_rate": summary["peak_match_ocr2_rate"],
        "n_images_ever_match_ocr2": summary["n_images_ever_match_ocr2"],
        "match_ocr2_threshold_median": summary["match_ocr2_threshold_median"],
        "fig_pdf": str(FIG_PDF),
    }, indent=2))

    print(f"\n[fwd-match] peak match-OCR2 rate: {summary['peak_match_ocr2_rate']:.2%}")
    print(f"[fwd-match] {summary['n_images_ever_match_ocr2']}/{len(all_results)} images reach OCR2's top-1 at some L")
    print(f"[fwd-match] match-OCR2 threshold: median L={summary['match_ocr2_threshold_median']}, mode L={match_mode}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
