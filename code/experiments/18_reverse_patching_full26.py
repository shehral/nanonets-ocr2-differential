"""reverse-direction residual patching (base -> OCR2).

A reviewer concern: the v3/v4 paper patches
OCR2's residuals into base to get the causal threshold. The symmetric
test — injecting base's residuals into OCR2's forward at layer L — is
what confirms whether the causal story is bidirectional.

Reuses this stage cached base hidden_states. Loads OCR2 once, runs 26
images x 36 layers of patching, records whether the patched OCR2 top-1
flips away from OCR2's natural top-1 (and whether it flips TO match
what base would predict). Gives per-image reverse-direction threshold.

Output:
    code/results/reverse_patching_full26.json
    code/figures/fig8_reverse_patching.pdf
    docs/checkpoints/phase_2_18_complete.json
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
RESULTS_PATH = Path("code/results/reverse_patching_full26.json")
FIG_PDF = Path("code/figures/fig8_reverse_patching.pdf")
FIG_PNG = Path("code/figures/fig8_reverse_patching.png")

OCR2_REPO = "nanonets/Nanonets-OCR2-3B"

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
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": OCR_PROMPT}],
    }]
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

def kl(a_logits, b_logits):
    a_lp = F.log_softmax(a_logits, dim=-1)
    b_lp = F.log_softmax(b_logits, dim=-1)
    a_p = a_lp.exp()
    return float((a_p * (a_lp - b_lp)).sum().item())

def main():
    pairs = load_manifest()
    print(f"[reverse-patching] {len(pairs)} images", flush=True)

    print("[reverse-patching] loading OCR2 + processor ...", flush=True)
    processor = AutoProcessor.from_pretrained(OCR2_REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        OCR2_REPO, dtype=torch.bfloat16, device_map="mps", attn_implementation="eager",
    )
    model.train(False)
    decoder_layers = model.model.language_model.layers
    num_layers = len(decoder_layers)
    print(f"[reverse-patching] OCR2 loaded; {num_layers} decoder layers", flush=True)

    all_results = []
    for img_i, (iid, cat) in enumerate(pairs):
        t_img = time.time()
        img_path = f"data/processed/image_{iid}_{cat}.png"
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"  [SKIP missing] {img_path}", flush=True)
            continue
        inputs = build_inputs(processor, img)
        T = inputs["input_ids"].shape[1]

        base_p = torch.load(ACT_ROOT / "base" / f"image_{iid}_{cat}.pt",
                            map_location="cpu", weights_only=False)
        base_hs = base_p["hidden_states"]
        if base_hs[0].shape[1] != T:
            print(f"  [SKIP T-mismatch] {iid}_{cat} input={T} cached={base_hs[0].shape[1]}", flush=True)
            continue

        # Also need base's natural top-1 — use the cached attentions or re-derive from final hidden state?
        # Cached base hidden_states go up to L=36 (post-norm). The lm_head isn't cached so we can't
        # trivially recompute base's top-1 from base_hs[-1]. But this stage already captured it separately.
        # For this script we'll load base's top-1 from 17's JSON if available, else fall back to "did it flip from ocr2's natural".
        try:
            sh = json.load(open("code/results/single_head_controls.json"))
            base_tops = {r["image_id"]: r["base_top1"] for r in sh["per_image"]}
            base_top1 = base_tops.get(iid)
        except Exception:
            base_top1 = None

        with torch.no_grad():
            ocr2_nat = model(**inputs, return_dict=True).logits[0, -1].float().cpu()

        per_layer = []
        for L in range(num_layers):
            patch = base_hs[L + 1].to("mps", dtype=torch.bfloat16)
            handle = decoder_layers[L].register_forward_hook(make_patch_hook(patch))
            try:
                with torch.no_grad():
                    out = model(**inputs, return_dict=True)
                patched_last = out.logits[0, -1].float().cpu()
            finally:
                handle.remove()
            del patch

            kl_vs_ocr2 = kl(patched_last, ocr2_nat)
            patched_top1 = int(patched_last.argmax().item())
            top1_eq_ocr2 = int(patched_top1 == int(ocr2_nat.argmax().item()))
            top1_eq_base = int(patched_top1 == base_top1) if base_top1 is not None else None
            per_layer.append({
                "layer": L, "kl_patched_vs_ocr2": kl_vs_ocr2,
                "patched_top1": patched_top1,
                "top1_eq_ocr2": top1_eq_ocr2,
                "top1_eq_base": top1_eq_base,
            })
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Per-image reverse threshold: smallest L where patched top-1 != ocr2's top-1
        rev_threshold = None
        for L in range(num_layers):
            if per_layer[L]["top1_eq_ocr2"] == 0:
                rev_threshold = L
                break

        # Per-image base-match threshold: smallest L where patched top-1 == base's top-1
        base_match_threshold = None
        if base_top1 is not None:
            for L in range(num_layers):
                if per_layer[L]["top1_eq_base"] == 1:
                    base_match_threshold = L
                    break

        all_results.append({
            "image_id": iid, "category": cat, "T": T,
            "ocr2_top1": int(ocr2_nat.argmax().item()),
            "base_top1": base_top1,
            "reverse_threshold_layer": rev_threshold,
            "base_match_threshold_layer": base_match_threshold,
            "per_layer": per_layer,
        })
        print(f"  {img_i + 1:2d}/{len(pairs)} {iid}_{cat}: rev_L={rev_threshold} base_match_L={base_match_threshold}  t={time.time()-t_img:.1f}s", flush=True)

    del model, processor
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if not all_results:
        print("no images")
        return 1

    num_layers = len(all_results[0]["per_layer"])
    mean_kl = []
    frac_flipped_from_ocr2 = []
    frac_matched_base = []
    for L in range(num_layers):
        kls = [r["per_layer"][L]["kl_patched_vs_ocr2"] for r in all_results]
        eq_ocr2 = [r["per_layer"][L]["top1_eq_ocr2"] for r in all_results]
        eq_base = [r["per_layer"][L]["top1_eq_base"] for r in all_results if r["per_layer"][L]["top1_eq_base"] is not None]
        mean_kl.append(float(np.mean(kls)))
        frac_flipped_from_ocr2.append(float(1 - np.mean(eq_ocr2)))
        frac_matched_base.append(float(np.mean(eq_base)) if eq_base else 0.0)

    thresholds = [r["reverse_threshold_layer"] for r in all_results if r["reverse_threshold_layer"] is not None]
    threshold_counter = Counter(thresholds)
    base_match_threshs = [r["base_match_threshold_layer"] for r in all_results if r["base_match_threshold_layer"] is not None]

    # Figure: 3-panel  A. KL curve  B. flip fraction  C. threshold histogram
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    xs = list(range(num_layers))
    axes[0].plot(xs, mean_kl, marker="o", ms=3, color="#7a0177")
    axes[0].set_xlabel("patching layer L")
    axes[0].set_ylabel("mean KL(patched || OCR2-natural)")
    axes[0].set_title("A. Mean KL at last position\nbase residual injected at L, OCR2 continues")
    axes[0].grid(alpha=0.3)

    axes[1].plot(xs, frac_flipped_from_ocr2, marker="s", ms=3, color="#cd4a4a", label="flipped from OCR2's top-1")
    axes[1].plot(xs, frac_matched_base, marker="^", ms=3, color="#2b8cbe", label="matched base's top-1")
    axes[1].set_xlabel("patching layer L")
    axes[1].set_ylabel("fraction of 26 images")
    axes[1].set_title("B. Reverse flip / base-match rate")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].bar(sorted(threshold_counter.keys()), [threshold_counter[k] for k in sorted(threshold_counter.keys())], color="#7a0177")
    axes[2].set_xlabel("reverse threshold layer (first L at which patched OCR2 != OCR2 natural)")
    axes[2].set_ylabel("image count")
    axes[2].set_title(f"C. Per-image reverse threshold distribution\nmedian L={int(np.median(thresholds))}, mode L={max(threshold_counter, key=threshold_counter.get)}")
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle(f"Figure 8 — Reverse-direction residual patching (base -> OCR2), N=26", fontsize=11)
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_images": len(all_results),
        "num_layers": num_layers,
        "mean_kl_per_layer": mean_kl,
        "fraction_flipped_from_ocr2_per_layer": frac_flipped_from_ocr2,
        "fraction_matched_base_per_layer": frac_matched_base,
        "threshold_distribution": {int(k): v for k, v in threshold_counter.items()},
        "reverse_threshold_median": int(np.median(thresholds)) if thresholds else None,
        "reverse_threshold_mean": float(np.mean(thresholds)) if thresholds else None,
        "reverse_threshold_mode": int(max(threshold_counter, key=threshold_counter.get)) if threshold_counter else None,
        "n_images_with_base_match_at_some_L": len(base_match_threshs),
        "base_match_threshold_median": int(np.median(base_match_threshs)) if base_match_threshs else None,
        "per_image": all_results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    ck = Path("docs/checkpoints/phase_2_18_complete.json")
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps({
        "phase": "2.18 reverse-direction residual patching",
        "status": "complete",
        "n_images": len(all_results),
        "reverse_threshold_median": summary["reverse_threshold_median"],
        "reverse_threshold_mode": summary["reverse_threshold_mode"],
        "n_images_with_base_match_at_some_L": summary["n_images_with_base_match_at_some_L"],
        "fig_pdf": str(FIG_PDF),
    }, indent=2))

    print(f"\n[reverse-patching] reverse threshold: median L={summary['reverse_threshold_median']}, mean {summary['reverse_threshold_mean']:.2f}, mode L={summary['reverse_threshold_mode']}")
    print(f"[reverse-patching] {summary['n_images_with_base_match_at_some_L']}/{len(all_results)} images reach base's top-1 at some L, median L={summary['base_match_threshold_median']}")
    print(f"[reverse-patching] threshold distribution: {dict(sorted(threshold_counter.items()))}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
