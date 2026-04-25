"""patching on all 26 images 

Reuses the working 08b approach: load base once, patch OCR2 residuals from
the this stage cache at every layer boundary for every image, record KL vs
base natural + top-1 flip fraction.

Output:
    code/results/residual_patching_full26.json
    code/figures/fig4_causal_patching.pdf  (updated — overwrites 4-image version)
    code/figures/fig4_causal_patching.png
    docs/checkpoints/phase_2_8_full26_complete.json
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
RESULTS_PATH = Path("code/results/residual_patching_full26.json")
FIG_PDF = Path("code/figures/fig4_causal_patching.pdf")
FIG_PNG = Path("code/figures/fig4_causal_patching.png")

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

def load_manifest() -> list[tuple[str, str]]:
    m = json.loads(Path("data/processed/eval_set_manifest.json").read_text())
    return [(e["image_id"], e["category"]) for e in m["images"]]

def build_inputs(processor, image):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": OCR_PROMPT},
        ],
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

def main() -> int:
    pairs = load_manifest()
    print(f"[patching26] {len(pairs)} images", flush=True)

    print("[patching26] loading base + processor ...", flush=True)
    processor = AutoProcessor.from_pretrained(BASE_REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_REPO, dtype=torch.bfloat16, device_map="mps", attn_implementation="eager",
    )
    model.train(False)
    decoder_layers = model.model.language_model.layers
    num_layers = len(decoder_layers)
    print(f"[patching26] base loaded; {num_layers} decoder layers", flush=True)

    all_results = []
    t0 = time.time()

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

        ocr2_p = torch.load(ACT_ROOT / "ocr2" / f"image_{iid}_{cat}.pt",
                            map_location="cpu", weights_only=False)
        ocr2_hs = ocr2_p["hidden_states"]
        if ocr2_hs[0].shape[1] != T:
            print(f"  [SKIP T-mismatch] {iid}_{cat} input={T} cached={ocr2_hs[0].shape[1]}", flush=True)
            continue

        with torch.no_grad():
            base_nat = model(**inputs, return_dict=True).logits[0, -1].float().cpu()

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

            kl_vs_base = kl(patched_last, base_nat)
            top1_eq_base = int(patched_last.argmax().item() == base_nat.argmax().item())
            per_layer.append({"layer": L, "kl_patched_vs_base": kl_vs_base, "top1_eq_base": top1_eq_base})

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Per-image threshold: smallest L where patched flips top-1 (100% -> 0 since it's binary per-image)
        threshold = None
        for L in range(num_layers):
            if per_layer[L]["top1_eq_base"] == 0:  # flipped
                threshold = L
                break

        all_results.append({
            "image_id": iid, "category": cat, "T": T,
            "base_top1": int(base_nat.argmax().item()),
            "threshold_layer": threshold,
            "per_layer": per_layer,
        })
        print(f"  {img_i + 1:2d}/{len(pairs)} {iid}_{cat}: threshold=L{threshold}  elapsed={time.time() - t_img:.1f}s", flush=True)

    del model, processor
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if not all_results:
        print("No images processed")
        return 1

    # Aggregate
    num_layers = len(all_results[0]["per_layer"])
    mean_kl = []
    frac_flipped = []
    for L in range(num_layers):
        kls = [r["per_layer"][L]["kl_patched_vs_base"] for r in all_results]
        eqs = [r["per_layer"][L]["top1_eq_base"] for r in all_results]
        mean_kl.append(float(np.mean(kls)))
        frac_flipped.append(float(1 - np.mean(eqs)))

    thresholds = [r["threshold_layer"] for r in all_results if r["threshold_layer"] is not None]
    threshold_dist = Counter(thresholds)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    xs = range(num_layers)
    axes[0].plot(xs, mean_kl, marker="o", color="#7a0177", ms=3)
    axes[0].set_xlabel("patched decoder layer (OCR2 residual replaces base's)")
    axes[0].set_ylabel("KL(patched || base_natural) at last-position")
    axes[0].set_title(f"A. Divergence from base under OCR2 residual patching\n(N={len(all_results)} images)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(xs, frac_flipped, marker="s", color="#2b8cbe", ms=3)
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    axes[1].set_xlabel("patched decoder layer")
    axes[1].set_ylabel("fraction of images with flipped top-1 prediction")
    axes[1].set_title("B. Top-1 token flip rate under patching")
    axes[1].grid(alpha=0.3)

    threshold_layers = sorted(threshold_dist.keys())
    threshold_counts = [threshold_dist[L] for L in threshold_layers]
    axes[2].bar(threshold_layers, threshold_counts, color="#2b8cbe", edgecolor="black")
    axes[2].set_xlabel("per-image causal threshold layer\n(earliest L where patched top-1 flips)")
    axes[2].set_ylabel("image count")
    axes[2].set_title(f"C. Per-image threshold distribution\n(mode L={max(threshold_dist, key=threshold_dist.get)}, N={len(thresholds)}/{len(all_results)} with threshold)")
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle(f"Figure 4 — Residual-stream causal patching ({len(all_results)} images)", fontsize=11)
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    dt = time.time() - t0
    summary = {
        "n_images_processed": len(all_results),
        "per_image": all_results,
        "per_layer_mean_kl_vs_base": mean_kl,
        "per_layer_frac_top1_flipped": frac_flipped,
        "threshold_distribution": dict(threshold_dist),
        "threshold_mode": int(max(threshold_dist, key=threshold_dist.get)) if threshold_dist else None,
        "threshold_mean": float(np.mean(thresholds)) if thresholds else None,
        "threshold_median": float(np.median(thresholds)) if thresholds else None,
        "duration_seconds": dt,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    ck_path = Path("docs/checkpoints/phase_2_8_full26_complete.json")
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    ck_path.write_text(json.dumps({
        "phase": "2.8 residual-stream causal patching (full 26 images)",
        "status": "complete",
        "n_images": len(all_results),
        "duration_seconds": round(dt, 1),
        "threshold_distribution": dict(threshold_dist),
        "threshold_mode": summary["threshold_mode"],
        "threshold_mean": summary["threshold_mean"],
        "threshold_median": summary["threshold_median"],
        "fig_pdf": str(FIG_PDF),
    }, indent=2))

    print(f"\n[patching26] threshold distribution: {dict(threshold_dist)}")
    print(f"[patching26] mode={summary['threshold_mode']}  mean={summary['threshold_mean']:.2f}  median={summary['threshold_median']:.2f}")
    print(f"[patching26] wrote {FIG_PDF}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
