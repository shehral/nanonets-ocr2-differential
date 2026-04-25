"""random-residual control 

The v4.2 paper reports a bimodal causal threshold in §4.2: patching OCR2's
residuals into base at the per-image threshold layer flips the top-1.
But a natural question: is L=11 OCR-specific, or is
it just where any sufficiently-perturbed residual becomes non-recoverable?

Control: for each image and each layer L, generate a random residual tensor
matched to OCR2's per-token L2 norm at that layer, inject it into base's
forward at L, and measure whether base's top-1 flips. If the random flip rate
is comparable to OCR2's flip rate at the same layer, the causal threshold
in §4.2 is a generic perturbation-tolerance story, not a fine-tune-specific
one. If much lower, OCR2's residuals have structural content the random
doesn't.

We use Gaussian-matched controls: draw N(0, I) of the same shape as OCR2's
residual at layer L, then rescale per-token to match OCR2's per-token L2
norm at that layer. This matches energy exactly while scrambling direction.

Reuses this stage cached OCR2 activations for the per-token norm target.

Output:
    code/results/random_residual_control.json
    code/figures/fig9_random_control.pdf
    docs/checkpoints/phase_2_19_complete.json
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForImageTextToText, AutoProcessor

ACT_ROOT = Path("code/activations")
RESULTS_PATH = Path("code/results/random_residual_control.json")
FIG_PDF = Path("code/figures/fig9_random_control.pdf")
FIG_PNG = Path("code/figures/fig9_random_control.png")

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

RNG_SEED = 42

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

def gaussian_matched(ocr2_layer_state: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Return a tensor shaped like ocr2_layer_state but with random Gaussian directions and
    per-token L2 norms matched to ocr2_layer_state's per-token norms. Scalar-per-token rescale."""
    # ocr2_layer_state shape: (1, T, hidden)
    x = torch.randn(ocr2_layer_state.shape, generator=generator, dtype=torch.float32)
    target_norm = ocr2_layer_state.float().norm(dim=-1, keepdim=True)  # (1, T, 1)
    curr_norm = x.norm(dim=-1, keepdim=True) + 1e-12
    x = x * (target_norm / curr_norm)
    return x

def main():
    pairs = load_manifest()
    print(f"[random-ctrl] {len(pairs)} images", flush=True)

    processor = AutoProcessor.from_pretrained(BASE_REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_REPO, dtype=torch.bfloat16, device_map="mps", attn_implementation="eager",
    )
    model.train(False)
    decoder_layers = model.model.language_model.layers
    num_layers = len(decoder_layers)
    print(f"[random-ctrl] base loaded; {num_layers} decoder layers", flush=True)

    gen = torch.Generator().manual_seed(RNG_SEED)

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

        ocr2_p = torch.load(ACT_ROOT / "ocr2" / f"image_{iid}_{cat}.pt",
                            map_location="cpu", weights_only=False)
        ocr2_hs = ocr2_p["hidden_states"]
        if ocr2_hs[0].shape[1] != T:
            continue

        with torch.no_grad():
            base_nat = model(**inputs, return_dict=True).logits[0, -1].float().cpu()
        base_top1 = int(base_nat.argmax().item())

        per_layer = []
        for L in range(num_layers):
            # Gaussian-matched at layer L+1 (the output of block L)
            random_patch = gaussian_matched(ocr2_hs[L + 1], gen)
            patch = random_patch.to("mps", dtype=torch.bfloat16)
            handle = decoder_layers[L].register_forward_hook(make_patch_hook(patch))
            try:
                with torch.no_grad():
                    out = model(**inputs, return_dict=True)
                patched_last = out.logits[0, -1].float().cpu()
            finally:
                handle.remove()
            del patch, random_patch

            patched_top1 = int(patched_last.argmax().item())
            flip = int(patched_top1 != base_top1)
            per_layer.append({"layer": L, "patched_top1": patched_top1, "flip": flip})
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Per-image random threshold
        random_threshold = None
        for L in range(num_layers):
            if per_layer[L]["flip"] == 1:
                random_threshold = L
                break

        all_results.append({
            "image_id": iid, "category": cat,
            "base_top1": base_top1,
            "random_threshold_layer": random_threshold,
            "per_layer": per_layer,
        })
        print(f"  {img_i + 1:2d}/{len(pairs)} {iid}_{cat}: random_L={random_threshold}  t={time.time()-t_img:.1f}s", flush=True)

    del model, processor
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if not all_results:
        return 1

    num_layers = len(all_results[0]["per_layer"])
    frac_flipped = [float(np.mean([r["per_layer"][L]["flip"] for r in all_results])) for L in range(num_layers)]

    # Load OCR2 flip curve from this stage for comparison
    ocr2_ref = json.loads(Path("code/results/residual_patching_full26.json").read_text())
    ocr2_flip = ocr2_ref.get("fraction_flipped_per_layer") or [float(1 - np.mean([r["per_layer"][L]["top1_eq_base"] for r in ocr2_ref["per_image"]])) for L in range(num_layers)]

    # Figure: overlay random vs OCR2 flip curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    xs = list(range(num_layers))
    axes[0].plot(xs, frac_flipped, marker="s", ms=3, color="#cd4a4a", label="Gaussian-matched random residual")
    axes[0].plot(xs, ocr2_flip, marker="o", ms=3, color="#7a0177", label="OCR2 residual (§4.2)")
    axes[0].set_xlabel("patching layer L")
    axes[0].set_ylabel("fraction of 26 images with top-1 flipped from base")
    axes[0].set_title("A. Random control vs OCR2 — is the §4.2 threshold OCR-specific?")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Panel B: threshold histogram — distribution of L at which random first flips
    thresh = [r["random_threshold_layer"] for r in all_results if r["random_threshold_layer"] is not None]
    if thresh:
        from collections import Counter
        c = Counter(thresh)
        xs2 = sorted(c.keys())
        axes[1].bar(xs2, [c[k] for k in xs2], color="#cd4a4a")
        axes[1].set_xlabel("random-patch threshold L")
        axes[1].set_ylabel("image count")
        axes[1].set_title(f"B. Per-image random-threshold distribution\nmedian L={int(np.median(thresh))}, mode L={max(c, key=c.get)}")
        axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Figure 9 — Random-residual control", fontsize=11)
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_images": len(all_results),
        "num_layers": num_layers,
        "random_fraction_flipped_per_layer": frac_flipped,
        "ocr2_fraction_flipped_per_layer_for_comparison": list(map(float, ocr2_flip)),
        "random_threshold_median": int(np.median(thresh)) if thresh else None,
        "random_threshold_mean": float(np.mean(thresh)) if thresh else None,
        "rng_seed": RNG_SEED,
        "per_image": all_results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    ck = Path("docs/checkpoints/phase_2_19_complete.json")
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps({
        "phase": "2.19 random-residual control",
        "status": "complete",
        "n_images": len(all_results),
        "random_threshold_median": summary["random_threshold_median"],
        "peak_flip_rate_random": max(frac_flipped),
        "peak_flip_rate_ocr2": max(ocr2_flip),
        "fig_pdf": str(FIG_PDF),
    }, indent=2))

    print(f"\n[random-ctrl] random threshold median L={summary['random_threshold_median']}, mean {summary['random_threshold_mean']:.2f}")
    print(f"[random-ctrl] peak random flip: {max(frac_flipped):.2%}  |  peak OCR2 flip: {max(ocr2_flip):.2%}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
