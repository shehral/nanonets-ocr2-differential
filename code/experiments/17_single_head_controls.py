"""single-head controls + match-OCR2 verification.

A reviewer asked us to run the H-only patching protocol from
this stage on control heads at L11 besides H14, and report whether
patched predictions match OCR2's actual top-1 (not just differ from
base's). Controls:
    L11.H14 — the Hua router head (target).
    L11.H7  — low Frobenius diff (3.82) within L11, reviewer's suggestion.
    L11.H8  — highest Frobenius diff (6.20) within L11, to test whether
              any top-rank head drives flips vs the H14-specific claim.
Also patches all 16 L11 heads together (full-layer) for completeness.

For each image and each intervention, we record:
  - base_top1 (natural base prediction)
  - ocr2_top1 (ocr2's actual prediction at the same token position)
  - patched_top1 (prediction under the intervention)
  - flip_vs_base  = patched_top1 != base_top1
  - match_ocr2    = patched_top1 == ocr2_top1

A 'clean' OCR-directional flip requires flip_vs_base AND match_ocr2.
This lets us separate 'head intervention pushes base toward OCR2' from
'head intervention just destabilizes base.'

Output:
    code/results/single_head_controls.json
    code/figures/fig7_single_head_controls.pdf
    docs/checkpoints/phase_2_17_complete.json
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
OCR2_REPO = "nanonets/Nanonets-OCR2-3B"

RESULTS_PATH = Path("code/results/single_head_controls.json")
FIG_PDF = Path("code/figures/fig7_single_head_controls.pdf")
FIG_PNG = Path("code/figures/fig7_single_head_controls.png")

TARGET_LAYER = 11
TARGET_HEADS = [14, 7, 8]  # H14 router, H7 low-Frobenius, H8 high-Frobenius-within-L11
HEAD_DIM = 128
NUM_HEADS = 16

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

def capture_attn_output_hook(captured_holder):
    def hook(module, inputs):
        captured_holder.append(inputs[0].detach().clone())
    return hook

def patch_head_hook(ocr2_concat, head_idx):
    start = head_idx * HEAD_DIM
    end = start + HEAD_DIM
    def hook(module, inputs):
        x = inputs[0]
        x_new = x.clone()
        T = min(x.shape[1], ocr2_concat.shape[1])
        x_new[:, :T, start:end] = ocr2_concat[:, :T, start:end].to(x.dtype).to(x.device)
        return (x_new,) + inputs[1:]
    return hook

def patch_all_heads_hook(ocr2_concat):
    def hook(module, inputs):
        x = inputs[0]
        x_new = x.clone()
        T = min(x.shape[1], ocr2_concat.shape[1])
        x_new[:, :T, :] = ocr2_concat[:, :T, :].to(x.dtype).to(x.device)
        return (x_new,) + inputs[1:]
    return hook

def main():
    pairs = load_manifest()
    print(f"[head-controls] {len(pairs)} images", flush=True)

    # Step 1: load OCR2, capture L11 attention output per image AND OCR2's top-1
    print("[head-controls] Step 1/2: OCR2 forward — capture L11 attn-out + OCR2 top-1 per image", flush=True)
    processor = AutoProcessor.from_pretrained(OCR2_REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        OCR2_REPO, dtype=torch.bfloat16, device_map="mps", attn_implementation="eager",
    )
    model.train(False)

    ocr2_l11 = {}
    ocr2_top1 = {}
    for i, (iid, cat) in enumerate(pairs):
        img_path = f"data/processed/image_{iid}_{cat}.png"
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            continue
        inputs = build_inputs(processor, img)
        held = []
        handle = model.model.language_model.layers[TARGET_LAYER].self_attn.o_proj.register_forward_pre_hook(
            capture_attn_output_hook(held)
        )
        try:
            with torch.no_grad():
                out = model(**inputs, return_dict=True)
        finally:
            handle.remove()
        if held:
            ocr2_l11[f"{iid}_{cat}"] = held[0].cpu()
        ocr2_top1[f"{iid}_{cat}"] = int(out.logits[0, -1].argmax().item())
        if (i + 1) % 6 == 0:
            print(f"  captured {i + 1}/{len(pairs)}", flush=True)

    del model, processor
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Step 2: load base, patch each head separately and all-heads, measure flips + match-OCR2
    print(f"\n[head-controls] Step 2/2: base forward — patch heads {TARGET_HEADS} + all-heads", flush=True)
    processor = AutoProcessor.from_pretrained(BASE_REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_REPO, dtype=torch.bfloat16, device_map="mps", attn_implementation="eager",
    )
    model.train(False)

    results = []
    for i, (iid, cat) in enumerate(pairs):
        key = f"{iid}_{cat}"
        if key not in ocr2_l11:
            continue
        img = Image.open(f"data/processed/image_{iid}_{cat}.png").convert("RGB")
        inputs = build_inputs(processor, img)

        with torch.no_grad():
            base_top1 = int(model(**inputs, return_dict=True).logits[0, -1].argmax().item())

        ocr2_concat = ocr2_l11[key].to("mps")
        patched_by_head = {}
        for h in TARGET_HEADS:
            handle = model.model.language_model.layers[TARGET_LAYER].self_attn.o_proj.register_forward_pre_hook(
                patch_head_hook(ocr2_concat, h)
            )
            try:
                with torch.no_grad():
                    logits = model(**inputs, return_dict=True).logits[0, -1]
                patched_by_head[h] = int(logits.argmax().item())
            finally:
                handle.remove()

        handle = model.model.language_model.layers[TARGET_LAYER].self_attn.o_proj.register_forward_pre_hook(
            patch_all_heads_hook(ocr2_concat)
        )
        try:
            with torch.no_grad():
                logits = model(**inputs, return_dict=True).logits[0, -1]
            patched_all = int(logits.argmax().item())
        finally:
            handle.remove()

        del ocr2_concat

        o_top1 = ocr2_top1[key]
        row = {
            "image_id": iid, "category": cat,
            "base_top1": base_top1, "ocr2_top1": o_top1,
            "patched_all_heads": patched_all,
            "flip_all_heads": int(patched_all != base_top1),
            "match_ocr2_all_heads": int(patched_all == o_top1),
        }
        for h in TARGET_HEADS:
            pred = patched_by_head[h]
            row[f"patched_h{h}"] = pred
            row[f"flip_h{h}"] = int(pred != base_top1)
            row[f"match_ocr2_h{h}"] = int(pred == o_top1)
        results.append(row)
        if (i + 1) % 6 == 0:
            print(f"  patched {i + 1}/{len(pairs)}", flush=True)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    del model, processor
    gc.collect()

    n = len(results)
    summary_by_head = {}
    for h in TARGET_HEADS:
        n_flip = sum(r[f"flip_h{h}"] for r in results)
        n_match = sum(r[f"match_ocr2_h{h}"] for r in results)
        n_clean = sum(1 for r in results if r[f"flip_h{h}"] and r[f"match_ocr2_h{h}"])
        summary_by_head[f"L11.H{h}"] = {
            "flip_rate": n_flip / n,
            "match_ocr2_rate": n_match / n,
            "clean_ocr_directional_flip_rate": n_clean / n,
            "n_flip": n_flip, "n_match": n_match, "n_clean": n_clean,
        }
    all_n_flip = sum(r["flip_all_heads"] for r in results)
    all_n_match = sum(r["match_ocr2_all_heads"] for r in results)
    all_n_clean = sum(1 for r in results if r["flip_all_heads"] and r["match_ocr2_all_heads"])
    summary_all = {
        "flip_rate": all_n_flip / n,
        "match_ocr2_rate": all_n_match / n,
        "clean_ocr_directional_flip_rate": all_n_clean / n,
        "n_flip": all_n_flip, "n_match": all_n_match, "n_clean": all_n_clean,
    }

    # Figure: grouped bars per head
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    xs = np.arange(len(TARGET_HEADS) + 1)
    head_labels = [f"L11.H{h}\n(Frobenius-diff\nrank-in-L11 ~= "
                   + ("2 router" if h == 14 else "1 top" if h == 8 else "14 low")
                   + ")"
                   for h in TARGET_HEADS]
    labels = head_labels + ["all L11 heads\n(full-layer)"]
    flip_rates = [summary_by_head[f"L11.H{h}"]["flip_rate"] for h in TARGET_HEADS] + [summary_all["flip_rate"]]
    match_rates = [summary_by_head[f"L11.H{h}"]["match_ocr2_rate"] for h in TARGET_HEADS] + [summary_all["match_ocr2_rate"]]
    clean_rates = [summary_by_head[f"L11.H{h}"]["clean_ocr_directional_flip_rate"] for h in TARGET_HEADS] + [summary_all["clean_ocr_directional_flip_rate"]]

    w = 0.28
    axes[0].bar(xs - w, flip_rates, w, color="#fe9929", label="flip vs base")
    axes[0].bar(xs, match_rates, w, color="#2b8cbe", label="match OCR2's top-1")
    axes[0].bar(xs + w, clean_rates, w, color="#7a0177", label="both (clean OCR-directional flip)")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("fraction of 26 images")
    axes[0].set_title("A. Intervention outcome, by head")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].set_ylim(0, max(max(flip_rates), max(match_rates), 0.1) * 1.4)

    # Panel B: per-category clean-flip rate for H14 only (the key result)
    by_cat_h14 = {}
    for r in results:
        c = r["category"].split("_")[0]
        by_cat_h14.setdefault(c, {"n": 0, "clean": 0, "flip": 0, "match": 0})
        by_cat_h14[c]["n"] += 1
        by_cat_h14[c]["flip"] += r["flip_h14"]
        by_cat_h14[c]["match"] += r["match_ocr2_h14"]
        by_cat_h14[c]["clean"] += int(r["flip_h14"] and r["match_ocr2_h14"])
    cats = sorted(by_cat_h14.keys())
    clean_rates_cat = [by_cat_h14[c]["clean"] / by_cat_h14[c]["n"] for c in cats]
    flip_rates_cat = [by_cat_h14[c]["flip"] / by_cat_h14[c]["n"] for c in cats]
    axes[1].bar(np.arange(len(cats)) - 0.2, flip_rates_cat, 0.4, color="#fe9929", label="any flip vs base")
    axes[1].bar(np.arange(len(cats)) + 0.2, clean_rates_cat, 0.4, color="#7a0177", label="clean OCR-directional flip")
    axes[1].set_xticks(np.arange(len(cats)))
    axes[1].set_xticklabels(cats, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("fraction of per-category images")
    axes[1].set_title("B. L11.H14 by category — does the flip point to OCR2?")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_ylim(0, 1)

    fig.suptitle(
        "Figure 7 — Single-head controls for L11 patching "
        "Orange = any prediction flip; Purple = flip matches OCR2's top-1 direction",
        fontsize=10,
    )

    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "target_layer": TARGET_LAYER,
        "target_heads": TARGET_HEADS,
        "n_images": n,
        "by_head": summary_by_head,
        "all_heads": summary_all,
        "by_category_h14": by_cat_h14,
        "per_image": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    ck_path = Path("docs/checkpoints/phase_2_17_complete.json")
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    ck_path.write_text(json.dumps({
        "phase": "2.17 single-head controls + match-OCR2 verification",
        "status": "complete",
        "n_images": n,
        "summary_by_head": summary_by_head,
        "summary_all": summary_all,
        "fig_pdf": str(FIG_PDF),
    }, indent=2))

    print("\n[head-controls] SUMMARY")
    for h in TARGET_HEADS:
        s = summary_by_head[f"L11.H{h}"]
        print(f"  L11.H{h:2d}: flip={s['flip_rate']:.2%}  match_ocr2={s['match_ocr2_rate']:.2%}  clean_OCR_directional={s['clean_ocr_directional_flip_rate']:.2%}")
    s = summary_all
    print(f"  all-L11: flip={s['flip_rate']:.2%}  match_ocr2={s['match_ocr2_rate']:.2%}  clean_OCR_directional={s['clean_ocr_directional_flip_rate']:.2%}")
    print("\n[head-controls] by_category (H14 only):")
    for c in sorted(by_cat_h14.keys()):
        d = by_cat_h14[c]
        print(f"  {c:15s}: flip {d['flip']}/{d['n']}  match {d['match']}/{d['n']}  clean {d['clean']}/{d['n']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
