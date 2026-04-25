"""ViT weight-diff.

The paper's Claim-1 triangulation argument assumes the vision encoder is
shared initialization and unchanged by fine-tuning. We should verify this
cheaply. Streams bf16 vision-block weights layer-by-layer and reports
per-layer + per-module fractional change.

Output:
    code/results/vit_weight_diff.json
    code/figures/fig5b_vit_weight_diff.pdf
    code/figures/fig5b_vit_weight_diff.png
    docs/checkpoints/phase_2_11_complete.json
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForImageTextToText

BASE_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
OCR2_REPO = "nanonets/Nanonets-OCR2-3B"

RESULTS_PATH = Path("code/results/vit_weight_diff.json")
FIG_PDF = Path("code/figures/fig5b_vit_weight_diff.pdf")
FIG_PNG = Path("code/figures/fig5b_vit_weight_diff.png")

def load_vit_blocks_bf16(repo: str) -> dict[int, dict[str, torch.Tensor]]:
    model = AutoModelForImageTextToText.from_pretrained(
        repo, dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager",
    )
    per_layer = {}
    for i, block in enumerate(model.model.visual.blocks):
        sd = {}
        for name, param in block.named_parameters():
            sd[name] = param.detach().clone()
        per_layer[i] = sd
    # Also grab the merger (patch-merger) and patch_embed
    extras = {
        "patch_embed": {n: p.detach().clone() for n, p in model.model.visual.patch_embed.named_parameters()},
        "merger": {n: p.detach().clone() for n, p in model.model.visual.merger.named_parameters()},
    }
    del model
    gc.collect()
    return per_layer, extras

def main() -> int:
    print("[vit-diff] loading base ViT ...", flush=True)
    base_blocks, base_extras = load_vit_blocks_bf16(BASE_REPO)
    print(f"[vit-diff] base: {len(base_blocks)} ViT blocks", flush=True)

    print("[vit-diff] loading OCR2 ViT ...", flush=True)
    ocr_blocks, ocr_extras = load_vit_blocks_bf16(OCR2_REPO)
    print(f"[vit-diff] ocr2: {len(ocr_blocks)} ViT blocks", flush=True)

    num_layers = len(base_blocks)
    per_layer_summary = []
    for L in range(num_layers):
        layer_diff_sq = 0.0
        layer_base_sq = 0.0
        modules = {}
        for name in base_blocks[L]:
            b = base_blocks[L][name]
            o = ocr_blocks[L].get(name)
            if o is None:
                continue
            diff_sq = float((o.float() - b.float()).pow(2).sum().item())
            base_sq = float(b.float().pow(2).sum().item())
            layer_diff_sq += diff_sq
            layer_base_sq += base_sq
            modules[name] = {
                "diff_frobenius": diff_sq ** 0.5,
                "base_frobenius": base_sq ** 0.5,
                "fractional_change": (diff_sq ** 0.5) / (base_sq ** 0.5 + 1e-12),
            }
        per_layer_summary.append({
            "layer": L,
            "total_diff_fro": layer_diff_sq ** 0.5,
            "total_base_fro": layer_base_sq ** 0.5,
            "fractional_change": (layer_diff_sq ** 0.5) / (layer_base_sq ** 0.5 + 1e-12),
            "modules": modules,
        })
        if L % 8 == 0:
            print(f"  vit layer {L:2d}: frac_change={per_layer_summary[-1]['fractional_change']:.6f}", flush=True)

    # Extras
    extras_summary = {}
    for region in ["patch_embed", "merger"]:
        total_diff_sq = 0.0
        total_base_sq = 0.0
        for name in base_extras[region]:
            b = base_extras[region][name]
            o = ocr_extras[region].get(name)
            if o is None:
                continue
            total_diff_sq += float((o.float() - b.float()).pow(2).sum().item())
            total_base_sq += float(b.float().pow(2).sum().item())
        extras_summary[region] = {
            "total_diff_fro": total_diff_sq ** 0.5,
            "total_base_fro": total_base_sq ** 0.5,
            "fractional_change": (total_diff_sq ** 0.5) / (total_base_sq ** 0.5 + 1e-12),
        }

    # Decoder baseline for comparison
    decoder_ref = json.loads(Path("code/results/weight_diff_analysis.json").read_text())
    decoder_frac_mean = float(np.mean([r["fractional_change"] for r in decoder_ref["per_layer"]]))

    vit_frac_mean = float(np.mean([r["fractional_change"] for r in per_layer_summary]))
    vit_frac_max = float(max(r["fractional_change"] for r in per_layer_summary))

    # Figure: compact 2-panel (vit layers + extras, overlay decoder mean for scale)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    xs = [r["layer"] for r in per_layer_summary]
    ys = [r["fractional_change"] for r in per_layer_summary]
    axes[0].bar(xs, ys, color="#2b8cbe")
    axes[0].axhline(decoder_frac_mean, color="#7a0177", linestyle="--", linewidth=1.5,
                    label=f"decoder layer-mean fractional change ({decoder_frac_mean:.4f})")
    axes[0].set_xlabel("ViT block index")
    axes[0].set_ylabel(r"$\|W_{ocr2} - W_{base}\|_F / \|W_{base}\|_F$")
    axes[0].set_title(f"A. Per-ViT-block weight change\n(ViT mean {vit_frac_mean:.5f}, max {vit_frac_max:.5f})")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")

    extras_labels = list(extras_summary.keys())
    extras_vals = [extras_summary[k]["fractional_change"] for k in extras_labels]
    axes[1].bar(extras_labels, extras_vals, color="#969696")
    axes[1].axhline(decoder_frac_mean, color="#7a0177", linestyle="--", linewidth=1.5,
                    label=f"decoder layer-mean ({decoder_frac_mean:.4f})")
    axes[1].set_ylabel(r"fractional Frobenius change")
    axes[1].set_title("B. Patch-embed and patch-merger")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Figure 5b — Vision encoder weight change (ViT + embed/merger)", fontsize=11)
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "num_vit_layers": num_layers,
        "per_layer": per_layer_summary,
        "extras": extras_summary,
        "vit_mean_fractional_change": vit_frac_mean,
        "vit_max_fractional_change": vit_frac_max,
        "decoder_mean_fractional_change_for_comparison": decoder_frac_mean,
        "vit_vs_decoder_ratio": vit_frac_mean / (decoder_frac_mean + 1e-12),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    ck_path = Path("docs/checkpoints/phase_2_11_complete.json")
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    ck_path.write_text(json.dumps({
        "phase": "2.11 ViT weight diff",
        "status": "complete",
        "vit_mean_frac_change": vit_frac_mean,
        "decoder_mean_frac_change": decoder_frac_mean,
        "ratio_vit_to_decoder": summary["vit_vs_decoder_ratio"],
        "fig_pdf": str(FIG_PDF),
    }, indent=2))

    print(f"\n[vit-diff] ViT mean fractional change: {vit_frac_mean:.5f}")
    print(f"[vit-diff] decoder mean fractional change: {decoder_frac_mean:.5f}")
    print(f"[vit-diff] ViT / decoder ratio: {summary['vit_vs_decoder_ratio']:.3f}")
    print(f"[vit-diff] Verdict: ViT is {'substantially' if summary['vit_vs_decoder_ratio'] > 0.5 else 'far less' if summary['vit_vs_decoder_ratio'] < 0.2 else 'moderately'} perturbed vs the decoder")
    return 0

if __name__ == "__main__":
    sys.exit(main())
