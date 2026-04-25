"""ViT weight diff for the second fine-tune (OCR-s vs base).

The v4 paper flags in §4.6 that the decoder triangulation (OCR2-vs-base and
OCR-s-vs-base residual curves at r=0.996) is not a test of ViT-side parity —
we only measured ViT on OCR2. This script measures the OCR-s ViT diff so the
paper can state directly whether the ViT-perturbation signature also
reproduces across independent fine-tunes. CPU-only, streams bf16.

Output:
    code/results/vit_weight_diff_ocr_s.json
    code/figures/fig5c_vit_weight_diff_triangulation.pdf
    docs/checkpoints/phase_2_11b_complete.json
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
OCR_S_REPO = "nanonets/Nanonets-OCR-s"

RESULTS_PATH = Path("code/results/vit_weight_diff_ocr_s.json")
FIG_PDF = Path("code/figures/fig5c_vit_weight_diff_triangulation.pdf")
FIG_PNG = Path("code/figures/fig5c_vit_weight_diff_triangulation.png")

def load_vit_blocks_bf16(repo: str):
    m = AutoModelForImageTextToText.from_pretrained(
        repo, dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager",
    )
    per_layer = {}
    for i, block in enumerate(m.model.visual.blocks):
        per_layer[i] = {n: p.detach().clone() for n, p in block.named_parameters()}
    extras = {
        "patch_embed": {n: p.detach().clone() for n, p in m.model.visual.patch_embed.named_parameters()},
        "merger": {n: p.detach().clone() for n, p in m.model.visual.merger.named_parameters()},
    }
    del m
    gc.collect()
    return per_layer, extras

def main() -> int:
    print("[vit-diff-ocr_s] loading base ViT ...", flush=True)
    base_blocks, base_extras = load_vit_blocks_bf16(BASE_REPO)
    print(f"[vit-diff-ocr_s] base: {len(base_blocks)} ViT blocks", flush=True)

    print("[vit-diff-ocr_s] loading OCR-s ViT ...", flush=True)
    ocr_blocks, ocr_extras = load_vit_blocks_bf16(OCR_S_REPO)
    print(f"[vit-diff-ocr_s] ocr_s: {len(ocr_blocks)} ViT blocks", flush=True)

    num_layers = len(base_blocks)
    per_layer_summary = []
    for L in range(num_layers):
        layer_diff_sq = 0.0
        layer_base_sq = 0.0
        for name in base_blocks[L]:
            b = base_blocks[L][name]
            o = ocr_blocks[L].get(name)
            if o is None:
                continue
            diff_sq = float((o.float() - b.float()).pow(2).sum().item())
            base_sq = float(b.float().pow(2).sum().item())
            layer_diff_sq += diff_sq
            layer_base_sq += base_sq
        per_layer_summary.append({
            "layer": L,
            "total_diff_fro": layer_diff_sq ** 0.5,
            "total_base_fro": layer_base_sq ** 0.5,
            "fractional_change": (layer_diff_sq ** 0.5) / (layer_base_sq ** 0.5 + 1e-12),
        })
        if L % 8 == 0:
            print(f"  vit layer {L:2d}: frac_change={per_layer_summary[-1]['fractional_change']:.6f}", flush=True)

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

    ocr2_ref = json.loads(Path("code/results/vit_weight_diff.json").read_text())
    ocr2_per_layer = [r["fractional_change"] for r in ocr2_ref["per_layer"]]
    ocr_s_per_layer = [r["fractional_change"] for r in per_layer_summary]
    pearson_r = float(np.corrcoef(ocr2_per_layer, ocr_s_per_layer)[0, 1])

    vit_frac_mean = float(np.mean(ocr_s_per_layer))
    vit_frac_max = float(max(ocr_s_per_layer))
    ocr2_frac_mean = float(np.mean(ocr2_per_layer))

    # Figure: overlay ocr2 and ocr_s per-layer curves
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    xs = list(range(num_layers))
    ax.plot(xs, ocr2_per_layer, marker="o", ms=3, color="#7a0177",
            label=f"OCR2 vs base  (mean {ocr2_frac_mean:.4f})")
    ax.plot(xs, ocr_s_per_layer, marker="s", ms=3, color="#2b8cbe",
            label=f"OCR-s vs base  (mean {vit_frac_mean:.4f})")
    ax.set_xlabel("ViT block index")
    ax.set_ylabel(r"$\|W_{ft} - W_{base}\|_F / \|W_{base}\|_F$")
    ax.set_title(f"Figure 5c — ViT weight-diff triangulation\n"
                 f"Pearson r = {pearson_r:.3f} across 32 blocks")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "num_vit_layers": num_layers,
        "per_layer": per_layer_summary,
        "extras": extras_summary,
        "ocr_s_vit_mean_fractional_change": vit_frac_mean,
        "ocr_s_vit_max_fractional_change": vit_frac_max,
        "ocr2_vit_mean_fractional_change_for_comparison": ocr2_frac_mean,
        "pearson_r_ocr2_vs_ocr_s_vit_curves": pearson_r,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    ck_path = Path("docs/checkpoints/phase_2_11b_complete.json")
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    ck_path.write_text(json.dumps({
        "phase": "2.11b ViT weight diff (OCR-s)",
        "status": "complete",
        "ocr_s_vit_mean": vit_frac_mean,
        "ocr2_vit_mean": ocr2_frac_mean,
        "pearson_r_curves": pearson_r,
        "fig_pdf": str(FIG_PDF),
    }, indent=2))

    print(f"\n[vit-diff-ocr_s] OCR-s ViT mean fractional change: {vit_frac_mean:.5f}")
    print(f"[vit-diff-ocr_s] OCR2 ViT mean (for comparison):     {ocr2_frac_mean:.5f}")
    print(f"[vit-diff-ocr_s] Pearson r across 32 blocks: {pearson_r:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
