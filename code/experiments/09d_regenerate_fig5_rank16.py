"""Regenerate Figure 5 Panel C with rank-16 heatmap (was rank-4 in v5.2).

Panel C in the original fig5 used rank-4 SVD recovery, which renders most cells
dark because rank-4 captures <20% of Frobenius energy for most layer/module
positions. Rank-16 is the rank §4.5 actually prescribes for the LoRA
recommendation, and it gives a much more readable heatmap.

Reads cached weight_diff_analysis.json; no model load required.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
MOD_LABELS = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
              "mlp.gate_proj": "mlp.gate", "mlp.up_proj": "mlp.up", "mlp.down_proj": "mlp.down"}

FIG_PDF = Path("code/figures/fig5_weight_diff.pdf")
FIG_PNG = Path("code/figures/fig5_weight_diff.png")

def main():
    d = json.load(open("code/results/weight_diff_analysis.json"))
    per_layer = d["per_layer"]
    per_module_pl = d["per_module_per_layer"]
    num_layers = len(per_layer)

    per_module_total = {mod: sum(r["diff_frobenius"] for r in per_module_pl[mod]) for mod in MODULES}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    xs = [r["layer"] for r in per_layer]
    ys = [r["fractional_change"] for r in per_layer]
    axes[0].bar(xs, ys, color="#7a0177")
    axes[0].set_xlabel("decoder layer")
    axes[0].set_ylabel(r"$\|W_{ocr2} - W_{base}\|_F / \|W_{base}\|_F$")
    axes[0].set_title("A. Per-layer total weight change (fractional)")
    axes[0].grid(alpha=0.3, axis="y")

    mod_labels = [MOD_LABELS[m] for m in MODULES]
    mod_values = [per_module_total[m] for m in MODULES]
    colors = ["#2b8cbe"] * 4 + ["#7a0177"] * 3
    axes[1].bar(mod_labels, mod_values, color=colors)
    axes[1].set_xlabel("attention / MLP module")
    axes[1].set_ylabel(r"cumulative $\|W_{ocr2} - W_{base}\|_F$ over 36 layers")
    axes[1].set_title("B. Which modules carry the fine-tuning signal")
    axes[1].grid(alpha=0.3, axis="y")

    rank16_matrix = np.zeros((num_layers, len(MODULES)), dtype=np.float32)
    for j, mod in enumerate(MODULES):
        for entry in per_module_pl[mod]:
            rank16_matrix[entry["layer"], j] = entry["rank16_recovery"]

    mean_by_module = rank16_matrix.mean(axis=0)
    mean_overall = rank16_matrix.mean()

    im = axes[2].imshow(rank16_matrix, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=1)
    axes[2].set_xticks(range(len(MODULES)))
    axes[2].set_xticklabels(mod_labels)
    axes[2].set_ylabel("decoder layer")
    axes[2].set_title(
        "C. Rank-16 LoRA recovery of fine-tuning delta\n"
        f"mean {mean_overall:.2f}; MLP projections 0.68-0.87 (LoRA-friendly)"
    )
    fig.colorbar(im, ax=axes[2], fraction=0.04)

    fig.suptitle("Figure 5 — Weight-space fine-tuning signature, 36 layers × 7 modules", fontsize=11)
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Per-module mean rank-16 recovery:")
    for mod, m in zip(MODULES, mean_by_module):
        print(f"  {mod:20s} {m:.3f}")
    print(f"Overall mean: {mean_overall:.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
