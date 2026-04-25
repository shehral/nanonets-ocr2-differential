"""this stage (streaming) — weight-space diff, memory-efficient.

Loads both models' decoder state dicts in bfloat16, iterates layer-by-layer
computing per-module diff Frobenius norm + rank-r SVD recovery, then releases.

Avoids the 20+ GB footprint of holding both models' float32 weights.

Output:
    code/results/weight_diff_analysis.json
    code/figures/fig5_weight_diff.pdf
    code/figures/fig5_weight_diff.png
    docs/checkpoints/phase_2_9_complete.json
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

RESULTS_PATH = Path("code/results/weight_diff_analysis.json")
FIG_PDF = Path("code/figures/fig5_weight_diff.pdf")
FIG_PNG = Path("code/figures/fig5_weight_diff.png")

MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
MOD_LABELS = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
              "mlp.gate_proj": "mlp.gate", "mlp.up_proj": "mlp.up", "mlp.down_proj": "mlp.down"}

def load_layer_state_dict(repo: str, layer_idx: int) -> dict[str, torch.Tensor]:
    """Load the full model, extract one layer's weights in bf16, return dict, then free model."""
    model = AutoModelForImageTextToText.from_pretrained(
        repo, dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager",
    )
    layer = model.model.language_model.layers[layer_idx]
    sd = {}
    for name, param in layer.named_parameters():
        sd[name] = param.detach().clone()  # bf16, CPU
    del model, layer
    gc.collect()
    return sd

def load_all_layers_bf16(repo: str) -> dict[int, dict[str, torch.Tensor]]:
    """Load full model once, extract all 36 layers' bf16 weights, free model."""
    model = AutoModelForImageTextToText.from_pretrained(
        repo, dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager",
    )
    per_layer = {}
    for i, layer in enumerate(model.model.language_model.layers):
        sd = {}
        for name, param in layer.named_parameters():
            sd[name] = param.detach().clone()
        per_layer[i] = sd
    del model
    gc.collect()
    return per_layer

def module_key(module: str) -> str:
    if module.startswith("mlp."):
        return f"{module}.weight"
    return f"self_attn.{module}.weight"

def rank_r_recovery(diff: torch.Tensor, r: int) -> float:
    d = diff.float()
    try:
        _, S, _ = torch.linalg.svd(d, full_matrices=False)
    except Exception:
        return 0.0
    total = (S ** 2).sum().item()
    if total == 0:
        return 0.0
    kept = (S[:r] ** 2).sum().item()
    return float(kept / total)

def effective_rank_95(diff: torch.Tensor) -> int:
    d = diff.float()
    try:
        _, S, _ = torch.linalg.svd(d, full_matrices=False)
    except Exception:
        return 0
    total = (S ** 2).sum().item()
    if total == 0:
        return 0
    cumsum = torch.cumsum(S ** 2, dim=0)
    return int(((cumsum / total) < 0.95).sum().item()) + 1

def main() -> int:
    print("[weight-diff] loading base decoder layers (bf16) ...", flush=True)
    base_layers = load_all_layers_bf16(BASE_REPO)
    print(f"[weight-diff] base: {len(base_layers)} layers", flush=True)

    print("[weight-diff] loading OCR2 decoder layers (bf16) ...", flush=True)
    ocr_layers = load_all_layers_bf16(OCR2_REPO)
    print(f"[weight-diff] ocr2: {len(ocr_layers)} layers", flush=True)

    num_layers = len(base_layers)
    results: dict[str, list] = {mod: [] for mod in MODULES}
    per_layer_total_diff = []

    for L in range(num_layers):
        layer_total_diff_sq = 0.0
        layer_total_base_sq = 0.0
        base_sd = base_layers[L]
        ocr_sd = ocr_layers[L]
        for mod in MODULES:
            key = module_key(mod)
            if key not in base_sd:
                continue
            b = base_sd[key]
            o = ocr_sd[key]
            diff = (o - b)
            diff_fro_sq = float((diff.float() ** 2).sum().item())
            base_fro_sq = float((b.float() ** 2).sum().item())
            layer_total_diff_sq += diff_fro_sq
            layer_total_base_sq += base_fro_sq
            diff_fro = diff_fro_sq ** 0.5
            frac_change = diff_fro / (base_fro_sq ** 0.5 + 1e-12)
            r4 = rank_r_recovery(diff, 4)
            r8 = rank_r_recovery(diff, 8)
            r16 = rank_r_recovery(diff, 16)
            eff_rank = effective_rank_95(diff)
            results[mod].append({
                "layer": L,
                "diff_frobenius": diff_fro,
                "fractional_change": frac_change,
                "rank4_recovery": r4,
                "rank8_recovery": r8,
                "rank16_recovery": r16,
                "effective_rank_95": eff_rank,
                "shape": list(diff.shape),
            })
        per_layer_total_diff.append({
            "layer": L,
            "total_diff_fro": layer_total_diff_sq ** 0.5,
            "total_base_fro": layer_total_base_sq ** 0.5,
            "fractional_change": (layer_total_diff_sq ** 0.5) / ((layer_total_base_sq ** 0.5) + 1e-12),
        })
        # Release this layer's state dicts after we're done with them
        del base_layers[L]
        del ocr_layers[L]
        if L % 6 == 0:
            print(f"  layer {L:2d}: Δfro={layer_total_diff_sq**0.5:.2f}  frac={per_layer_total_diff[-1]['fractional_change']:.4f}", flush=True)
        gc.collect()

    per_module_total = {mod: sum(r["diff_frobenius"] for r in results[mod]) for mod in MODULES}
    per_module_rank4_mean = {mod: (sum(r["rank4_recovery"] for r in results[mod]) / max(1, len(results[mod]))) for mod in MODULES}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    xs = [r["layer"] for r in per_layer_total_diff]
    ys = [r["fractional_change"] for r in per_layer_total_diff]
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

    rank4_matrix = np.zeros((num_layers, len(MODULES)), dtype=np.float32)
    for j, mod in enumerate(MODULES):
        for entry in results[mod]:
            rank4_matrix[entry["layer"], j] = entry["rank4_recovery"]
    im = axes[2].imshow(rank4_matrix, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=1)
    axes[2].set_xticks(range(len(MODULES)))
    axes[2].set_xticklabels(mod_labels)
    axes[2].set_ylabel("decoder layer")
    axes[2].set_title("C. Rank-4 LoRA recovery of fine-tuning delta\n(high = LoRA-friendly locus)")
    fig.colorbar(im, ax=axes[2], fraction=0.04)

    fig.suptitle("Figure 5 — Weight-space fine-tuning signature, 36 layers × 7 modules", fontsize=11)
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "per_layer": per_layer_total_diff,
        "per_module_per_layer": results,
        "per_module_total": per_module_total,
        "per_module_rank4_mean": per_module_rank4_mean,
        "max_diff_layer": max(per_layer_total_diff, key=lambda r: r["total_diff_fro"]),
        "max_fractional_change_layer": max(per_layer_total_diff, key=lambda r: r["fractional_change"]),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    ck_path = Path("docs/checkpoints/phase_2_9_complete.json")
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    ck_path.write_text(json.dumps({
        "phase": "2.9 weight-diff LoRA candidate",
        "status": "complete",
        "fig_pdf": str(FIG_PDF),
        "max_diff_layer": summary["max_diff_layer"],
        "max_fractional_change_layer": summary["max_fractional_change_layer"],
        "per_module_rank4_mean": per_module_rank4_mean,
    }, indent=2))

    print(f"\n[weight-diff] max total-diff layer: L{summary['max_diff_layer']['layer']} "
          f"fro={summary['max_diff_layer']['total_diff_fro']:.2f}", flush=True)
    print(f"[weight-diff] max fractional-change layer: L{summary['max_fractional_change_layer']['layer']} "
          f"frac={summary['max_fractional_change_layer']['fractional_change']:.4f}", flush=True)
    print("\n[weight-diff] per-module cumulative change (high = more fine-tuning happens here):", flush=True)
    for mod, val in sorted(per_module_total.items(), key=lambda kv: -kv[1]):
        rank4 = per_module_rank4_mean[mod]
        print(f"  {mod:20s} cum_fro={val:10.2f}  rank4_mean_recovery={rank4:.3f}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
