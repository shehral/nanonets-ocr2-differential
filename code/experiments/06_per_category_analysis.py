"""per-category breakdown of residual-stream L2 diff.

Reviewer nice-to-have #2 (`docs/review.md`): which doc categories drive the
L35 peak? If handwriting drives it, the fine-tune worked hardest on
handwritten data. If multilingual is an outlier, we may be seeing
tokenization effects, not interp signal.

Renders a supplementary 2-panel figure:
    - Left:  per-category mean diff vs layer (one curve per category)
    - Right: peak-layer distribution per image (bar chart of argmax[L] for each image)

Output:
    code/results/per_category_residual.json
    code/figures/fig1b_per_category.pdf
    code/figures/fig1b_per_category.png
    docs/checkpoints/phase_2_6_complete.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ACT_ROOT = Path("code/activations")
RESULTS_PATH = Path("code/results/per_category_residual.json")
FIG_PDF = Path("code/figures/fig1b_per_category.pdf")
FIG_PNG = Path("code/figures/fig1b_per_category.png")

def manifest_pairs() -> list[tuple[str, str, str]]:
    """Return (image_id, full_category, base_category) for files common to base+ocr2."""
    base_files = {p.name for p in (ACT_ROOT / "base").glob("*.pt")}
    ocr2_files = {p.name for p in (ACT_ROOT / "ocr2").glob("*.pt")}
    common = sorted(base_files & ocr2_files)
    out = []
    for fn in common:
        parts = fn.replace(".pt", "").split("_", 2)
        if len(parts) == 3 and parts[0] == "image":
            iid, full_cat = parts[1], parts[2]
            base_cat = full_cat.split("_")[0]  # multilingual_zh_cn → multilingual
            out.append((iid, full_cat, base_cat))
    return out

def layer_diff_per_image(iid: str, full_cat: str) -> np.ndarray:
    base_p = torch.load(ACT_ROOT / "base" / f"image_{iid}_{full_cat}.pt",
                        map_location="cpu", weights_only=False)
    ocr_p = torch.load(ACT_ROOT / "ocr2" / f"image_{iid}_{full_cat}.pt",
                       map_location="cpu", weights_only=False)
    hs_b = base_p["hidden_states"]
    hs_o = ocr_p["hidden_states"]
    n = len(hs_b)
    layer_means = np.zeros(n, dtype=np.float32)
    for L in range(n):
        hb = hs_b[L].squeeze(0).float()
        ho = hs_o[L].squeeze(0).float()
        T = min(hb.shape[0], ho.shape[0])
        diff = (ho[:T] - hb[:T]).norm(dim=-1)
        layer_means[L] = diff.mean().item()
    return layer_means

def main() -> int:
    triples = manifest_pairs()
    if not triples:
        print("No common files")
        return 1

    per_image_curves: dict[str, np.ndarray] = {}
    per_image_peaklayer: dict[str, int] = {}
    by_category: dict[str, list[np.ndarray]] = defaultdict(list)
    for iid, full_cat, base_cat in triples:
        curve = layer_diff_per_image(iid, full_cat)
        key = f"{iid}_{full_cat}"
        per_image_curves[key] = curve
        per_image_peaklayer[key] = int(curve.argmax())
        by_category[base_cat].append(curve)

    # Aggregate per category
    cat_means = {c: np.stack(v).mean(axis=0) for c, v in by_category.items()}
    num_layers = len(next(iter(cat_means.values())))

    # Render 2-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    for k, (cat, curve) in enumerate(sorted(cat_means.items())):
        n_imgs = len(by_category[cat])
        axes[0].plot(range(num_layers), curve, label=f"{cat} (n={n_imgs})",
                     color=palette(k), marker="o", markersize=3)
    axes[0].set_xlabel("decoder layer")
    axes[0].set_ylabel(r"$\mathrm{mean\ }\|h^\mathrm{ocr2}_L - h^\mathrm{base}_L\|_2$")
    axes[0].set_title("Per-category residual-stream diff vs. layer")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(alpha=0.3)

    # Peak-layer argmax histogram
    peaks = list(per_image_peaklayer.values())
    axes[1].hist(peaks, bins=range(min(peaks) - 1, max(peaks) + 2), color="#7a0177", edgecolor="white")
    axes[1].set_xlabel("per-image argmax residual-diff layer")
    axes[1].set_ylabel("count of images")
    axes[1].set_title(f"Per-image peak-layer distribution (N={len(peaks)})")
    axes[1].grid(alpha=0.3)

    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300)
    fig.savefig(FIG_PNG, dpi=200)
    plt.close(fig)

    summary = {
        "num_images": len(triples),
        "num_layers": num_layers,
        "categories": {cat: {
            "n_images": len(curves),
            "mean_curve": np.stack(curves).mean(axis=0).tolist(),
            "peak_layer": int(np.stack(curves).mean(axis=0).argmax()),
            "peak_value": float(np.stack(curves).mean(axis=0).max()),
        } for cat, curves in by_category.items()},
        "per_image_peak_layer": per_image_peaklayer,
        "peak_layer_mode": int(max(set(peaks), key=peaks.count)),
        "peak_layer_distribution": {str(L): peaks.count(L) for L in sorted(set(peaks))},
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    ck_path = Path("docs/checkpoints/phase_2_6_complete.json")
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    ck_path.write_text(json.dumps({
        "phase": "2.6 per-category residual",
        "status": "complete",
        "fig_pdf": str(FIG_PDF),
        "category_peak_layers": {cat: v["peak_layer"] for cat, v in summary["categories"].items()},
        "peak_layer_mode": summary["peak_layer_mode"],
    }, indent=2))

    print(f"wrote fig={FIG_PDF}  results={RESULTS_PATH}")
    for cat, info in sorted(summary["categories"].items()):
        print(f"  {cat}: peak L{info['peak_layer']}  value={info['peak_value']:.2f}  (n={info['n_images']})")
    print(f"per-image peak-layer distribution: {summary['peak_layer_distribution']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
