"""residual-stream L2 diff. Figure 1 data + render.

Loads every per-image activation .pt file for the three models, computes
layer-wise L2 differences between (ocr2, base) and (ocr_s, base), and renders
a two-panel heatmap.

Output:
    code/results/residual_diff.json
    code/figures/fig1_residual_diff.pdf
    code/figures/fig1_residual_diff.png
    docs/checkpoints/phase_2_3_complete.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ACT_ROOT = Path("code/activations")
RESULTS_PATH = Path("code/results/residual_diff.json")
FIG_PDF = Path("code/figures/fig1_residual_diff.pdf")
FIG_PNG = Path("code/figures/fig1_residual_diff.png")
MODEL_TAGS = ("base", "ocr2", "ocr_s")

def manifest_pairs() -> list[tuple[str, str]]:
    """Image-id × category pairs that exist for all three models."""
    base_files = {p.name for p in (ACT_ROOT / "base").glob("*.pt")}
    ocr2_files = {p.name for p in (ACT_ROOT / "ocr2").glob("*.pt")}
    ocr_s_files = {p.name for p in (ACT_ROOT / "ocr_s").glob("*.pt")}
    common = sorted(base_files & ocr2_files & ocr_s_files)
    pairs = []
    for fn in common:
        stem = fn.replace(".pt", "")
        # image_NN_category
        parts = stem.split("_", 2)
        if len(parts) == 3 and parts[0] == "image":
            pairs.append((parts[1], parts[2]))
    return pairs

def load_hidden(tag: str, image_id: str, category: str) -> list[torch.Tensor]:
    p = ACT_ROOT / tag / f"image_{image_id}_{category}.pt"
    payload = torch.load(p, map_location="cpu", weights_only=False)
    return [h.squeeze(0).float() for h in payload["hidden_states"]]  # each (T, D)

def align_token_count(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim to shared token count (models should return same length given same input)."""
    T = min(a.shape[0], b.shape[0])
    return a[:T], b[:T]

def compute_pair_diff(tag_a: str, tag_b: str, pairs: list[tuple[str, str]]) -> np.ndarray:
    """Return (num_images, num_layers) mean-over-tokens L2-norm of (h_a - h_b).

    Layers are indexed from the embedding (index 0) through each decoder
    block's output — standard HF convention with len == num_hidden_layers + 1.
    We keep all layers so the figure shows the full column.
    """
    per_image = []
    num_layers = None
    for (iid, cat) in pairs:
        hs_a = load_hidden(tag_a, iid, cat)
        hs_b = load_hidden(tag_b, iid, cat)
        assert len(hs_a) == len(hs_b), (len(hs_a), len(hs_b))
        if num_layers is None:
            num_layers = len(hs_a)
        layer_means = np.zeros(len(hs_a), dtype=np.float32)
        for L, (ha, hb) in enumerate(zip(hs_a, hs_b)):
            ha, hb = align_token_count(ha, hb)
            # per-token L2 norm then mean over tokens (as in spec §6.1)
            diff = (ha - hb).norm(dim=-1)  # (T,)
            layer_means[L] = diff.mean().item()
        per_image.append(layer_means)
    arr = np.stack(per_image, axis=0) if per_image else np.zeros((0, num_layers or 0))
    return arr

def render_fig(ocr2_diff: np.ndarray, ocr_s_diff: np.ndarray, pairs: list[tuple[str, str]], out_pdf: Path, out_png: Path) -> None:
    """Two-panel heatmap. Columns = layers, rows = images, colormap = magma."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), constrained_layout=True)
    vmax = max(ocr2_diff.max() if ocr2_diff.size else 1.0, ocr_s_diff.max() if ocr_s_diff.size else 1.0)

    for ax, data, label in (
        (axes[0], ocr2_diff, "OCR2-3B minus Qwen2.5-VL-3B"),
        (axes[1], ocr_s_diff, "OCR-s minus Qwen2.5-VL-3B"),
    ):
        im = ax.imshow(data, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.set_xlabel("layer index (0 = embedding, 36 = final decoder output)")
        ax.set_ylabel("eval image")
        ax.set_title(label)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels([f"{p[0]} {p[1]}" for p in pairs], fontsize=7)
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.03)
    cbar.set_label(r"mean-over-tokens $\|h^\mathrm{tuned}_L - h^\mathrm{base}_L\|_2$")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def corroboration_score(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between the two panels' mean-over-images layer curves.

    High correlation (>0.7) supports the spec's anchor claim: both fine-tunes
    concentrate change in the same layers, so the localization is a property
    of the training procedure rather than a single noisy run.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    a_curve = a.mean(axis=0)
    b_curve = b.mean(axis=0)
    if a_curve.std() < 1e-8 or b_curve.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a_curve, b_curve)[0, 1])

def main() -> int:
    pairs = manifest_pairs()
    if not pairs:
        print("No activation files shared across all three models — run this stage first")
        return 1
    print(f"[residual_diff] {len(pairs)} images common to all three models")

    ocr2_diff = compute_pair_diff("ocr2", "base", pairs)
    ocr_s_diff = compute_pair_diff("ocr_s", "base", pairs)

    summary = {
        "num_images": len(pairs),
        "num_layers": int(ocr2_diff.shape[1]) if ocr2_diff.size else 0,
        "ocr2_vs_base_layer_mean": ocr2_diff.mean(axis=0).tolist() if ocr2_diff.size else [],
        "ocr_s_vs_base_layer_mean": ocr_s_diff.mean(axis=0).tolist() if ocr_s_diff.size else [],
        "ocr2_peak_layer": int(ocr2_diff.mean(axis=0).argmax()) if ocr2_diff.size else -1,
        "ocr_s_peak_layer": int(ocr_s_diff.mean(axis=0).argmax()) if ocr_s_diff.size else -1,
        "ocr2_peak_value": float(ocr2_diff.mean(axis=0).max()) if ocr2_diff.size else 0.0,
        "ocr_s_peak_value": float(ocr_s_diff.mean(axis=0).max()) if ocr_s_diff.size else 0.0,
        "corroboration_score_pearson_r": corroboration_score(ocr2_diff, ocr_s_diff),
        "image_ids": [f"{p[0]}_{p[1]}" for p in pairs],
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    render_fig(ocr2_diff, ocr_s_diff, pairs, FIG_PDF, FIG_PNG)

    ck_path = Path("docs/checkpoints/phase_2_3_complete.json")
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    ck_path.write_text(json.dumps({
        "phase": "2.3 residual diff",
        "status": "complete",
        "fig_pdf": str(FIG_PDF),
        "fig_png": str(FIG_PNG),
        "peak_layer_ocr2": summary["ocr2_peak_layer"],
        "peak_layer_ocr_s": summary["ocr_s_peak_layer"],
        "corroboration_score": summary["corroboration_score_pearson_r"],
        "num_images": summary["num_images"],
    }, indent=2))
    print(f"wrote fig={FIG_PDF}  results={RESULTS_PATH}  checkpoint={ck_path}")
    print(f"peak layers — ocr2:{summary['ocr2_peak_layer']}  ocr_s:{summary['ocr_s_peak_layer']}  corroboration r={summary['corroboration_score_pearson_r']:.3f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
