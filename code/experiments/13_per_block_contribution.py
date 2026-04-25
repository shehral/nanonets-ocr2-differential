"""per-block residual contribution diff (CPU, decomposes L35 peak).

Fig 1 shows the residual stream L2 diff grows from ~25 at L0 to ~214 at L35.
That's CUMULATIVE: each block adds some perturbation on top of the previous.
Which blocks contribute the most to the per-layer DELTA? If most of the L35
peak comes from blocks 30-35, fine-tuning re-wrote the late MLP/attention
outputs; if it's spread across all 36 blocks, it's diffuse.

We compute per-block contribution diff:
    delta_block[L] = mean over images of || (h_fine[L+1] - h_fine[L]) - (h_base[L+1] - h_base[L]) ||_2
                   = mean || (block output)_fine - (block output)_base ||_2

This is CPU-only (reads cached this stage activations), and disambiguates
"cumulative residual-stream signature" (Fig 1) from "per-block contribution
signature" (this figure).

Output:
    code/results/per_block_contribution.json
    code/figures/fig1c_per_block_contribution.pdf
    docs/checkpoints/phase_2_13_complete.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ACT_ROOT = Path("code/activations")
RESULTS_PATH = Path("code/results/per_block_contribution.json")
FIG_PDF = Path("code/figures/fig1c_per_block_contribution.pdf")
FIG_PNG = Path("code/figures/fig1c_per_block_contribution.png")

def manifest_pairs() -> list[tuple[str, str]]:
    base_files = {p.name for p in (ACT_ROOT / "base").glob("*.pt")}
    ocr2_files = {p.name for p in (ACT_ROOT / "ocr2").glob("*.pt")}
    ocr_s_files = {p.name for p in (ACT_ROOT / "ocr_s").glob("*.pt")}
    common = sorted(base_files & ocr2_files & ocr_s_files)
    pairs = []
    for fn in common:
        parts = fn.replace(".pt", "").split("_", 2)
        if len(parts) == 3 and parts[0] == "image":
            pairs.append((parts[1], parts[2]))
    return pairs

def per_block_contrib_for_image(tag_fine: str, tag_base: str, iid: str, cat: str) -> np.ndarray:
    """Return per-block L2 diff of block output (h[L+1] - h[L] for each block L)."""
    fine = torch.load(ACT_ROOT / tag_fine / f"image_{iid}_{cat}.pt",
                      map_location="cpu", weights_only=False)
    base = torch.load(ACT_ROOT / tag_base / f"image_{iid}_{cat}.pt",
                      map_location="cpu", weights_only=False)
    hs_f = fine["hidden_states"]
    hs_b = base["hidden_states"]
    n = len(hs_f)  # 37
    n_blocks = n - 1  # 36
    out = np.zeros(n_blocks, dtype=np.float32)
    for L in range(n_blocks):
        fb = (hs_f[L + 1].squeeze(0).float() - hs_f[L].squeeze(0).float())
        bb = (hs_b[L + 1].squeeze(0).float() - hs_b[L].squeeze(0).float())
        T = min(fb.shape[0], bb.shape[0])
        out[L] = (fb[:T] - bb[:T]).norm(dim=-1).mean().item()
    return out

def main() -> int:
    pairs = manifest_pairs()
    print(f"[per-block] {len(pairs)} images common to all three models", flush=True)

    ocr2_contribs = []
    ocr_s_contribs = []
    for iid, cat in pairs:
        ocr2_contribs.append(per_block_contrib_for_image("ocr2", "base", iid, cat))
        ocr_s_contribs.append(per_block_contrib_for_image("ocr_s", "base", iid, cat))

    ocr2_mat = np.stack(ocr2_contribs)  # (N, 36)
    ocr_s_mat = np.stack(ocr_s_contribs)
    ocr2_mean = ocr2_mat.mean(axis=0)
    ocr_s_mean = ocr_s_mat.mean(axis=0)

    # Also load the original Fig 1 data for overlay comparison (cumulative)
    fig1 = json.loads(Path("code/results/residual_diff.json").read_text())
    ocr2_cumulative = fig1["ocr2_vs_base_layer_mean"]  # 37 values
    ocr_s_cumulative = fig1["ocr_s_vs_base_layer_mean"]

    # Peak contribution analysis
    peak_block_ocr2 = int(ocr2_mean.argmax())
    peak_val_ocr2 = float(ocr2_mean.max())
    top5_ocr2 = np.argsort(ocr2_mean)[::-1][:5].tolist()
    contrib_share_last6 = float(ocr2_mean[-6:].sum() / ocr2_mean.sum())
    contrib_share_first6 = float(ocr2_mean[:6].sum() / ocr2_mean.sum())

    print(f"[per-block] peak contribution block: L{peak_block_ocr2}, value {peak_val_ocr2:.3f}", flush=True)
    print(f"[per-block] top-5 OCR2 contrib blocks: {top5_ocr2}", flush=True)
    print(f"[per-block] share from last 6 blocks (30-35): {contrib_share_last6:.3f}", flush=True)
    print(f"[per-block] share from first 6 blocks (0-5): {contrib_share_first6:.3f}", flush=True)

    # Figure: 2-panel — per-block contribution vs cumulative residual-diff
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    xs = list(range(36))
    axes[0].bar(xs, ocr2_mean, alpha=0.75, color="#7a0177", label="OCR2 block contrib")
    axes[0].bar(xs, ocr_s_mean, alpha=0.55, color="#2b8cbe", label="OCR-s block contrib")
    axes[0].set_xlabel("block L (attn + MLP contribution at this block)")
    axes[0].set_ylabel(r"$\|\Delta_\mathrm{ocr2}[L] - \Delta_\mathrm{base}[L]\|_2$ mean")
    axes[0].set_title(f"A. Per-block contribution diff\npeak block L{peak_block_ocr2}; last 6 blocks = {contrib_share_last6*100:.0f}% of total")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")

    xs_full = list(range(37))
    axes[1].plot(xs_full, ocr2_cumulative, marker="o", ms=3, color="#7a0177", label="OCR2 cumulative residual-diff")
    axes[1].plot(xs_full, ocr_s_cumulative, marker="s", ms=3, color="#2b8cbe", label="OCR-s cumulative")
    axes[1].set_xlabel("residual-stream layer index (0 = embed, 36 = post-norm output)")
    axes[1].set_ylabel(r"$\|h_\mathrm{ocr2}[L] - h_\mathrm{base}[L]\|_2$ mean")
    axes[1].set_title("B. Cumulative residual-stream diff (ref: Fig 1)\nPeak at L35; L36 drops due to post-norm.")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Figure 1c — Per-block contribution vs cumulative residual-stream diff", fontsize=11)
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_images": len(pairs),
        "ocr2_per_block_mean": ocr2_mean.tolist(),
        "ocr_s_per_block_mean": ocr_s_mean.tolist(),
        "ocr2_peak_block": peak_block_ocr2,
        "ocr2_peak_value": peak_val_ocr2,
        "ocr2_top5_blocks": top5_ocr2,
        "ocr2_share_last6_blocks": contrib_share_last6,
        "ocr2_share_first6_blocks": contrib_share_first6,
        "ocr2_share_mid24_blocks": float(ocr2_mean[6:30].sum() / ocr2_mean.sum()),
        "corroboration_r_on_block_curves": float(np.corrcoef(ocr2_mean, ocr_s_mean)[0, 1]),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    ck_path = Path("docs/checkpoints/phase_2_13_complete.json")
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    ck_path.write_text(json.dumps({
        "phase": "2.13 per-block contribution",
        "status": "complete",
        "n_images": len(pairs),
        "peak_block": peak_block_ocr2,
        "top5_blocks": top5_ocr2,
        "share_last6": contrib_share_last6,
        "share_first6": contrib_share_first6,
        "cross_pair_r": summary["corroboration_r_on_block_curves"],
        "fig_pdf": str(FIG_PDF),
    }, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
