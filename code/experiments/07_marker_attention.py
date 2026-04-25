"""attention TO structured-marker tokens per head.

Tests the "structured-output specialization" claim directly rather than via
average Frobenius diff. For each head (L, H), compute how much attention
flows TO the structured-marker positions in the instruction — i.e., the
positions where the tokens {img, water, mark, page, _number, signature,
☐, ☑} appear inside the prompt's few-shot example. If OCR2's top-12 heads
specialize in structured-output emission, they should attend MORE to those
marker positions than the base does.

Output:
    code/results/marker_attention.json
    code/figures/fig2b_marker_attention.pdf
    code/figures/fig2b_marker_attention.png
    docs/checkpoints/phase_2_7_complete.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ACT_ROOT = Path("code/activations")
RESULTS_PATH = Path("code/results/marker_attention.json")
FIG_PDF = Path("code/figures/fig2b_marker_attention.pdf")
FIG_PNG = Path("code/figures/fig2b_marker_attention.png")

# Subword pieces that appear as parts of the structured-output markers
# in Qwen's BPE tokenization of the Nanonets OCR prompt.
MARKER_SUBWORDS = {
    "img", "water", "mark", "page", "_number",
    "signature", "watermark", "page_number",
}

def manifest_pairs() -> list[tuple[str, str]]:
    base_files = {p.name for p in (ACT_ROOT / "base").glob("*.pt")}
    ocr2_files = {p.name for p in (ACT_ROOT / "ocr2").glob("*.pt")}
    common = sorted(base_files & ocr2_files)
    out = []
    for fn in common:
        parts = fn.replace(".pt", "").split("_", 2)
        if len(parts) == 3 and parts[0] == "image":
            out.append((parts[1], parts[2]))
    return out

def find_marker_positions(payload: dict) -> np.ndarray:
    """Binary mask over tokens: 1 if token is part of a structured marker."""
    toks = payload["token_strings"]
    T = len(toks)
    mask = np.zeros(T, dtype=np.int64)
    for i, t in enumerate(toks):
        tclean = t.replace("Ġ", "").replace("▁", "").strip().lower()
        if tclean in {m.lower() for m in MARKER_SUBWORDS}:
            mask[i] = 1
    return mask

def per_head_attention_to_markers(tag: str, iid: str, cat: str) -> np.ndarray | None:
    """Return (num_layers, num_heads) tensor: mean attention flowing to marker positions per head."""
    p = torch.load(ACT_ROOT / tag / f"image_{iid}_{cat}.pt",
                   map_location="cpu", weights_only=False)
    marker_mask = find_marker_positions(p)
    if marker_mask.sum() == 0:
        return None
    marker_idx = np.where(marker_mask == 1)[0]
    attn = p["attentions"]  # list of (1, H, T, T)
    num_layers = len(attn)
    num_heads = attn[0].shape[1]
    result = np.zeros((num_layers, num_heads), dtype=np.float32)
    for L in range(num_layers):
        A = attn[L].squeeze(0).float().numpy()  # (H, T_q, T_k)
        # sum attention weight placed ONTO marker positions, averaged over query positions
        # (how much does each query position attend to markers, averaged over queries)
        attn_to_markers = A[:, :, marker_idx].sum(axis=-1).mean(axis=-1)  # (H,)
        result[L] = attn_to_markers
    return result

def main() -> int:
    pairs = manifest_pairs()
    if not pairs:
        return 1

    # Sample the first image to check marker count
    sample = torch.load(ACT_ROOT / "base" / f"image_{pairs[0][0]}_{pairs[0][1]}.pt",
                        map_location="cpu", weights_only=False)
    marker_mask = find_marker_positions(sample)
    print(f"[markers] {int(marker_mask.sum())} marker tokens out of {len(sample['token_strings'])} in sample image")
    if marker_mask.sum() == 0:
        print("No marker tokens found — check subword list against prompt tokenization")
        return 1

    # Accumulate across images
    n_images = 0
    ocr2_sum = None
    base_sum = None
    for iid, cat in pairs:
        ocr2_a = per_head_attention_to_markers("ocr2", iid, cat)
        base_a = per_head_attention_to_markers("base", iid, cat)
        if ocr2_a is None or base_a is None:
            continue
        if ocr2_sum is None:
            ocr2_sum = np.zeros_like(ocr2_a)
            base_sum = np.zeros_like(base_a)
        ocr2_sum += ocr2_a
        base_sum += base_a
        n_images += 1

    ocr2_mean = ocr2_sum / n_images
    base_mean = base_sum / n_images
    marker_diff = ocr2_mean - base_mean  # positive = OCR2 attends more to markers

    # Cross-reference with top-12 heads from attention_diff.json
    ad = json.loads(Path("code/results/attention_diff.json").read_text())
    top12 = ad["top_heads"]

    top_heads_marker_story = []
    for h in top12:
        L, H = h["layer"], h["head"]
        top_heads_marker_story.append({
            "rank_in_fro_diff": h["rank"],
            "layer": L,
            "head": H,
            "fro_delta": h["mean_delta"],
            "ocr2_attn_to_markers": float(ocr2_mean[L, H]),
            "base_attn_to_markers": float(base_mean[L, H]),
            "marker_attn_diff": float(marker_diff[L, H]),
        })

    # Heads that most INCREASED marker attention (different ranking than Frobenius)
    flat_diff = marker_diff.flatten()
    order = np.argsort(flat_diff)[::-1][:12]
    num_heads = marker_diff.shape[1]
    top_marker_heads = []
    for rank, idx in enumerate(order, start=1):
        L, H = int(idx // num_heads), int(idx % num_heads)
        top_marker_heads.append({
            "rank_in_marker_diff": rank,
            "layer": L,
            "head": H,
            "ocr2_attn_to_markers": float(ocr2_mean[L, H]),
            "base_attn_to_markers": float(base_mean[L, H]),
            "marker_attn_diff": float(marker_diff[L, H]),
            "fro_delta": float(np.array(ad["mean_delta"])[L, H]),
        })

    # Render figure: 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    # Left: layer × head marker-attention diff heatmap
    im = axes[0].imshow(marker_diff, aspect="auto", cmap="RdBu_r",
                         vmin=-np.abs(marker_diff).max(), vmax=np.abs(marker_diff).max(), origin="lower")
    axes[0].set_xlabel("head index")
    axes[0].set_ylabel("decoder layer")
    axes[0].set_title("A. OCR2 − base attention to structured-marker tokens")
    fig.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02)
    # Overlay top-12 Frobenius heads
    for h in top12:
        axes[0].scatter(h["head"], h["layer"], s=30, facecolors="none", edgecolors="yellow", linewidth=1.2)

    # Right: scatter of per-head frobenius-diff vs marker-attention-diff for top-12
    xs = [h["fro_delta"] for h in top_heads_marker_story]
    ys = [h["marker_attn_diff"] for h in top_heads_marker_story]
    labels = [f"L{h['layer']:02d}.H{h['head']:02d}" for h in top_heads_marker_story]
    axes[1].scatter(xs, ys, s=70, color="#7a0177", alpha=0.8, edgecolor="black")
    for x, y, lab in zip(xs, ys, labels):
        axes[1].annotate(lab, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.7)
    axes[1].set_xlabel(r"Frobenius $\delta[L,H]$")
    axes[1].set_ylabel(r"marker-attention diff (OCR2 − base)")
    axes[1].set_title("B. Top-12 Frobenius heads: are they also marker-attending?")
    axes[1].grid(alpha=0.3)

    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300)
    fig.savefig(FIG_PNG, dpi=200)
    plt.close(fig)

    summary = {
        "num_images": n_images,
        "n_marker_tokens_per_image_sample": int(marker_mask.sum()),
        "top_heads_frobenius_marker_story": top_heads_marker_story,
        "top_heads_marker_increase": top_marker_heads,
        "layer_head_marker_diff": marker_diff.tolist(),
        "max_marker_increase": {
            "value": float(marker_diff.max()),
            "layer": int(np.unravel_index(marker_diff.argmax(), marker_diff.shape)[0]),
            "head": int(np.unravel_index(marker_diff.argmax(), marker_diff.shape)[1]),
        },
        "max_marker_decrease": {
            "value": float(marker_diff.min()),
            "layer": int(np.unravel_index(marker_diff.argmin(), marker_diff.shape)[0]),
            "head": int(np.unravel_index(marker_diff.argmin(), marker_diff.shape)[1]),
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    ck_path = Path("docs/checkpoints/phase_2_7_complete.json")
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    ck_path.write_text(json.dumps({
        "phase": "2.7 marker attention",
        "status": "complete",
        "n_images": n_images,
        "fig_pdf": str(FIG_PDF),
        "max_increase_head": summary["max_marker_increase"],
        "max_decrease_head": summary["max_marker_decrease"],
        "top12_with_positive_marker_diff": sum(1 for h in top_heads_marker_story if h["marker_attn_diff"] > 0),
        "top12_with_negative_marker_diff": sum(1 for h in top_heads_marker_story if h["marker_attn_diff"] < 0),
    }, indent=2))

    print(f"\nwrote fig={FIG_PDF}  results={RESULTS_PATH}")
    print(f"\nmax marker-attn increase: L{summary['max_marker_increase']['layer']}.H{summary['max_marker_increase']['head']} = {summary['max_marker_increase']['value']:+.4f}")
    print(f"max marker-attn decrease: L{summary['max_marker_decrease']['layer']}.H{summary['max_marker_decrease']['head']} = {summary['max_marker_decrease']['value']:+.4f}")
    print(f"\ntop-12 Frobenius heads — marker-attention story:")
    for h in top_heads_marker_story:
        sign = "+" if h["marker_attn_diff"] >= 0 else ""
        print(f"  rank {h['rank_in_fro_diff']:2d}  L{h['layer']:02d}.H{h['head']:02d}  fro_δ={h['fro_delta']:.2f}  marker_diff={sign}{h['marker_attn_diff']:.4f}  ocr2={h['ocr2_attn_to_markers']:.4f}  base={h['base_attn_to_markers']:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
