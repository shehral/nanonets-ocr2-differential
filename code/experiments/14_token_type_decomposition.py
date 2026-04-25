"""token-type decomposition of the residual diff (CPU).

A natural question: does the L35 residual-diff peak come from
image-token positions or text-token positions? We have the image_token_span
in every cached payload. Split the per-layer residual diff into three
regions — pre-image text (instruction), image tokens, post-image text
(continuation / structured-output markers in the prompt).

Output:
    code/results/token_type_decomposition.json
    code/figures/fig1d_token_type.pdf
    docs/checkpoints/phase_2_14_complete.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ACT_ROOT = Path("code/activations")
RESULTS_PATH = Path("code/results/token_type_decomposition.json")
FIG_PDF = Path("code/figures/fig1d_token_type.pdf")
FIG_PNG = Path("code/figures/fig1d_token_type.png")

def manifest_pairs() -> list[tuple[str, str]]:
    base_files = {p.name for p in (ACT_ROOT / "base").glob("*.pt")}
    ocr2_files = {p.name for p in (ACT_ROOT / "ocr2").glob("*.pt")}
    common = sorted(base_files & ocr2_files)
    pairs = []
    for fn in common:
        parts = fn.replace(".pt", "").split("_", 2)
        if len(parts) == 3 and parts[0] == "image":
            pairs.append((parts[1], parts[2]))
    return pairs

def per_layer_diff_by_region(iid: str, cat: str) -> dict[str, np.ndarray] | None:
    """For one image, split per-layer L2 diff into pre-image / image / post-image text regions."""
    base = torch.load(ACT_ROOT / "base" / f"image_{iid}_{cat}.pt",
                      map_location="cpu", weights_only=False)
    ocr = torch.load(ACT_ROOT / "ocr2" / f"image_{iid}_{cat}.pt",
                     map_location="cpu", weights_only=False)
    span = base.get("image_token_span")
    if span is None:
        return None
    img_start, img_end = span
    T = base["token_count"]
    hs_b = base["hidden_states"]
    hs_o = ocr["hidden_states"]
    n = len(hs_b)  # 37 layers

    pre_diff = np.zeros(n, dtype=np.float32)   # tokens 0..img_start-1
    img_diff = np.zeros(n, dtype=np.float32)   # tokens img_start..img_end-1
    post_diff = np.zeros(n, dtype=np.float32)  # tokens img_end..T-1
    for L in range(n):
        b = hs_b[L].squeeze(0).float()
        o = hs_o[L].squeeze(0).float()
        Tc = min(b.shape[0], o.shape[0])
        d = (o[:Tc] - b[:Tc]).norm(dim=-1).numpy()
        if img_start > 0:
            pre_diff[L] = float(d[:img_start].mean())
        img_diff[L] = float(d[img_start:img_end].mean()) if img_end > img_start else 0.0
        if img_end < Tc:
            post_diff[L] = float(d[img_end:Tc].mean())

    return {
        "pre_image": pre_diff,
        "image": img_diff,
        "post_image": post_diff,
        "span_img_start": img_start,
        "span_img_end": img_end,
        "token_count": T,
    }

def main() -> int:
    pairs = manifest_pairs()
    print(f"[token-type] {len(pairs)} images", flush=True)

    per_image = []
    for iid, cat in pairs:
        r = per_layer_diff_by_region(iid, cat)
        if r is not None:
            r["image_id"] = iid
            r["category"] = cat
            per_image.append(r)
    if not per_image:
        print("No images with image_token_span")
        return 1

    n_layers = len(per_image[0]["pre_image"])
    pre_stack = np.stack([r["pre_image"] for r in per_image])
    img_stack = np.stack([r["image"] for r in per_image])
    post_stack = np.stack([r["post_image"] for r in per_image])

    pre_mean = pre_stack.mean(axis=0)
    img_mean = img_stack.mean(axis=0)
    post_mean = post_stack.mean(axis=0)

    # Peak layer per region
    peak_pre = int(pre_mean.argmax())
    peak_img = int(img_mean.argmax())
    peak_post = int(post_mean.argmax())

    # Also report: at L35 (the overall peak), what's each region's value?
    L35_pre = float(pre_mean[35]) if n_layers > 35 else 0.0
    L35_img = float(img_mean[35]) if n_layers > 35 else 0.0
    L35_post = float(post_mean[35]) if n_layers > 35 else 0.0
    L35_total = L35_pre + L35_img + L35_post
    L35_share_img = L35_img / (L35_total + 1e-12)
    L35_share_post = L35_post / (L35_total + 1e-12)
    L35_share_pre = L35_pre / (L35_total + 1e-12)

    print(f"[token-type] peak layers — pre-image: L{peak_pre}, image: L{peak_img}, post-image text: L{peak_post}", flush=True)
    print(f"[token-type] at L35 — pre: {L35_pre:.2f}, image: {L35_img:.2f}, post: {L35_post:.2f}", flush=True)
    print(f"[token-type] L35 share — pre: {L35_share_pre:.2f}, image: {L35_share_img:.2f}, post: {L35_share_post:.2f}", flush=True)

    # Figure: 1 panel with 3 curves
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    xs = list(range(n_layers))
    ax.plot(xs, pre_mean, marker="o", ms=3, color="#2b8cbe", label=f"pre-image instruction tokens (peak L{peak_pre})")
    ax.plot(xs, img_mean, marker="s", ms=3, color="#7a0177", label=f"image tokens (peak L{peak_img})")
    ax.plot(xs, post_mean, marker="^", ms=3, color="#cd4a4a", label=f"post-image instruction tokens (peak L{peak_post})")
    ax.set_xlabel("layer (0 = embedding, 36 = post-norm output)")
    ax.set_ylabel(r"mean $\|h_\mathrm{ocr2}[L,t] - h_\mathrm{base}[L,t]\|_2$")
    ax.set_title(f"Figure 1d — Residual-diff decomposed by token type (N={len(per_image)})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_images": len(per_image),
        "pre_image_per_layer_mean": pre_mean.tolist(),
        "image_per_layer_mean": img_mean.tolist(),
        "post_image_per_layer_mean": post_mean.tolist(),
        "peak_layer_pre_image": peak_pre,
        "peak_layer_image": peak_img,
        "peak_layer_post_image": peak_post,
        "L35_values": {"pre": L35_pre, "image": L35_img, "post": L35_post, "total": L35_total},
        "L35_shares": {"pre": L35_share_pre, "image": L35_share_img, "post": L35_share_post},
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    ck_path = Path("docs/checkpoints/phase_2_14_complete.json")
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    ck_path.write_text(json.dumps({
        "phase": "2.14 token-type decomposition",
        "status": "complete",
        "peak_layers": {"pre": peak_pre, "image": peak_img, "post": peak_post},
        "L35_shares": {"pre": L35_share_pre, "image": L35_share_img, "post": L35_share_post},
        "fig_pdf": str(FIG_PDF),
    }, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
