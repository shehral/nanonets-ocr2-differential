"""per-category token-type decomposition at L35.

A natural question: does the 53/20/27 L35 split (pre-image / image /
post-image) hold uniformly across categories, or do image-heavy categories
(arxiv, handwritten) show a larger image-token share at the L35 peak?

If the share is roughly category-invariant, §4.1's token-type remark stays
as one line. If it varies noticeably, the bimodality from §4.2 may carry
through into the token-type split.

CPU-only; reuses cached this stage activations. No model load.

Output:
    code/results/token_type_per_category.json
    code/figures/fig1e_token_type_per_category.pdf
    docs/checkpoints/phase_2_14b_complete.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

ACT_ROOT = Path("code/activations")
RESULTS_PATH = Path("code/results/token_type_per_category.json")
FIG_PDF = Path("code/figures/fig1e_token_type_per_category.pdf")
FIG_PNG = Path("code/figures/fig1e_token_type_per_category.png")

TARGET_LAYER = 35

def per_image_l35_split(iid, cat):
    base = torch.load(ACT_ROOT / "base" / f"image_{iid}_{cat}.pt",
                      map_location="cpu", weights_only=False)
    ocr = torch.load(ACT_ROOT / "ocr2" / f"image_{iid}_{cat}.pt",
                     map_location="cpu", weights_only=False)
    span = base.get("image_token_span")
    if span is None:
        return None
    img_start, img_end = span
    hs_b = base["hidden_states"][TARGET_LAYER].squeeze(0).float()
    hs_o = ocr["hidden_states"][TARGET_LAYER].squeeze(0).float()
    T = min(hs_b.shape[0], hs_o.shape[0])
    diff = (hs_o[:T] - hs_b[:T]).norm(dim=-1).numpy()  # (T,)
    pre_mean = float(diff[:img_start].mean()) if img_start > 0 else 0.0
    img_mean = float(diff[img_start:img_end].mean()) if img_end > img_start else 0.0
    post_mean = float(diff[img_end:T].mean()) if img_end < T else 0.0
    total = pre_mean + img_mean + post_mean + 1e-12
    return {
        "image_id": iid, "category": cat,
        "pre_mean": pre_mean, "image_mean": img_mean, "post_mean": post_mean,
        "pre_share": pre_mean / total,
        "image_share": img_mean / total,
        "post_share": post_mean / total,
    }

def main():
    base_files = {p.name for p in (ACT_ROOT / "base").glob("*.pt")}
    ocr2_files = {p.name for p in (ACT_ROOT / "ocr2").glob("*.pt")}
    common = sorted(base_files & ocr2_files)

    per_image = []
    for fn in common:
        parts = fn.replace(".pt", "").split("_", 2)
        if len(parts) == 3 and parts[0] == "image":
            iid, cat = parts[1], parts[2]
            r = per_image_l35_split(iid, cat)
            if r is not None:
                per_image.append(r)

    print(f"[token-type-per-cat] {len(per_image)} images at L{TARGET_LAYER}", flush=True)

    by_cat = defaultdict(lambda: {"pre": [], "image": [], "post": [], "ids": []})
    for r in per_image:
        c = r["category"].split("_")[0]
        by_cat[c]["pre"].append(r["pre_share"])
        by_cat[c]["image"].append(r["image_share"])
        by_cat[c]["post"].append(r["post_share"])
        by_cat[c]["ids"].append(r["image_id"])

    cats = sorted(by_cat.keys())
    summary = {"target_layer": TARGET_LAYER, "n_images": len(per_image), "by_category": {}}
    for c in cats:
        summary["by_category"][c] = {
            "n": len(by_cat[c]["pre"]),
            "pre_share_mean": float(np.mean(by_cat[c]["pre"])),
            "image_share_mean": float(np.mean(by_cat[c]["image"])),
            "post_share_mean": float(np.mean(by_cat[c]["post"])),
            "pre_share_std": float(np.std(by_cat[c]["pre"])),
            "image_share_std": float(np.std(by_cat[c]["image"])),
            "post_share_std": float(np.std(by_cat[c]["post"])),
        }

    overall = {
        "pre": float(np.mean([r["pre_share"] for r in per_image])),
        "image": float(np.mean([r["image_share"] for r in per_image])),
        "post": float(np.mean([r["post_share"] for r in per_image])),
    }
    summary["overall_share_mean"] = overall

    # Figure: stacked bar per category
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    xs = np.arange(len(cats) + 1)
    pre_mean = [summary["by_category"][c]["pre_share_mean"] for c in cats] + [overall["pre"]]
    img_mean = [summary["by_category"][c]["image_share_mean"] for c in cats] + [overall["image"]]
    post_mean = [summary["by_category"][c]["post_share_mean"] for c in cats] + [overall["post"]]
    labels = cats + ["overall"]
    ax.bar(xs, pre_mean, label="pre-image instruction tokens", color="#2b8cbe")
    ax.bar(xs, img_mean, bottom=pre_mean, label="image tokens", color="#7a0177")
    ax.bar(xs, post_mean,
           bottom=[p + i for p, i in zip(pre_mean, img_mean)],
           label="post-image instruction/output tokens", color="#cd4a4a")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("share of L35 residual diff (per-token mean L2)")
    ax.set_title(f"Figure 1e — L{TARGET_LAYER} residual-diff token-type share, by category")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1)

    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({**summary, "per_image": per_image}, indent=2))

    ck_path = Path("docs/checkpoints/phase_2_14b_complete.json")
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    ck_path.write_text(json.dumps({
        "phase": "2.14b token-type decomposition per category",
        "status": "complete",
        "overall_shares": overall,
        "by_category_shares": {c: {k: v for k, v in summary["by_category"][c].items() if k.endswith("mean") or k == "n"} for c in cats},
        "fig_pdf": str(FIG_PDF),
    }, indent=2))

    print("\n[token-type-per-cat] SUMMARY — L35 token-type share by category (pre / image / post):")
    print(f"  {'category':<15}  {'n':>2}   {'pre':>7}  {'image':>7}  {'post':>7}")
    for c in cats:
        d = summary["by_category"][c]
        print(f"  {c:<15}  {d['n']:>2}   {d['pre_share_mean']:>7.2%}  {d['image_share_mean']:>7.2%}  {d['post_share_mean']:>7.2%}")
    d = overall
    print(f"  {'overall':<15}  {len(per_image):>2}   {d['pre']:>7.2%}  {d['image']:>7.2%}  {d['post']:>7.2%}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
