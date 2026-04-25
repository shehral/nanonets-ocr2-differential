"""head-wise attention-pattern diff. Figure 2 data + render.

Primary claim 2: a small set of attention heads in OCR2-3B specialize in
structured-output-token emission compared to the base. Three panels in
Figure 2:

    A. 36 x 16 heatmap of mean δ[L, H] — shows global distribution of
       attention-pattern change across all 576 heads
    B. bar chart of top-12 heads ordered by δ, colored by router/promoter
       fallback label on OCR2
    C. 4 × 3 grid: top-4 heads (one per column) × 3 representative images
       across different categories (docvqa, receipt, form); each cell is
       side-by-side OCR2 vs base attention

Output:
    code/results/attention_diff.json
    code/figures/fig2_attention_diff.pdf
    code/figures/fig2_attention_diff.png
    docs/checkpoints/phase_2_4_complete.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ACT_ROOT = Path("code/activations")
RESULTS_PATH = Path("code/results/attention_diff.json")
FIG_PDF = Path("code/figures/fig2_attention_diff.pdf")
FIG_PNG = Path("code/figures/fig2_attention_diff.png")

# Router/promoter fallback thresholds (plan §2.4 fallback; real taxonomy in
# docs/orient/head_taxonomy.md requires conflicting-modality ablation runs
# that are explicitly out of scope for the overnight chain).
ROUTER_ENTROPY_THRESHOLD = 3.0
PROMOTER_MAX_ATTN_THRESHOLD = 0.5

def manifest_pairs() -> list[tuple[str, str]]:
    base_files = {p.name for p in (ACT_ROOT / "base").glob("*.pt")}
    ocr2_files = {p.name for p in (ACT_ROOT / "ocr2").glob("*.pt")}
    common = sorted(base_files & ocr2_files)
    pairs = []
    for fn in common:
        stem = fn.replace(".pt", "")
        parts = stem.split("_", 2)
        if len(parts) == 3 and parts[0] == "image":
            pairs.append((parts[1], parts[2]))
    return pairs

def load_payload(tag: str, image_id: str, category: str) -> dict:
    return torch.load(ACT_ROOT / tag / f"image_{image_id}_{category}.pt",
                      map_location="cpu", weights_only=False)

def compute_head_diff(pairs: list[tuple[str, str]]) -> tuple[np.ndarray, int, int]:
    accum = None
    counts = None
    num_layers = num_heads = 0
    for (iid, cat) in pairs:
        a = load_payload("ocr2", iid, cat)
        b = load_payload("base", iid, cat)
        a_attn = a["attentions"]
        b_attn = b["attentions"]
        if len(a_attn) != len(b_attn):
            continue
        num_layers = len(a_attn)
        num_heads = a_attn[0].shape[1]
        if a_attn[0].shape[-1] != b_attn[0].shape[-1]:
            continue
        if accum is None:
            accum = np.zeros((num_layers, num_heads), dtype=np.float64)
            counts = np.zeros((num_layers, num_heads), dtype=np.int64)
        for L in range(num_layers):
            A = a_attn[L].squeeze(0).float()
            B = b_attn[L].squeeze(0).float()
            diff = (A - B).reshape(num_heads, -1).norm(dim=-1).numpy()
            accum[L] += diff
            counts[L] += 1
    if accum is None:
        return np.zeros((0, 0)), 0, 0
    mean_delta = accum / np.clip(counts, 1, None)
    return mean_delta.astype(np.float32), num_layers, num_heads

def head_entropy(attn: torch.Tensor) -> float:
    p = attn.clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=-1)
    return float(ent.mean().item())

def head_max_attn(attn: torch.Tensor) -> float:
    return float(attn.max(dim=-1).values.mean().item())

def classify_head(attn: torch.Tensor) -> str:
    ent = head_entropy(attn)
    mx = head_max_attn(attn)
    if ent > ROUTER_ENTROPY_THRESHOLD:
        return "router"
    if mx > PROMOTER_MAX_ATTN_THRESHOLD:
        return "promoter"
    return "neither"

def pick_panel_images(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """One image per category, three categories total, in a deterministic order.

    The spec's top-heads figure needs visual variety: a DocVQA page (mixed
    layout with headers and tables), a receipt (structured-output candidate
    with prices and page-number tags), and a form (key-value layout).
    """
    by_cat: dict[str, tuple[str, str]] = {}
    for iid, cat in pairs:
        base_cat = cat.split("_")[0] if "_" in cat else cat
        if base_cat not in by_cat:
            by_cat[base_cat] = (iid, cat)
    preferred_order = ["docvqa", "receipt", "form", "handwritten", "arxiv", "multilingual"]
    chosen = []
    for c in preferred_order:
        if c in by_cat:
            chosen.append(by_cat[c])
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        chosen += [p for p in pairs if p not in chosen][: 3 - len(chosen)]
    return chosen[:3]

def render_figure(mean_delta: np.ndarray, pairs: list[tuple[str, str]], top_k: int, out_pdf: Path, out_png: Path) -> list[dict]:
    num_layers, num_heads = mean_delta.shape

    # Sort heads by delta
    flat = mean_delta.flatten()
    order = np.argsort(flat)[::-1]
    top_idx = order[:top_k]
    top_pairs = [(int(i // num_heads), int(i % num_heads)) for i in top_idx]

    panel_imgs = pick_panel_images(pairs)

    fig = plt.figure(figsize=(14, 13), constrained_layout=False)
    gs = GridSpec(3, 8, figure=fig, height_ratios=[3, 2, 4.5], hspace=0.5, wspace=0.35)

    # Panel A: layer × head heatmap of mean delta
    ax_a = fig.add_subplot(gs[0, :4])
    im = ax_a.imshow(mean_delta, aspect="auto", cmap="magma", origin="lower")
    ax_a.set_xlabel("head index (0..15)")
    ax_a.set_ylabel("decoder layer (0..35)")
    ax_a.set_title("A. Per-head attention-pattern divergence\n" r"$\delta[L,H] = \|A_\mathrm{ocr2}[L,H] - A_\mathrm{base}[L,H]\|_F$" " (mean over 26 images)", fontsize=10)
    fig.colorbar(im, ax=ax_a, fraction=0.04, pad=0.02)
    # Annotate top-K head positions
    for rank, (L, H) in enumerate(top_pairs[:6], start=1):
        ax_a.annotate(f"{rank}", xy=(H, L), xytext=(H + 0.1, L + 0.1),
                      fontsize=7, color="white", weight="bold")

    # Panel B: bar chart of top-K heads
    ax_b = fig.add_subplot(gs[0, 4:])
    xs = np.arange(top_k)
    vals = [float(mean_delta[L, H]) for L, H in top_pairs]
    labels = [f"L{L:02d}.H{H:02d}" for L, H in top_pairs]
    # Classify using the first panel image so colors reflect a specific instance,
    # but we'll show a consistent color: purple=promoter, teal=router, gray=neither
    ref_img = panel_imgs[0]
    ref_payload = load_payload("ocr2", ref_img[0], ref_img[1])
    colors = []
    labels_class = []
    for L, H in top_pairs:
        cls = classify_head(ref_payload["attentions"][L].squeeze(0)[H].float())
        labels_class.append(cls)
        colors.append({"router": "#2b8cbe", "promoter": "#7a0177", "neither": "#969696"}[cls])
    bars = ax_b.bar(xs, vals, color=colors)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_b.set_ylabel(r"$\delta[L,H]$")
    ax_b.set_title("B. Top-12 heads by attention-pattern divergence", fontsize=10)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#7a0177", label="promoter (threshold on max-attn)"),
        plt.Rectangle((0, 0), 1, 1, color="#2b8cbe", label="router (threshold on entropy)"),
        plt.Rectangle((0, 0), 1, 1, color="#969696", label="neither"),
    ]
    ax_b.legend(handles=legend_handles, fontsize=7, loc="upper right")

    # Panel C: top-4 heads × 3 images, side-by-side OCR2 vs base
    top4 = top_pairs[:4]
    payloads_ocr2 = {img: load_payload("ocr2", *img) for img in panel_imgs}
    payloads_base = {img: load_payload("base", *img) for img in panel_imgs}

    head_meta = []
    subplot_gs = GridSpec(3, 8, figure=fig, hspace=0.55, wspace=0.1,
                          top=0.40, bottom=0.02)
    for row_idx, img_key in enumerate(panel_imgs):
        a_p = payloads_ocr2[img_key]
        b_p = payloads_base[img_key]
        for col_idx, (L, H) in enumerate(top4):
            A_ocr2 = a_p["attentions"][L].squeeze(0)[H].float()
            A_base = b_p["attentions"][L].squeeze(0)[H].float()
            vmax = max(A_ocr2.max().item(), A_base.max().item())
            ax_l = fig.add_subplot(subplot_gs[row_idx, col_idx * 2])
            ax_r = fig.add_subplot(subplot_gs[row_idx, col_idx * 2 + 1])
            ax_l.imshow(A_ocr2.numpy(), cmap="magma", vmin=0, vmax=vmax, aspect="auto")
            ax_r.imshow(A_base.numpy(), cmap="magma", vmin=0, vmax=vmax, aspect="auto")
            ax_l.set_xticks([])
            ax_l.set_yticks([])
            ax_r.set_xticks([])
            ax_r.set_yticks([])
            if row_idx == 0:
                ax_l.set_title(f"L{L:02d}.H{H:02d}\n" f"ocr2", fontsize=7)
                ax_r.set_title("base", fontsize=7)
            else:
                ax_l.set_title("ocr2", fontsize=6)
                ax_r.set_title("base", fontsize=6)
            if col_idx == 0:
                ax_l.set_ylabel(f"{img_key[0]} {img_key[1]}", fontsize=7)

    for rank, (L, H) in enumerate(top_pairs, start=1):
        # Use each image separately for label if it appears in panel_imgs
        for img in panel_imgs:
            payload_ocr = payloads_ocr2[img]
            cls = classify_head(payload_ocr["attentions"][L].squeeze(0)[H].float())
        head_meta.append({
            "rank": rank,
            "layer": L,
            "head": H,
            "mean_delta": float(mean_delta[L, H]),
            "label_ocr2": labels_class[rank - 1] if rank - 1 < len(labels_class) else "n/a",
        })

    fig.suptitle("Figure 2 — Attention-pattern divergence between Nanonets-OCR2-3B and its Qwen2.5-VL-3B base", fontsize=12, y=0.995)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return head_meta

def main() -> int:
    pairs = manifest_pairs()
    if not pairs:
        print("No common activation files for ocr2+base — run this stage first")
        return 1
    print(f"[attention_diff] {len(pairs)} common images")

    mean_delta, num_layers, num_heads = compute_head_diff(pairs)
    if mean_delta.size == 0:
        print("No usable pairs (token-count mismatches?). Abort.")
        return 1

    top_k = 12
    head_meta = render_figure(mean_delta, pairs, top_k, FIG_PDF, FIG_PNG)

    router = sum(1 for h in head_meta if h["label_ocr2"] == "router")
    promoter = sum(1 for h in head_meta if h["label_ocr2"] == "promoter")
    neither = sum(1 for h in head_meta if h["label_ocr2"] == "neither")

    # Layer-concentration summary: how many of top-K are in late (>= 30) layers?
    late = sum(1 for h in head_meta if h["layer"] >= 30)
    early = sum(1 for h in head_meta if h["layer"] <= 2)

    summary = {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "total_heads_swept": num_layers * num_heads,
        "num_images": len(pairs),
        "mean_delta": mean_delta.tolist(),
        "top_heads": head_meta,
        "router_count": router,
        "promoter_count": promoter,
        "neither_count": neither,
        "late_layer_count_top12": late,
        "early_layer_count_top12": early,
        "threshold_router_entropy": ROUTER_ENTROPY_THRESHOLD,
        "threshold_promoter_max_attn": PROMOTER_MAX_ATTN_THRESHOLD,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    ck_path = Path("docs/checkpoints/phase_2_4_complete.json")
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    ck_path.write_text(json.dumps({
        "phase": "2.4 attention diff",
        "status": "complete",
        "fig_pdf": str(FIG_PDF),
        "fig_png": str(FIG_PNG),
        "top_heads_list": [{"layer": h["layer"], "head": h["head"], "delta": h["mean_delta"]} for h in head_meta],
        "router_count": router,
        "promoter_count": promoter,
        "neither_count": neither,
        "late_layer_count": late,
        "early_layer_count": early,
    }, indent=2))
    print(f"wrote fig={FIG_PDF}  results={RESULTS_PATH}  checkpoint={ck_path}")
    print(f"top head L{head_meta[0]['layer']}.H{head_meta[0]['head']} δ={head_meta[0]['mean_delta']:.3f}")
    print(f"top-12 distribution: router={router}  promoter={promoter}  neither={neither}  late(L≥30)={late}  early(L≤2)={early}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
