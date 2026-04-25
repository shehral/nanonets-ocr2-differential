"""bootstrap CIs on per-head marker-attention diffs (review #3).

For each head (L, H), bootstrap-resample the 26 images and compute mean
marker-attention diff (OCR2 − base). Report 95% CI. Also compute a null
comparison: the same diff on a matched random non-marker token set.

Output:
    code/results/marker_attention_ci.json
    code/figures/fig2c_marker_ci.pdf
    docs/checkpoints/phase_2_12_complete.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ACT_ROOT = Path("code/activations")
RESULTS_PATH = Path("code/results/marker_attention_ci.json")
FIG_PDF = Path("code/figures/fig2c_marker_ci.pdf")
FIG_PNG = Path("code/figures/fig2c_marker_ci.png")

MARKER_SUBWORDS = {
    "img", "water", "mark", "page", "_number",
    "signature", "watermark", "page_number",
}
N_BOOTSTRAP = 1000
RNG_SEED = 0

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

def find_marker_positions(payload: dict) -> np.ndarray:
    toks = payload["token_strings"]
    T = len(toks)
    mask = np.zeros(T, dtype=np.int64)
    for i, t in enumerate(toks):
        tclean = t.replace("Ġ", "").replace("▁", "").strip().lower()
        if tclean in {m.lower() for m in MARKER_SUBWORDS}:
            mask[i] = 1
    return mask

def find_matched_nonmarker_positions(payload: dict, marker_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample len(markers) non-marker positions from the text region (post-image-token span)."""
    T = len(payload["token_strings"])
    span = payload.get("image_token_span")
    if span is None:
        text_start = 0
    else:
        text_start = span[1]
    candidates = np.where((marker_mask == 0) & (np.arange(T) >= text_start))[0]
    n_markers = int(marker_mask.sum())
    if len(candidates) < n_markers or n_markers == 0:
        return np.array([], dtype=np.int64)
    return rng.choice(candidates, size=n_markers, replace=False)

def attn_to_positions(attn: torch.Tensor, positions: np.ndarray) -> np.ndarray:
    """For a (H, T, T) attention tensor, return mean attention weight TO the given positions per head."""
    if len(positions) == 0:
        return np.zeros(attn.shape[0], dtype=np.float32)
    A = attn.float().numpy()  # (H, T, T)
    return A[:, :, positions].sum(axis=-1).mean(axis=-1).astype(np.float32)

def collect_per_image(tag: str, iid: str, cat: str, rng: np.random.Generator) -> tuple[np.ndarray | None, np.ndarray | None]:
    p = torch.load(ACT_ROOT / tag / f"image_{iid}_{cat}.pt",
                   map_location="cpu", weights_only=False)
    marker_mask = find_marker_positions(p)
    if marker_mask.sum() == 0:
        return None, None
    marker_idx = np.where(marker_mask == 1)[0]
    control_idx = find_matched_nonmarker_positions(p, marker_mask, rng)
    if len(control_idx) == 0:
        return None, None
    attn_list = p["attentions"]  # list of (1, H, T, T)
    num_layers = len(attn_list)
    num_heads = attn_list[0].shape[1]
    marker_per_head = np.zeros((num_layers, num_heads), dtype=np.float32)
    control_per_head = np.zeros((num_layers, num_heads), dtype=np.float32)
    for L in range(num_layers):
        A = attn_list[L].squeeze(0)
        marker_per_head[L] = attn_to_positions(A, marker_idx)
        control_per_head[L] = attn_to_positions(A, control_idx)
    return marker_per_head, control_per_head

def main() -> int:
    pairs = manifest_pairs()
    rng = np.random.default_rng(RNG_SEED)
    print(f"[marker-ci] {len(pairs)} image pairs", flush=True)

    # Per-image marker and control attention, for both models
    per_image_ocr2_marker = []
    per_image_base_marker = []
    per_image_ocr2_control = []
    per_image_base_control = []
    for iid, cat in pairs:
        o_m, o_c = collect_per_image("ocr2", iid, cat, rng)
        b_m, b_c = collect_per_image("base", iid, cat, rng)
        if o_m is None or b_m is None:
            continue
        per_image_ocr2_marker.append(o_m)
        per_image_ocr2_control.append(o_c)
        per_image_base_marker.append(b_m)
        per_image_base_control.append(b_c)

    N = len(per_image_ocr2_marker)
    ocr_m = np.stack(per_image_ocr2_marker)
    ocr_c = np.stack(per_image_ocr2_control)
    base_m = np.stack(per_image_base_marker)
    base_c = np.stack(per_image_base_control)
    num_layers, num_heads = ocr_m.shape[1], ocr_m.shape[2]
    print(f"[marker-ci] N={N} images, {num_layers} layers × {num_heads} heads", flush=True)

    # Per-image diff (OCR2 - base) for markers and controls
    diff_marker = ocr_m - base_m  # (N, L, H)
    diff_control = ocr_c - base_c

    # Mean and 95% bootstrap CI per (L, H) for both
    mean_marker = diff_marker.mean(axis=0)
    mean_control = diff_control.mean(axis=0)

    lo_marker = np.zeros((num_layers, num_heads), dtype=np.float32)
    hi_marker = np.zeros((num_layers, num_heads), dtype=np.float32)
    lo_control = np.zeros((num_layers, num_heads), dtype=np.float32)
    hi_control = np.zeros((num_layers, num_heads), dtype=np.float32)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, N, N)
        bm = diff_marker[idx].mean(axis=0)
        bc = diff_control[idx].mean(axis=0)
        # Running quantiles (store all samples — memory 1000 * 36 * 16 * 4 bytes = 2.3 MB)
        if b == 0:
            boots_marker = np.zeros((N_BOOTSTRAP, num_layers, num_heads), dtype=np.float32)
            boots_control = np.zeros((N_BOOTSTRAP, num_layers, num_heads), dtype=np.float32)
        boots_marker[b] = bm
        boots_control[b] = bc
    lo_marker = np.percentile(boots_marker, 2.5, axis=0)
    hi_marker = np.percentile(boots_marker, 97.5, axis=0)
    lo_control = np.percentile(boots_control, 2.5, axis=0)
    hi_control = np.percentile(boots_control, 97.5, axis=0)
    significant_marker = (lo_marker > 0) | (hi_marker < 0)

    # Cross-reference with top-12 Frobenius heads
    ad = json.loads(Path("code/results/attention_diff.json").read_text())
    top12 = ad["top_heads"]

    top12_with_ci = []
    for h in top12:
        L, H = h["layer"], h["head"]
        top12_with_ci.append({
            "rank": h["rank"],
            "layer": L,
            "head": H,
            "fro_delta": h["mean_delta"],
            "marker_diff_mean": float(mean_marker[L, H]),
            "marker_diff_ci_low": float(lo_marker[L, H]),
            "marker_diff_ci_high": float(hi_marker[L, H]),
            "control_diff_mean": float(mean_control[L, H]),
            "control_diff_ci_low": float(lo_control[L, H]),
            "control_diff_ci_high": float(hi_control[L, H]),
            "marker_significant": bool(significant_marker[L, H]),
            "control_significant": bool((lo_control[L, H] > 0) | (hi_control[L, H] < 0)),
            "marker_bigger_than_control": abs(float(mean_marker[L, H])) > abs(float(mean_control[L, H])),
        })

    # Figure: top-12 head bar chart with marker + control CI overlay
    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    xs = np.arange(len(top12_with_ci))
    width = 0.4
    marker_means = [h["marker_diff_mean"] for h in top12_with_ci]
    marker_errs_lo = [h["marker_diff_mean"] - h["marker_diff_ci_low"] for h in top12_with_ci]
    marker_errs_hi = [h["marker_diff_ci_high"] - h["marker_diff_mean"] for h in top12_with_ci]
    control_means = [h["control_diff_mean"] for h in top12_with_ci]
    control_errs_lo = [h["control_diff_mean"] - h["control_diff_ci_low"] for h in top12_with_ci]
    control_errs_hi = [h["control_diff_ci_high"] - h["control_diff_mean"] for h in top12_with_ci]
    ax.bar(xs - width/2, marker_means, width,
           yerr=[marker_errs_lo, marker_errs_hi],
           color="#7a0177", label="marker positions", capsize=3)
    ax.bar(xs + width/2, control_means, width,
           yerr=[control_errs_lo, control_errs_hi],
           color="#969696", label="matched non-marker control", capsize=3)
    ax.axhline(0, color="black", linewidth=0.6)
    labels = [f"L{h['layer']:02d}.H{h['head']:02d}" for h in top12_with_ci]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("OCR2 − base attention weight")
    ax.set_title(f"Top-12 Frobenius heads — marker attention vs non-marker control\nBootstrap 95% CI over N={N} images, {N_BOOTSTRAP} resamples")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_images": N,
        "n_bootstrap": N_BOOTSTRAP,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "top12_with_ci": top12_with_ci,
        "n_top12_significant_on_markers": int(sum(1 for h in top12_with_ci if h["marker_significant"])),
        "n_top12_significant_on_control": int(sum(1 for h in top12_with_ci if h["control_significant"])),
        "n_top12_marker_abs_bigger": int(sum(1 for h in top12_with_ci if h["marker_bigger_than_control"])),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    ck_path = Path("docs/checkpoints/phase_2_12_complete.json")
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    ck_path.write_text(json.dumps({
        "phase": "2.12 marker-attention bootstrap CIs",
        "status": "complete",
        "fig_pdf": str(FIG_PDF),
        "n_images": N,
        "n_bootstrap": N_BOOTSTRAP,
        "top12_marker_significant": summary["n_top12_significant_on_markers"],
        "top12_control_significant": summary["n_top12_significant_on_control"],
        "top12_marker_dominates": summary["n_top12_marker_abs_bigger"],
    }, indent=2))

    print(f"\n[marker-ci] top-12 with marker-significant CI: {summary['n_top12_significant_on_markers']}/12")
    print(f"[marker-ci] top-12 with control-significant CI: {summary['n_top12_significant_on_control']}/12")
    print(f"[marker-ci] top-12 where |marker diff| > |control diff|: {summary['n_top12_marker_abs_bigger']}/12")
    return 0

if __name__ == "__main__":
    sys.exit(main())
