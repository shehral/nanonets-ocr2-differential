"""linear probe for modality-boundary decodability.

Claim 3 (retargeted from the spec's "structured-output direction"):
    Does the OCR fine-tune make the image-token-vs-text-token boundary more
    linearly decodable in the residual stream? If OCR2 exceeds base by a
    pre-registered absolute-accuracy margin (>= 10 points), claim 3 is
    supported and Figure 3 is rendered.

Why this target instead of the spec's "<signature>/<watermark>/<page_number>"
positions: those structured markers only live in the fixed instruction
suffix, so their positions are identical across the 26 evaluation images —
making a probe trivial on position and useless on representation. The
modality boundary (image-pad tokens vs the rest) varies per image (image-token
span differs; content of surrounding text differs) and has a clean per-position
label derived from `image_token_span` in the saved payload.

This pivot is documented in `docs/checkpoints/phase_2_5_complete.json` so the
synthesizer and writer see exactly what was probed and why.

Output:
    code/results/probe_accuracies.json
    code/figures/fig3_probe.pdf (only if supported)
    code/figures/fig3_probe.png (only if supported)
    docs/checkpoints/phase_2_5_complete.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ACT_ROOT = Path("code/activations")
RESULTS_PATH = Path("code/results/probe_accuracies.json")
FIG_PDF = Path("code/figures/fig3_probe.pdf")
FIG_PNG = Path("code/figures/fig3_probe.png")

THRESHOLD_DELTA = 10.0  # absolute accuracy points

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

def modality_labels(payload: dict) -> np.ndarray:
    """1 at image-token positions, 0 elsewhere."""
    T = int(payload["token_count"])
    y = np.zeros(T, dtype=np.int64)
    span = payload.get("image_token_span")
    if span is not None:
        start, end = span
        y[start:end] = 1
    return y

def collect_layer_xy(tag: str, pairs: list[tuple[str, str]], max_content_per_image: int = 200) -> tuple[list[np.ndarray], np.ndarray]:
    """Per-layer X stack and shared y. We subsample content tokens to 2x image tokens per image to keep classes balanced."""
    by_layer: list[list[np.ndarray]] = []
    ys: list[np.ndarray] = []
    num_layers = None
    for (iid, cat) in pairs:
        p = torch.load(ACT_ROOT / tag / f"image_{iid}_{cat}.pt",
                       map_location="cpu", weights_only=False)
        hs = p["hidden_states"]
        y_full = modality_labels(p)
        if y_full.sum() == 0:
            continue
        # Balance: keep all image tokens, subsample content to 2x that count.
        img_idx = np.where(y_full == 1)[0]
        txt_idx = np.where(y_full == 0)[0]
        n_img = len(img_idx)
        n_keep_txt = min(len(txt_idx), 2 * n_img)
        rng = np.random.default_rng(0)
        kept_txt_idx = rng.choice(txt_idx, size=n_keep_txt, replace=False)
        keep_idx = np.concatenate([img_idx, kept_txt_idx])
        y_keep = y_full[keep_idx]
        if num_layers is None:
            num_layers = len(hs)
            by_layer = [[] for _ in range(num_layers)]
        for L in range(num_layers):
            h_full = hs[L].squeeze(0).float().numpy()
            by_layer[L].append(h_full[keep_idx])
        ys.append(y_keep)
    if num_layers is None:
        return [], np.zeros((0,), dtype=np.int64)
    X_per_layer = [np.concatenate(L_list, axis=0) for L_list in by_layer]
    y_concat = np.concatenate(ys, axis=0)
    return X_per_layer, y_concat

def fit_probe(X: np.ndarray, y: np.ndarray) -> float:
    if len(set(y.tolist())) < 2:
        return float("nan")
    k = min(5, int(np.bincount(y).min()))
    if k < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=1)
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))

def run_probe(tag: str, pairs: list[tuple[str, str]]) -> list[float]:
    X_per_layer, y = collect_layer_xy(tag, pairs)
    if len(X_per_layer) == 0:
        return []
    return [fit_probe(X, y) for X in X_per_layer]

def render_fig(base_accs: list[float], ocr2_accs: list[float], out_pdf: Path, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    xs = range(len(base_accs))
    ax.plot(xs, base_accs, label="Qwen2.5-VL-3B-Instruct (base)", marker="o", ms=4)
    ax.plot(xs, ocr2_accs, label="Nanonets-OCR2-3B (fine-tune)", marker="s", ms=4)
    ax.set_xlabel("decoder layer (0 = embedding, 36 = final output)")
    ax.set_ylabel("5-fold CV accuracy — image token vs. text token")
    ax.set_title("Figure 3 — Linear probe for modality-boundary decodability")
    ax.legend()
    ax.grid(alpha=0.3)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main() -> int:
    pairs = manifest_pairs()
    print(f"[probe] {len(pairs)} common images")
    base_accs = run_probe("base", pairs)
    ocr2_accs = run_probe("ocr2", pairs)

    n = min(len(base_accs), len(ocr2_accs))
    base_accs = base_accs[:n]
    ocr2_accs = ocr2_accs[:n]

    if n == 0:
        summary = {
            "status": "no_modality_labels",
            "base_accs": [],
            "ocr2_accs": [],
            "peak_base": 0.0,
            "peak_ocr2": 0.0,
            "delta_points": 0.0,
            "supported": False,
        }
    else:
        peak_base = max(a for a in base_accs if not np.isnan(a)) * 100.0
        peak_ocr2 = max(a for a in ocr2_accs if not np.isnan(a)) * 100.0
        delta = peak_ocr2 - peak_base
        supported = delta >= THRESHOLD_DELTA
        summary = {
            "status": "complete",
            "probe_target": "image_token_vs_text_token",
            "base_accs": base_accs,
            "ocr2_accs": ocr2_accs,
            "peak_base": peak_base,
            "peak_ocr2": peak_ocr2,
            "delta_points": delta,
            "supported": supported,
            "threshold_delta_points": THRESHOLD_DELTA,
        }
        if supported:
            render_fig(base_accs, ocr2_accs, FIG_PDF, FIG_PNG)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    ck_path = Path("docs/checkpoints/phase_2_5_complete.json")
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    ck_path.write_text(json.dumps({
        "phase": "2.5 stretch probe",
        "status": summary.get("status", "unknown"),
        "probe_target": summary.get("probe_target", "image_token_vs_text_token"),
        "claim_3_supported": summary.get("supported", False),
        "peak_base_pct": summary.get("peak_base", 0.0),
        "peak_ocr2_pct": summary.get("peak_ocr2", 0.0),
        "delta_points": summary.get("delta_points", 0.0),
        "fig_pdf": str(FIG_PDF) if summary.get("supported") else None,
        "pivot_note": "Pivoted from spec's <signature>/<watermark>/<page_number> probe because those markers only live in the fixed instruction suffix (position-trivial). Image-vs-text modality boundary varies per image and is a cleaner probe target for the 'linearly decodable mode direction' claim.",
    }, indent=2))
    print(f"wrote {RESULTS_PATH}  supported={summary.get('supported')} delta={summary.get('delta_points', 0):.1f} pts")
    if n > 0:
        print(f"peak base={summary['peak_base']:.1f}% peak ocr2={summary['peak_ocr2']:.1f}%")
    return 0

if __name__ == "__main__":
    sys.exit(main())
