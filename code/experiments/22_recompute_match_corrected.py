"""recompute match-OCR2-corrected rates for phases 2.17 and 2.20.

Once 2.17 and 2.20 have stored patched_top1 per intervention/layer, this
script reads those JSONs, reads ocr2_corrected_top1.json, and reports
match-rate curves against the corrected reference.

Key question for §4.2: at L=35, does patched base's top-1 equal OCR2's
corrected top-1 for all 26 images? If yes, the forward patching is
genuinely OCR-directional at the late layers, and the §4.2 threshold is
the earliest layer at which the base's downstream can carry OCR2's
residual to OCR2's own prediction.

Output:
    code/results/match_corrected_audit.json
    code/figures/fig4c_match_corrected.pdf
    docs/checkpoints/phase_2_22_complete.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def main():
    corrected = json.load(open("code/results/ocr2_corrected_top1.json"))
    ocr2_true = {r["image_id"]: r["corrected_top1_from_hs36_direct"] for r in corrected["per_image"]}

    out = {"ocr2_corrected_reference": len(ocr2_true)}

    # this stage recomputation (head-level patches)
    try:
        sh = json.load(open("code/results/single_head_controls.json"))
        for intervention_key in ["patched_h14", "patched_h7", "patched_h8", "patched_all_heads"]:
            n_match = sum(1 for r in sh["per_image"] if r[intervention_key] == ocr2_true[r["image_id"]])
            out[f"phase_2_17_{intervention_key}_match_corrected"] = {"n_match": n_match, "n": 26, "rate": n_match / 26}
    except Exception as e:
        out["phase_2_17_error"] = str(e)

    # this stage recomputation (layer-level patches)
    fwd_path = Path("code/results/forward_patching_match_ocr2.json")
    if not fwd_path.exists():
        out["phase_2_20_status"] = "not_yet_written"
        print("[match-audit] this stage not yet complete; run again later", flush=True)
        Path("code/results/match_corrected_audit.json").write_text(json.dumps(out, indent=2))
        return 0

    fwd = json.load(open(fwd_path))
    num_layers = fwd["num_layers"]

    match_rate_per_layer = []
    first_match_L = []  # earliest L at which patched_top1 == ocr2_true
    late_match_peak = 0
    for L in range(num_layers):
        rates = []
        for r in fwd["per_image"]:
            p_top1 = r["per_layer"][L]["patched_top1"]
            oc_true = ocr2_true[r["image_id"]]
            rates.append(int(p_top1 == oc_true))
        match_rate_per_layer.append(float(np.mean(rates)))

    for r in fwd["per_image"]:
        first = None
        for L in range(num_layers):
            if r["per_layer"][L]["patched_top1"] == ocr2_true[r["image_id"]]:
                first = L
                break
        first_match_L.append(first)

    late_match_peak = max(match_rate_per_layer)
    late_L35_rate = match_rate_per_layer[35] if len(match_rate_per_layer) > 35 else None

    out["phase_2_20_match_corrected_per_layer"] = match_rate_per_layer
    out["phase_2_20_peak_match_rate"] = late_match_peak
    out["phase_2_20_L35_match_rate"] = late_L35_rate
    out["phase_2_20_first_match_L"] = first_match_L
    out["phase_2_20_n_ever_match"] = sum(1 for x in first_match_L if x is not None)
    out["phase_2_20_first_match_median"] = int(np.median([x for x in first_match_L if x is not None])) if any(x is not None for x in first_match_L) else None

    # Figure: match-corrected rate curve + original flip-vs-base overlay
    flip_rate = fwd["fraction_flipped_vs_base_per_layer"]
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    xs = list(range(num_layers))
    ax.plot(xs, flip_rate, marker="o", ms=3, color="#cd4a4a", label="fraction flipped vs base (§4.2 definition)")
    ax.plot(xs, match_rate_per_layer, marker="s", ms=3, color="#2b8cbe", label="fraction matching OCR2-corrected top-1")
    ax.set_xlabel("patching layer L")
    ax.set_ylabel("fraction of 26 images")
    ax.set_title(f"Forward patching: flip-vs-base (§4.2) vs match-OCR2-corrected\nPeak match rate {late_match_peak:.1%}, L35 match rate {late_L35_rate:.1%}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    FIG_PDF = Path("code/figures/fig4c_match_corrected.pdf")
    FIG_PNG = Path("code/figures/fig4c_match_corrected.png")
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    out["figure_pdf"] = str(FIG_PDF)

    Path("code/results/match_corrected_audit.json").write_text(json.dumps(out, indent=2))

    ck = Path("docs/checkpoints/phase_2_22_complete.json")
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps({
        "phase": "2.22 match-OCR2-corrected audit",
        "status": "complete",
        "peak_match_rate": late_match_peak,
        "L35_match_rate": late_L35_rate,
        "n_ever_match": out["phase_2_20_n_ever_match"],
    }, indent=2))

    print(f"[match-audit] peak match-OCR2-corrected rate: {late_match_peak:.2%}")
    print(f"[match-audit] L35 match rate (should be ~100%): {late_L35_rate:.2%}")
    print(f"[match-audit] median first-match layer: {out['phase_2_20_first_match_median']}")
    print(f"[match-audit] {out['phase_2_20_n_ever_match']}/26 images ever match at some L")
    return 0

if __name__ == "__main__":
    sys.exit(main())
