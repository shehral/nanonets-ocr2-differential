#!/usr/bin/env bash
# Run the full diagnostic pipeline end-to-end.
# Total wall-clock on M4 Pro 24GB: ~2 hours.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

run() {
    echo "▶  $1"
    python3 "code/experiments/$1"
}

# Stage 0 — pre-pull weights + smoke test
run 00_download_weights.py
run 00_smoke_forward.py

# Stage 1 — build evaluation set
run 01_build_eval_set.py

# Stage 2 — extract activations from each model (sequential)
python3 code/experiments/02_extract_activations.py --model base
python3 code/experiments/02_extract_activations.py --model ocr2
python3 code/experiments/02_extract_activations.py --model ocr_s

# Stage 3 — correlational diagnostics (CPU on cached activations)
run 03_residual_diff.py
run 04_attention_diff.py
run 06_per_category_analysis.py
run 07_marker_attention.py
run 12_marker_bootstrap_ci.py
run 13_per_block_contribution.py
run 14_token_type_decomposition.py
run 14b_token_type_per_category.py

# Stage 4 — weight-space diagnostics (streams bf16 weights)
run 09b_weight_diff_streaming.py
run 09d_regenerate_fig5_rank16.py
run 11_vit_weight_diff.py
run 11b_vit_weight_diff_ocr_s.py

# Stage 5 — causal interventions (loads model on MPS)
run 21_ocr2_corrected_top1.py
run 08c_residual_patching_full26.py
run 17_single_head_controls.py
run 18_reverse_patching_full26.py
run 19_random_residual_control.py
run 20_forward_patching_with_match_ocr2.py
run 22_recompute_match_corrected.py

# Stage 6 — compile paper
( cd paper && tectonic main.tex )

echo ""
echo "✓ pipeline complete. Output:"
echo "    paper/main.pdf"
echo "    code/figures/*.pdf"
echo "    code/results/*.json"
