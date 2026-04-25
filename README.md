# Where a VLM Fine-Tune Commits Depends on the Task

**Differential and causal localization of an OCR-specialized vision-language model.**

A mechanistic-interpretability study of [Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B) compared against its public base, [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct). The work asks: *when supervised fine-tuning turns a generalist vision-language model into an OCR specialist, what moves?* Nine forward-pass-only diagnostics on 26 document images, runnable on a 24 GB Apple Silicon laptop.

The full report is [`paper/main.pdf`](paper/main.pdf) — read that for the science. This README covers reproduction.

---

## Headline findings

1. **The representational edit concentrates in the upper decoder.** Per-layer residual-stream L2 diff peaks at L35 with Pearson r = 0.996 between two independently-trained fine-tunes of the same base — the localization is structural, not training-recipe specific.

2. **The causal threshold is bimodal, not unimodal.** Patching OCR2's residual into the base flips the top-1 prediction at layers 0–2 for content the base reads poorly (handwriting, LaTeX equations, non-Latin scripts) and at layers 9–15 for content the base already reads (English typewritten documents, forms, receipts). Median threshold L = 10. Under a corrected reference, full residual-stream patching produces the OCR2 prediction on 26/26 images (peak match 92.3% at L34); a random matched-magnitude control produces only 3.85% (24× differential).

3. **L11.H14 — the documented cross-modal router head — is not the mechanistic pivot.** Its attention pattern barely changes (rank 141 of 576 in the Frobenius diff), and head-level patching at L11 (router + two controls + full attention layer) produces zero OCR-directional flips across 26 images. The mid-cluster causal work must live in L11's MLP or in cross-layer dynamics, not in attention routing.

4. **Reverse-direction patching is asymmetric.** Injecting base's residual into OCR2's forward never produces base's prediction (0/26 across all layers). OCR2's upper decoder transforms base's residual into a configuration that yields neither model's natural output — consistent with fine-tuning as specialization of base's decoder geometry.

5. **MLPs carry 69% of the decoder weight change**; rank-16 SVD recovers 0.39–0.67 of each module's energy. A rank-16 LoRA on the three MLP projections captures roughly 24–34% of the total fine-tune edit — a starting point, not a full recipe.

6. **The vision encoder edit is fine-tune-specific, not general.** OCR2's ViT shows 1.60× more per-layer perturbation than its decoder. The parallel measurement on a second fine-tune (Nanonets-OCR-s) shows only 0.011 mean fractional change — an order of magnitude smaller, with cross-pair Pearson r = 0.51 against r = 0.996 on the decoder.

---

## Setup

**Hardware**: Apple Silicon with at least 24 GB unified memory (tested on M4 Pro, 24 GB). MPS backend. CUDA should also work with minor edits to `02_extract_activations.py` (replace `device_map="mps"` with `"cuda"`).

**Storage**: 22.5 GB of model weights (downloaded once on first forward pass) + 18 GB of cached activations once the pipeline runs.

```bash
# Clone
git clone https://github.com/shehral/nanonets-ocr2-differential.git
cd nanonets-ocr2-differential

# Python 3.12 recommended
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# HuggingFace login (some Nanonets checkpoints rate-limit anonymous pulls)
hf auth login
```

---

## Run the pipeline

Each stage writes to disk; later stages read those outputs. You can run the full pipeline with:

```bash
bash run_all.sh
```

Or stage-by-stage:

| Stage | Script | What it does | Output |
|---|---|---|---|
| 0 | `00_download_weights.py` | Pre-pulls base + OCR2 + OCR-s weights | `~/.cache/huggingface/` |
| 0 | `00_smoke_forward.py` | One-image forward sanity check | console |
| 1 | `01_build_eval_set.py` | Builds 26-image manifest across 6 categories | `data/processed/eval_set_manifest.json` |
| 2 | `02_extract_activations.py --model {base,ocr2,ocr_s}` | Forward pass per model, caches hidden states + attention to disk | `code/activations/{tag}/image_NN_cat.pt` (~18 GB total) |
| 3 | `03_residual_diff.py` | Per-layer L2 diff (Figure 1) | `code/figures/fig1_residual_diff.pdf`, `code/results/residual_diff.json` |
| 4 | `04_attention_diff.py` | 576-head Frobenius diff (Figure 2) | `code/figures/fig2_attention_diff.pdf`, `code/results/attention_diff.json` |
| 6 | `06_per_category_analysis.py` | Per-category residual curves (Figure 1b) | `code/figures/fig1b_per_category.pdf` |
| 7 | `07_marker_attention.py` | Attention to structured-marker tokens (Figure 2b) | `code/figures/fig2b_marker_attention.pdf` |
| 8 | `08c_residual_patching_full26.py` | Forward residual-stream patching (Figure 4) | `code/figures/fig4_causal_patching.pdf` |
| 9 | `09b_weight_diff_streaming.py` | Per-module weight diff + SVD (Figure 5 panels A, B) | `code/figures/fig5_weight_diff.pdf`, `code/results/weight_diff_analysis.json` |
| 9 | `09d_regenerate_fig5_rank16.py` | Rank-16 SVD heatmap (Figure 5 Panel C) | (overwrites Figure 5) |
| 11 | `11_vit_weight_diff.py` | ViT weight diff vs base (Appendix A) | `code/figures/fig5b_vit_weight_diff.pdf` |
| 11b | `11b_vit_weight_diff_ocr_s.py` | ViT triangulation OCR-s vs OCR2 (Appendix A) | `code/figures/fig5c_vit_weight_diff_triangulation.pdf` |
| 12 | `12_marker_bootstrap_ci.py` | Bootstrap 95% CIs on marker attention (Appendix C) | `code/figures/fig2c_marker_ci.pdf` |
| 13 | `13_per_block_contribution.py` | Per-block residual contribution diff | `code/results/per_block_contribution.json` |
| 14 | `14_token_type_decomposition.py` | L35 split by pre-image / image / post-image tokens | `code/results/token_type_decomposition.json` |
| 14b | `14b_token_type_per_category.py` | Per-category token-type split | `code/results/token_type_per_category.json` |
| 17 | `17_single_head_controls.py` | Head-level patching at L11 (H14 + H7 + H8 + full layer) | `code/figures/fig7_single_head_controls.pdf` |
| 18 | `18_reverse_patching_full26.py` | Reverse direction (base → OCR2) | `code/figures/fig8_reverse_patching.pdf` |
| 19 | `19_random_residual_control.py` | Random Gaussian-matched-norm patching control | `code/figures/fig9_random_control.pdf` |
| 20 | `20_forward_patching_with_match_ocr2.py` | Forward patching with match-OCR2-corrected tracking | `code/results/forward_patching_match_ocr2.json` |
| 21 | `21_ocr2_corrected_top1.py` | OCR2-corrected top-1 reference (lm_head bug workaround, see paper §3) | `code/results/ocr2_corrected_top1.json` |
| 22 | `22_recompute_match_corrected.py` | Match-OCR2 audit against corrected reference | `code/figures/fig4c_match_corrected.pdf` |

**Approximate compute time** on M4 Pro 24 GB:

- Stage 2 activations: ~30 minutes (3 models × 26 images, sequential model loading)
- Stage 8 residual patching: ~30 minutes (26 × 36-layer forwards on base with patching hooks)
- Stages 17 / 18 / 19 / 20: ~15–20 minutes each
- Other stages: minutes (CPU, reads cached activations)

Full pipeline end-to-end: about 2 hours.

---

## Compile the paper

```bash
cd paper
tectonic main.tex
```

`tectonic` is required (not `pdflatex` / `xelatex` / `latexmk`). Self-contained, auto-downloads packages, single-pass.

---

## Repository structure

```
nanonets-ocr2-differential/
├── README.md                 — this file
├── requirements.txt          — Python dependencies
├── run_all.sh                — orchestrate the full pipeline
├── paper/
│   ├── main.pdf              — final report (12 pages)
│   ├── main.tex              — LaTeX source
│   ├── references.bib        — 17 cited references
│   └── figures/              — all figures used in the paper
├── code/
│   ├── experiments/          — 24 Python scripts implementing the pipeline
│   ├── figures/              — generated figure PDFs (regenerable)
│   └── results/              — small JSON result files (regenerable)
└── data/processed/
    └── eval_set_manifest.json — 26-image evaluation manifest
```

---

## Reproducibility notes

A few load-bearing implementation details that took time to discover:

- **`attn_implementation="eager"` is required** when loading any model whose attention patterns you want to read. HuggingFace's default `sdpa` (and `flash_attention_2`) don't materialize the per-head softmax matrix — `output_attentions=True` returns empty tuples on the fast paths. Eager is slower but populates the tensors the head-wise analysis depends on.

- **`tie_word_embeddings` config bug in OCR2**: OCR2's shipped `config.json` sets the top-level `tie_word_embeddings: False` while the text-config flag is `True`, and no separate `lm_head.weight` is shipped. HuggingFace's loader initializes `lm_head` randomly. The corrected reference is `base.embed_tokens.T` applied to OCR2's cached final hidden state — base's tied embeddings are numerically identical to OCR2's. Paper §3 documents this; `21_ocr2_corrected_top1.py` implements it.

- **Sequential model loading**: 24 GB unified memory cannot hold three 3B VLMs simultaneously. Each script loads one model, runs forward passes, frees, then loads the next. `gc.collect()` and `torch.mps.empty_cache()` are required between loads.

- **Streaming weight diff** (`09b_weight_diff_streaming.py`): full fp32 weight differencing of two 3B models OOMs the system. Loading at bf16, extracting per-layer state dicts, freeing the model, and only then subtracting on CPU keeps peak memory under 16 GB.

- **Image-token budget pinned to ~256 tokens per image**: `min_pixels == max_pixels == 256 * 28 * 28` in the processor call ensures every image produces a consistent number of LLM image tokens, which is what makes element-wise activation diffs across models sensible.

---

## License

All rights reserved. See [`LICENSE`](LICENSE). Published for academic review and personal portfolio use; copying, modification, redistribution, or derivative works require explicit written permission.

---

## Citation

If you use this work, please cite the paper. A BibTeX entry is forthcoming with the workshop submission; for now:

```
Mohammad Ali Shehral. "Where a VLM Fine-Tune Commits Depends on the Task:
Differential and Causal Localization in Nanonets-OCR2-3B." 2026.
```
