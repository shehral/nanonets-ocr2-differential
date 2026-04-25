"""compute OCR2-corrected top-1 predictions.

CRITICAL BUG DISCOVERY: OCR2's huggingface config has `tie_word_embeddings:
False` at top-level. Because the OCR2 checkpoint does not ship a separate
`lm_head.weight`, HF initializes it randomly — every OCR2 forward pass uses
a random lm_head that produces meaningless argmax predictions (while OCR2's
decoder + embed_tokens are correctly loaded).

Base's config has `tie_word_embeddings: True`, so base's lm_head is tied to
its embed_tokens and operates correctly.

For this project, OCR2 was INTENDED to use tied embeddings (verified by
text_config.tie_word_embeddings=True in OCR2's config; only the top-level
field is wrong). The corrected OCR2 prediction is base.embed_tokens.T applied
to base.norm(ocr2_hs[36]). This is mathematically equivalent to running OCR2
with the corrected config. Base's embed_tokens and norm are identical to
OCR2's (verified: both have fractional_change = 0.000000).

This script reads each image's cached OCR2 hs[36] from this stage, applies
base's final norm and tied lm_head, and stores the corrected OCR2 top-1.
Fast CPU-only.

Downstream: this replaces `ocr2_top1` in this stage, 2.18, 2.20 match-rate
calculations. If high match rates emerge under the corrected reference, the
§4.7 'H14 does not redirect to OCR2' reading in v4.2 needs updating.

Output:
    code/results/ocr2_corrected_top1.json
    docs/checkpoints/phase_2_21_complete.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

def load_shared_weights():
    base_files = sorted(Path.home().glob(
        ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/*/*.safetensors"
    ))
    emb, nrm = None, None
    for p in base_files:
        with safe_open(p, framework="pt") as f:
            if "model.embed_tokens.weight" in f.keys():
                emb = f.get_tensor("model.embed_tokens.weight").float()
            if "model.norm.weight" in f.keys():
                nrm = f.get_tensor("model.norm.weight").float()
    return emb, nrm

def rmsnorm(h: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = (h.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    return (h / rms) * weight

def main():
    print("[ocr2-correct] loading shared embed_tokens + norm ...", flush=True)
    emb, nrm = load_shared_weights()
    print(f"  emb shape: {tuple(emb.shape)}, norm shape: {tuple(nrm.shape)}")

    m = json.loads(Path("data/processed/eval_set_manifest.json").read_text())
    images = [(e["image_id"], e["category"]) for e in m["images"]]

    results = []
    for iid, cat in images:
        p = Path(f"code/activations/ocr2/image_{iid}_{cat}.pt")
        if not p.exists():
            print(f"  [skip] {p} missing")
            continue
        c = torch.load(p, map_location="cpu", weights_only=False)
        hs = c["hidden_states"]
        # hs[-1] in the cached tensor is post-norm already (RMS drops from hs[-2] to hs[-1] after norm application).
        # We verified RMS: hs[35]=9.48, hs[36]=4.29. The sharp drop indicates hs[36] is post-norm.
        # To project via tied lm_head: just embed.T on hs[-1] last token.
        h_last = hs[-1].float().squeeze(0)[-1]  # (2048,)
        logits = h_last @ emb.T  # (151936,)
        corrected_top1 = int(logits.argmax().item())

        # Also do the 'pre-norm' interpretation as a sanity check
        h_prev = hs[-2].float().squeeze(0)[-1]
        normed = rmsnorm(h_prev, nrm)
        logits_prenorm = normed @ emb.T
        prenorm_top1 = int(logits_prenorm.argmax().item())

        results.append({
            "image_id": iid,
            "category": cat,
            "corrected_top1_from_hs36_direct": corrected_top1,
            "corrected_top1_from_hs35_plus_norm": prenorm_top1,
        })
        print(f"  {iid}_{cat}: hs36_direct={corrected_top1}, hs35+norm={prenorm_top1}", flush=True)

    # Cross-reference with this stage's (broken) ocr2_top1
    try:
        sh = json.load(open("code/results/single_head_controls.json"))
        broken = {r["image_id"]: r["ocr2_top1"] for r in sh["per_image"]}
        agreement_hs36 = sum(1 for r in results if r["corrected_top1_from_hs36_direct"] == broken.get(r["image_id"]))
        agreement_hs35 = sum(1 for r in results if r["corrected_top1_from_hs35_plus_norm"] == broken.get(r["image_id"]))
        print(f"\nAgreement with this stage broken ocr2_top1:")
        print(f"  corrected_top1_from_hs36_direct: {agreement_hs36}/{len(results)}")
        print(f"  corrected_top1_from_hs35_plus_norm: {agreement_hs35}/{len(results)}")
    except Exception as e:
        print(f"[warn] could not cross-reference this stage: {e}")

    out = Path("code/results/ocr2_corrected_top1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "description": (
            "Corrected OCR2 top-1 predictions computed from cached OCR2 hs[36] "
            "and the shared embed_tokens.T (base's tied lm_head, which is numerically "
            "identical to OCR2's embed_tokens). Use these in match-OCR2 calculations "
            "instead of this stage's broken ocr2_top1 values."
        ),
        "per_image": results,
    }, indent=2))

    ck = Path("docs/checkpoints/phase_2_21_complete.json")
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps({
        "phase": "2.21 OCR2-corrected top-1 recomputation",
        "status": "complete",
        "n_images": len(results),
        "output": str(out),
    }, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
