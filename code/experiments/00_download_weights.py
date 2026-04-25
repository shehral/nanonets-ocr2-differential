"""Phase 0.5 Step 5 — download HF weights for the three target models.

Sequentially pulls weights + tokenizer/processor/config files for:
    Qwen/Qwen2.5-VL-3B-Instruct   (base)
    nanonets/Nanonets-OCR2-3B     (primary fine-tune under analysis)
    nanonets/Nanonets-OCR-s       (precursor triangulation)

Uses huggingface_hub.snapshot_download with an explicit allow_patterns list so
we skip on-disk bloat like onnx mirrors or fp32 variants when the repo ships
both. Logs per-model size to stderr for the prelaunch checkpoint.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

REPOS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "nanonets/Nanonets-OCR2-3B",
    "nanonets/Nanonets-OCR-s",
]

ALLOW = [
    "*.json",
    "*.safetensors",
    "*.py",
    "*.txt",
    "tokenizer*",
    "vocab*",
    "merges*",
    "preprocessor*",
    "generation_config*",
    "chat_template*",
]

def dir_size_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

def main() -> int:
    total_bytes = 0
    for repo in REPOS:
        t0 = time.time()
        print(f"\n[downloading] {repo}", flush=True)
        local = snapshot_download(repo_id=repo, allow_patterns=ALLOW)
        sz = dir_size_bytes(Path(local))
        total_bytes += sz
        dt = time.time() - t0
        print(
            f"[done] {repo}  path={local}  size={sz / 1e9:.2f} GB  elapsed={dt:.1f}s",
            flush=True,
        )
    print(f"\n[total cached] {total_bytes / 1e9:.2f} GB across {len(REPOS)} models")
    return 0

if __name__ == "__main__":
    sys.exit(main())
