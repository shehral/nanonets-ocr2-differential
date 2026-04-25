"""assemble the 24-image differential-interpretability eval set.

Six categories, four to six images each:

    Printed mixed-layout    : 6  DocVQA validation split
    Handwritten             : 4  IAM handwriting lines
    Scientific w/ LaTeX     : 4  cropped equation regions from arXiv:1706.03762
    Receipts                : 4  SROIE samples
    Forms                   : 4  FUNSD samples
    Multilingual            : 4  zh / ar / ja / fr — Wikipedia screenshots

Each image is resized so that the Qwen2.5-VL image processor produces
approximately 1024 LLM image tokens (target = 1024 * 28 * 28 ≈ 802816 pixels;
patch_size=14 * spatial_merge_size=2 → one LLM image token covers 28x28 px).
The processor's min_pixels=max_pixels argument enforces this at runtime; here
we just resize so the aspect ratio is reasonable and the image isn't absurdly
tall or wide for the manifest.

Output manifest:  data/processed/eval_set_manifest.json
Output images:    data/processed/image_{NN}_{category}.png

The manifest records per-image SHA-256 so a downstream rerun can spot silent
changes to the source datasets.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

OUT_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
TARGET_LONG_SIDE = 896  # ~ 32 patches * (14 * 2) = 896 → ~1024 LLM tokens

@dataclass
class ImageRecord:
    image_id: str
    category: str
    source: str
    source_id: str
    sha256: str
    image_path: str
    width: int
    height: int
    ground_truth: str | None

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def resize_to_long_side(img: Image.Image, long_side: int = TARGET_LONG_SIDE) -> Image.Image:
    """Scale the longer dimension to `long_side`, preserve aspect ratio."""
    w, h = img.size
    scale = long_side / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS).convert("RGB")

def save_record(img: Image.Image, image_id: str, category: str, source: str, source_id: str, ground_truth: str | None) -> ImageRecord:
    resized = resize_to_long_side(img)
    out_path = OUT_DIR / f"image_{image_id}_{category}.png"
    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    raw = buf.getvalue()
    out_path.write_bytes(raw)
    return ImageRecord(
        image_id=image_id,
        category=category,
        source=source,
        source_id=source_id,
        sha256=sha256_bytes(raw),
        image_path=str(out_path),
        width=resized.size[0],
        height=resized.size[1],
        ground_truth=ground_truth,
    )

def build_docvqa(records: list[ImageRecord]) -> None:
    """6 DocVQA val-split images from deterministic indices."""
    from datasets import load_dataset
    print("[docvqa] loading lmms-lab/DocVQA validation...", flush=True)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=False)
    picks = [0, 100, 500, 1000, 2000, 3000]
    for n, idx in enumerate(picks, start=1):
        if idx >= len(ds):
            idx = min(len(ds) - 1, idx)
        ex = ds[idx]
        img = ex["image"]
        gt = ex.get("answers") or ex.get("answer") or None
        if isinstance(gt, list):
            gt = gt[0] if gt else None
        records.append(
            save_record(img, f"{n:02d}", "docvqa", "lmms-lab/DocVQA", f"validation[{idx}]", gt)
        )
    print(f"[docvqa] {len(picks)} saved", flush=True)

def build_iam(records: list[ImageRecord], start: int = 7) -> None:
    """4 IAM handwriting lines — English handwritten text."""
    from datasets import load_dataset
    print("[iam] loading Teklia/IAM-line train split...", flush=True)
    ds = load_dataset("Teklia/IAM-line", split="train", streaming=False)
    picks = [0, 500, 2000, 5000]
    for k, idx in enumerate(picks):
        if idx >= len(ds):
            idx = min(len(ds) - 1, idx)
        ex = ds[idx]
        img = ex["image"]
        gt = ex.get("text") or ex.get("label")
        n = start + k
        records.append(
            save_record(img, f"{n:02d}", "handwritten", "Teklia/IAM-line", f"train[{idx}]", gt)
        )
    print(f"[iam] {len(picks)} saved", flush=True)

def build_arxiv_equations(records: list[ImageRecord], start: int = 11) -> None:
    """4 equation-dense crops from arXiv:1706.03762 'Attention Is All You Need'.

    If pdf2image or the arXiv PDF is unavailable, fall back to generating four
    placeholder images that render a LaTeX equation with matplotlib. Logged
    in the manifest so Phase 3 sees the fallback.
    """
    pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
    try:
        import urllib.request
        from pdf2image import convert_from_bytes
    except Exception as e:  # noqa: BLE001
        print(f"[arxiv] pdf2image not available ({e}); using matplotlib fallback", flush=True)
        _arxiv_fallback(records, start)
        return

    try:
        print(f"[arxiv] fetching {pdf_url}", flush=True)
        with urllib.request.urlopen(pdf_url, timeout=30) as r:
            pdf_bytes = r.read()
        pages = convert_from_bytes(pdf_bytes, dpi=180, first_page=2, last_page=6)
    except Exception as e:  # noqa: BLE001
        print(f"[arxiv] fetch/convert failed ({e}); using matplotlib fallback", flush=True)
        _arxiv_fallback(records, start)
        return

    # One middle-band crop per page (where equations live in the Transformer paper).
    def middle_strip(page: Image.Image) -> Image.Image:
        w, h = page.size
        top = int(h * 0.35)
        bot = int(h * 0.75)
        return page.crop((int(w * 0.08), top, int(w * 0.92), bot))

    for k in range(4):
        page = pages[k]
        crop = middle_strip(page)
        n = start + k
        records.append(
            save_record(crop, f"{n:02d}", "arxiv_equation", "arxiv:1706.03762", f"page_{k + 2}_middle_strip", None)
        )
    print(f"[arxiv] 4 equation-crop strips saved", flush=True)

def _arxiv_fallback(records: list[ImageRecord], start: int) -> None:
    """Render four LaTeX equations via matplotlib's mathtext (no tex install required)."""
    import matplotlib.pyplot as plt

    equations = [
        r"$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V$",
        r"$\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^{O}$",
        r"$\mathrm{FFN}(x) = \max(0, xW_1 + b_1)\,W_2 + b_2$",
        r"$\mathrm{PE}_{(pos,2i)} = \sin\!\left(pos / 10000^{2i/d_\mathrm{model}}\right)$",
    ]
    for k, eq in enumerate(equations):
        fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
        ax.axis("off")
        ax.text(0.5, 0.5, eq, ha="center", va="center", fontsize=22)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        n = start + k
        records.append(
            save_record(img, f"{n:02d}", "arxiv_equation", "matplotlib_fallback", f"eq_{k + 1}", eq)
        )

def build_sroie(records: list[ImageRecord], start: int = 15) -> None:
    """4 SROIE receipt samples."""
    from datasets import load_dataset

    candidates = [
        ("darentang/sroie", {"split": "test"}),
        ("mychen76/invoices-and-receipts_ocr_v2", {"split": "train"}),
    ]
    ds = None
    chosen = None
    for name, kwargs in candidates:
        try:
            print(f"[sroie] trying {name} ...", flush=True)
            ds = load_dataset(name, **kwargs)
            chosen = name
            break
        except Exception as e:  # noqa: BLE001
            print(f"[sroie] {name} failed: {e}", flush=True)

    if ds is None:
        print("[sroie] all candidates failed — rendering placeholder receipts", flush=True)
        _text_fallback(records, start, "receipt",
                       ["Walmart $12.99 Milk\nSubtotal 12.99\nTax 0.89\nTotal 13.88",
                        "Starbucks Latte 4.75\nCroissant 3.25\nTotal 8.00",
                        "Store #214 Receipt\nItem A 1.50\nItem B 3.20\nItem C 0.75\nTotal 5.45",
                        "Target Super 49.99\nApples 2.19\nBread 3.49\nTotal 55.67"], source="placeholder")
        return

    picks = [0, 5, 10, 15]
    image_field = None
    for f in ["image", "img", "pixel_values"]:
        if f in ds.features:
            image_field = f
            break
    if image_field is None:
        # fallback: first column of type Image
        for f, ft in ds.features.items():
            if "Image" in type(ft).__name__:
                image_field = f
                break

    for k, idx in enumerate(picks):
        if idx >= len(ds):
            idx = min(len(ds) - 1, idx)
        ex = ds[idx]
        img = ex[image_field]
        n = start + k
        records.append(
            save_record(img, f"{n:02d}", "receipt", chosen, f"[{idx}]", None)
        )
    print(f"[sroie] 4 saved from {chosen}", flush=True)

def build_funsd(records: list[ImageRecord], start: int = 19) -> None:
    """4 FUNSD form samples."""
    from datasets import load_dataset

    candidates = [
        ("nielsr/funsd", {"split": "test"}),
        ("nielsr/funsd-layoutlmv3", {"split": "test"}),
    ]
    ds = None
    chosen = None
    for name, kwargs in candidates:
        try:
            print(f"[funsd] trying {name} ...", flush=True)
            ds = load_dataset(name, **kwargs)
            chosen = name
            break
        except Exception as e:  # noqa: BLE001
            print(f"[funsd] {name} failed: {e}", flush=True)

    if ds is None:
        print("[funsd] all candidates failed — using placeholder", flush=True)
        _text_fallback(records, start, "form",
                       ["Name: ____\nAddress: ____\nPhone: ____",
                        "Applicant Info\nSSN: ___-__-____\nDOB: __/__/____",
                        "Emergency Contact\nName: ____\nRelation: ____",
                        "Signature: ____  Date: __/__/____"], source="placeholder")
        return

    picks = [0, 2, 5, 10]
    image_field = "image"
    for k, idx in enumerate(picks):
        if idx >= len(ds):
            idx = min(len(ds) - 1, idx)
        ex = ds[idx]
        img = ex[image_field]
        n = start + k
        records.append(
            save_record(img, f"{n:02d}", "form", chosen, f"test[{idx}]", None)
        )
    print(f"[funsd] 4 saved from {chosen}", flush=True)

def build_multilingual(records: list[ImageRecord], start: int = 23) -> None:
    """4 multilingual samples — rendered text blocks in zh, ar, ja, fr.

    Using rendered text blocks rather than Wikipedia screenshots avoids the
    Wikimedia WebFetch dependency and keeps the manifest reproducible offline.
    This is a known simplification documented in spec § 5 footnote.
    """
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    samples = [
        ("zh", "人工智能正在改变世界。\n机器学习是其核心技术。\n深度学习模型的参数数量\n已经达到万亿级别。", "zh_cn"),
        ("ar", "الذكاء الاصطناعي يغير العالم.\nتعلم الآلة هو تقنيته الأساسية.\nوصلت نماذج التعلم العميق\nإلى تريليونات المعاملات.", "ar"),
        ("ja", "人工知能は世界を変えています。\n機械学習はその中核技術です。\n深層学習モデルのパラメータは\n数兆個に達しています。", "ja"),
        ("fr", "L'intelligence artificielle transforme\nle monde. L'apprentissage automatique\nen est la technologie fondamentale.\nLes modèles ont des milliards\nde paramètres.", "fr"),
    ]

    # Try to find fonts with coverage
    font_candidates = {
        "zh": ["Heiti TC", "PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS"],
        "ar": ["Geeza Pro", "Al Bayan", "Arial Unicode MS"],
        "ja": ["Hiragino Sans", "Hiragino Maru Gothic Pro", "Arial Unicode MS"],
        "fr": ["Helvetica", "Arial"],
    }

    for k, (lang, text, tag) in enumerate(samples):
        chosen_font = None
        for name in font_candidates.get(lang, []):
            try:
                path = font_manager.findfont(name, fallback_to_default=False)
                if path:
                    chosen_font = path
                    break
            except Exception:  # noqa: BLE001
                continue
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        ax.axis("off")
        ax.set_facecolor("white")
        font_kwargs = {"fontsize": 18}
        if chosen_font:
            from matplotlib.font_manager import FontProperties
            font_kwargs["fontproperties"] = FontProperties(fname=chosen_font)
        ax.text(0.05, 0.5, text, ha="left", va="center", **font_kwargs)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        n = start + k
        records.append(
            save_record(img, f"{n:02d}", f"multilingual_{tag}", "rendered_text", lang, text.replace("\n", " "))
        )
    print("[multilingual] 4 rendered-text samples saved", flush=True)

def _text_fallback(records: list[ImageRecord], start: int, cat: str, texts: Iterable[str], source: str) -> None:
    import matplotlib.pyplot as plt

    for k, text in enumerate(texts):
        fig, ax = plt.subplots(figsize=(6, 8), dpi=150)
        ax.axis("off")
        ax.set_facecolor("white")
        ax.text(0.05, 0.95, text, ha="left", va="top", fontsize=14, family="monospace")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        n = start + k
        records.append(
            save_record(img, f"{n:02d}", cat, source, f"placeholder_{k}", text)
        )

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    records: list[ImageRecord] = []
    build_docvqa(records)
    build_iam(records)
    build_arxiv_equations(records)
    build_sroie(records)
    build_funsd(records)
    build_multilingual(records)

    manifest = {
        "version": "2026-04-23",
        "total_images": len(records),
        "target_long_side_px": TARGET_LONG_SIDE,
        "processor_target_tokens": 1024,
        "notes": "Category counts: docvqa=6, handwritten=4, arxiv_equation=4, receipt=4, form=4, multilingual=4 (= 26 images, reconciling spec §5 table against the narrative '24')",
        "images": [asdict(r) for r in records],
    }
    manifest_path = OUT_DIR / "eval_set_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nmanifest written: {manifest_path}  ({len(records)} records)")
    assert len(records) == 26, f"expected 26 images (spec §5 table), got {len(records)}"
    return 0

if __name__ == "__main__":
    sys.exit(main())
