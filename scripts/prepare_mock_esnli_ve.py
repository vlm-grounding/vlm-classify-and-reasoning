# ============================================================
# Build a 1000/100 e-SNLI-VE mock split for train_heads / eval_heads.
#
# gold_label → category in {entailment, neutral, contradiction}
# binary     → 1 if entailment else 0
# rationale  → Explanation_1
# images     → synthetic MOCK JPEGs (safe to publish; not Flickr30k photos)
# ============================================================

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PARQUET_URL = (
    "https://huggingface.co/datasets/sedrickkeh/e-snli-ve/"
    "resolve/main/data/train-00000-of-00001.parquet"
)

LABEL_ALIASES = {
    "entailment": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
    "0": "entailment",
    "1": "neutral",
    "2": "contradiction",
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    parser.add_argument(
        "--from-manifests",
        action="store_true",
        help="Rewrite synthetic images from existing mock jsonl (no HF download).",
    )
    parser.add_argument(
        "--try-real-images",
        action="store_true",
        help="Private local use only. Do not commit downloaded Flickr photos.",
    )
    return parser.parse_args()


def normalize_label(raw) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        key = raw.strip().lower()
        return LABEL_ALIASES.get(key)
    return LABEL_ALIASES.get(raw)


def flickr_id(row: dict) -> str:
    value = row.get("Flikr30kID") or row.get("Flickr30K_ID") or row.get("flickr_id")
    text = str(value).strip()
    if text.lower().endswith(".jpg"):
        text = text[:-4]
    return text


def write_placeholder_jpeg(path: Path, image_id: str) -> None:
    from PIL import Image, ImageDraw

    digest = hashlib.md5(image_id.encode("utf-8")).digest()
    color = (digest[0], digest[1], digest[2])
    image = Image.new("RGB", (256, 256), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 246, 246), outline=(255, 255, 255), width=3)
    draw.text((18, 88), "MOCK", fill=(255, 255, 255))
    draw.text((18, 118), "not Flickr30k", fill=(255, 255, 255))
    draw.text((18, 160), image_id[:22], fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=80)


def image_id_from_record(record: dict) -> str:
    meta = record.get("metadata") or {}
    value = meta.get("flickr30k_id") or Path(str(record.get("image") or "")).stem
    return str(value).strip()


def write_placeholders_from_manifests(manifests: list[Path], image_dir: Path) -> int:
    ids = []
    seen = set()
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                image_id = image_id_from_record(json.loads(line))
                if not image_id or image_id in seen:
                    continue
                seen.add(image_id)
                ids.append(image_id)
    for image_id in ids:
        write_placeholder_jpeg(image_dir / f"{image_id}.jpg", image_id)
    return len(ids)


def try_download_flickr(image_id: str, dest: Path) -> bool:
    filename = f"{image_id}.jpg"
    urls = [
        f"https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images/{filename}",
        f"https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/{filename}",
    ]
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                payload = response.read()
            if len(payload) < 1000:
                continue
            dest.write_bytes(payload)
            return True
        except Exception:
            continue
    return False


def load_esnli_rows() -> list[dict]:
    try:
        from datasets import load_dataset

        dataset = load_dataset("sedrickkeh/e-snli-ve", split="train")
        return [dict(row) for row in dataset]
    except Exception as exc:
        print("datasets.load_dataset failed, downloading parquet:", exc)

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Install pandas+pyarrow or datasets to prepare the mock split:\n"
            "  pip install datasets pandas pyarrow pillow"
        ) from exc

    cache = ROOT / "workspace" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    parquet_path = cache / "e_snli_ve_train.parquet"
    if not parquet_path.exists():
        print("Downloading", PARQUET_URL)
        urllib.request.urlretrieve(PARQUET_URL, parquet_path)
    frame = pd.read_parquet(parquet_path)
    return frame.to_dict(orient="records")


def to_record(row: dict, image_path: Path) -> dict | None:
    category = normalize_label(row.get("gold_label"))
    if category is None:
        return None
    hypothesis = str(row.get("sentence2") or "").strip()
    premise = str(row.get("sentence1") or "").strip()
    rationale = str(row.get("Explanation_1") or "").strip()
    if not rationale:
        rationale = f"The hypothesis is judged as {category} given the image."
    image_id = flickr_id(row)
    question = hypothesis
    if premise:
        question = f"Premise caption: {premise}\nHypothesis: {hypothesis}"
    return {
        "id": str(row.get("pairID") or f"esnli|{image_id}|{category}"),
        "dataset": "esnli_ve",
        "image": str(image_path).replace("\\", "/"),
        "question": question,
        "binary": 1 if category == "entailment" else 0,
        "category": category,
        "rationale": rationale,
        "metadata": {
            "flickr30k_id": image_id,
            "gold_label": category,
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    data_root = Path(args.data_root)
    image_dir = data_root / "esnli_ve" / "images"
    train_path = data_root / "manifests" / "mock_train.jsonl"
    eval_path = data_root / "manifests" / "mock_eval.jsonl"

    if args.try_real_images:
        print(
            "WARNING: downloaded Flickr30k photos must stay local. "
            "Do not commit them to GitHub."
        )

    if args.from_manifests:
        if not train_path.exists() or not eval_path.exists():
            raise SystemExit(f"Missing manifests: {train_path} or {eval_path}")
        count = write_placeholders_from_manifests(
            [train_path, eval_path],
            image_dir,
        )
        print(f"wrote {count} synthetic MOCK images -> {image_dir}")
        return 0

    rows = load_esnli_rows()
    usable = []
    for row in rows:
        if normalize_label(row.get("gold_label")) is None:
            continue
        if not flickr_id(row):
            continue
        usable.append(row)
    if len(usable) < args.train_size + args.eval_size:
        raise SystemExit(
            f"Only {len(usable)} usable e-SNLI-VE rows; "
            f"need {args.train_size + args.eval_size}."
        )

    random.shuffle(usable)
    chosen = usable[: args.train_size + args.eval_size]
    train_rows, eval_rows = chosen[: args.train_size], chosen[args.train_size :]

    real = 0
    placeholder = 0
    manifests = []
    for split_name, split_rows in (("train", train_rows), ("eval", eval_rows)):
        converted = []
        for row in split_rows:
            image_id = flickr_id(row)
            image_path = image_dir / f"{image_id}.jpg"
            if not image_path.exists():
                got = False
                if args.try_real_images:
                    got = try_download_flickr(image_id, image_path)
                if got:
                    real += 1
                else:
                    write_placeholder_jpeg(image_path, image_id)
                    placeholder += 1
            try:
                rel_image = image_path.relative_to(data_root)
            except ValueError:
                rel_image = image_path
            record = to_record(row, rel_image)
            if record:
                converted.append(record)
        manifests.append((split_name, converted))

    write_jsonl(train_path, manifests[0][1])
    write_jsonl(eval_path, manifests[1][1])

    print(f"train: {len(manifests[0][1])} -> {train_path}")
    print(f"eval:  {len(manifests[1][1])} -> {eval_path}")
    print(f"images: {image_dir}  (real={real}, placeholder={placeholder})")
    print("categories: entailment, neutral, contradiction")
    print("binary: 1 if entailment else 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
