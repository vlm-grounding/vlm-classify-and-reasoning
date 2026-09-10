# ============================================================
# data_utils.py
# Dataset + collator for Qwen3-VL SFT
# ============================================================

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

DATASETS = (
    "chartqa",
    "textvqa",
    "gqa",
    "docvqa",
    "visual_genome",
    "esnli_ve",
)

# Keep vision tokens inside max_length=2048. Qwen3-VL default longest_edge
# is huge, which produced 2100+ image tokens and a token/feature mismatch.
MAX_IMAGE_SIDE = 1024

HEAD_IGNORE_INDEX = -100

_YES_VALUES = {
    "yes", "y", "true", "t", "1", "positive", "pos", "present",
}
_NO_VALUES = {
    "no", "n", "false", "f", "0", "negative", "neg", "absent",
}


def answer_to_text(answer):
    """
    Normalize different VQA answer formats into one target string.
    """
    if answer is None:
        return ""

    if isinstance(answer, str):
        return answer

    if isinstance(answer, (int, float)):
        return str(answer)

    if isinstance(answer, list):
        if len(answer) == 0:
            return ""

        # VQA may contain list[str]
        if all(isinstance(x, str) for x in answer):
            # use first answer for SFT
            return answer[0]

        # Sometimes list[dict]
        if isinstance(answer[0], dict):
            for key in ["answer", "text", "label"]:
                if key in answer[0]:
                    return str(answer[0][key])

        return str(answer[0])

    if isinstance(answer, dict):
        for key in ["answer", "text", "label"]:
            if key in answer:
                return str(answer[key])

    return str(answer)


def _first_text_field(record, keys):
    for key in keys:
        if record.get(key) is not None:
            text = answer_to_text(record.get(key)).strip()
            if text:
                return text
    meta = record.get("metadata") or {}
    for key in keys:
        if meta.get(key) is not None:
            text = answer_to_text(meta.get(key)).strip()
            if text:
                return text
    return ""


def rationale_from_record(record):
    """
    Free-text reasoning that supports the classification heads.

    Label-only strings like "yes" / "no" are not used as rationale.
    """
    text = _first_text_field(
        record,
        (
            "rationale",
            "reasoning",
            "explanation",
            "justification",
            "rationale_text",
            "thought",
        ),
    )
    if text:
        return text

    answer = answer_from_record(record)
    raw = answer_to_text(answer).strip() if answer is not None else ""
    if not raw or parse_binary_value(raw) is not None:
        return ""
    return raw


def default_supporting_rationale(binary, category_name=None):
    if binary == 0:
        return (
            "The image does not show positive evidence for any of the "
            "listed categories."
        )
    if binary == 1 and category_name:
        return (
            f"Visual evidence supports a positive finding in category "
            f"{category_name}."
        )
    if binary == 1:
        return "Visual evidence supports a positive finding."
    return (
        "The image is inspected for a positive finding and the "
        "visual evidence is described."
    )


def classification_user_prompt(question, category_names=None):
    """Instance text only. Task instructions live in the system prompt."""
    del category_names
    return (question or "").strip()


def build_chat_messages(image, user_text, assistant_text=None, system_prompt=None):
    """Standard LoRA SFT turn: optional system, user (image+text), optional assistant."""
    messages = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": str(system_prompt),
        })
    user_content = [{"type": "image", "image": image}]
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    messages.append({"role": "user", "content": user_content})
    if assistant_text is not None:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}],
        })
    return messages


def question_from_record(record):
    """Read question from the top-level field, then metadata fallbacks."""
    question = record.get("question")
    if question is not None:
        return str(question).strip()

    meta = record.get("metadata") or {}
    for key in ["question", "query", "prompt", "instruction"]:
        if meta.get(key) is not None:
            return str(meta[key]).strip()

    return ""


def answer_from_record(record):
    """Read answer from the top-level field, then metadata fallbacks."""
    answer = record.get("answer")
    if answer is not None:
        return answer

    meta = record.get("metadata") or {}
    for key in ["answer", "answers", "label", "labels", "target"]:
        if meta.get(key) is not None:
            return meta[key]

    return None


def parse_binary_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = int(value)
        if number in (0, 1):
            return number
        return None
    text = str(value).strip().lower()
    if text in _YES_VALUES:
        return 1
    if text in _NO_VALUES:
        return 0
    return None


def binary_from_record(record):
    """Yes/no label: 1=yes, 0=no. None if the record has no binary target."""
    for key in (
        "binary",
        "binary_label",
        "yes_no",
        "is_positive",
        "positive",
        "label_binary",
    ):
        if record.get(key) is not None:
            parsed = parse_binary_value(record.get(key))
            if parsed is not None:
                return parsed

    meta = record.get("metadata") or {}
    for key in (
        "binary",
        "binary_label",
        "yes_no",
        "is_positive",
        "positive",
    ):
        if meta.get(key) is not None:
            parsed = parse_binary_value(meta.get(key))
            if parsed is not None:
                return parsed

    answer = answer_from_record(record)
    return parse_binary_value(answer_to_text(answer) if answer is not None else None)


def category_from_record(record, vocab):
    """Positive-class id in [0, K), or None if missing / not applicable."""
    for key in (
        "category",
        "category_id",
        "class",
        "class_id",
        "positive_category",
        "label_category",
    ):
        if record.get(key) is not None:
            encoded = vocab.encode(record.get(key))
            if encoded is not None:
                return encoded

    meta = record.get("metadata") or {}
    for key in (
        "category",
        "category_id",
        "class",
        "class_id",
        "positive_category",
    ):
        if meta.get(key) is not None:
            encoded = vocab.encode(meta.get(key))
            if encoded is not None:
                return encoded
    return None


def image_path_candidates(image_path, data_root=None):
    """
    Ordered paths to try. Remapped data/ locations come first.

    Manifests store Colab paths like:
      /content/drive/MyDrive/classify_and_reasoning_qwen3-vlm8b/gqa/images/...

    Files actually live under:
      <repo>/data/gqa/images/...
    """
    if image_path is None:
        return []

    if isinstance(image_path, list):
        image_path = image_path[0]

    original = Path(str(image_path))
    posix = str(image_path).replace("\\", "/")
    candidates = []

    if data_root is not None:
        data_root = Path(data_root)
        for dataset in DATASETS:
            token = f"/{dataset}/"
            padded = f"/{posix}" if not posix.startswith("/") else posix
            if token in padded:
                rel = padded.split(token, 1)[1]
                candidates.append(data_root / dataset / rel)
                break

    marker = "classify_and_reasoning_qwen3-vlm8b/"
    if marker in posix and f"{marker}data/" not in posix:
        idx = posix.find(marker) + len(marker)
        candidates.append(Path(posix[:idx] + "data/" + posix[idx:]))

    candidates.append(original)

    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_image_path(image_path, data_root=None):
    """Return the first existing candidate, else the remapped data/ path."""
    candidates = image_path_candidates(image_path, data_root)
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def open_rgb_image(image_path, data_root=None):
    last_error = None
    candidates = image_path_candidates(image_path, data_root)
    if not candidates:
        raise FileNotFoundError(f"No image path in record: {image_path}")
    for candidate in candidates:
        try:
            with Image.open(candidate) as image:
                image.load()
                rgb = image.convert("RGB")
            rgb.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            return rgb, candidate
        except (FileNotFoundError, OSError, Image.UnidentifiedImageError) as exc:
            last_error = exc
    raise FileNotFoundError(
        f"Image not readable. Tried: {[str(c) for c in candidates]}"
    ) from last_error


class ManifestDataset(Dataset):

    def __init__(
        self,
        manifest_path,
        data_root=None,
        categories=None,
        prompt_mode="vqa",
        category_ignore_on_negative=True,
        system_prompt=None,
    ):
        from src.heads import CategoryVocab, DEFAULT_POSITIVE_CATEGORIES

        if prompt_mode not in {"vqa", "classification_reasoning"}:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root) if data_root else None
        self.skipped_images = 0
        self.prompt_mode = prompt_mode
        self.category_ignore_on_negative = bool(category_ignore_on_negative)
        if isinstance(categories, CategoryVocab):
            self.categories = categories
        else:
            self.categories = CategoryVocab(categories or DEFAULT_POSITIVE_CATEGORIES)
        if prompt_mode == "classification_reasoning" and system_prompt is None:
            from src.config_io import render_system_prompt

            system_prompt = render_system_prompt(category_names=self.categories.names)
        self.system_prompt = system_prompt

        self.records = []

        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                r = json.loads(line)
                image = r.get("image")
                if image is None:
                    continue

                if self.prompt_mode == "classification_reasoning":
                    has_binary = binary_from_record(r) is not None
                    has_rationale = bool(rationale_from_record(r))
                    if not has_binary and not has_rationale:
                        continue
                else:
                    question = question_from_record(r)
                    answer = answer_from_record(r)
                    if not question or answer is None:
                        continue

                self.records.append(r)

        print(
            f"Loaded {len(self.records):,} usable examples "
            f"from {self.manifest_path}"
        )

    def __len__(self):
        return len(self.records)

    def _example_from_record(self, r):
        image, _resolved = open_rgb_image(r["image"], self.data_root)
        question = question_from_record(r)
        answer = answer_to_text(answer_from_record(r)).strip()
        binary = binary_from_record(r)
        category = category_from_record(r, self.categories)
        if binary == 0 and self.category_ignore_on_negative:
            category_label = HEAD_IGNORE_INDEX
            category_name = None
        elif category is None:
            category_label = HEAD_IGNORE_INDEX
            category_name = None
        else:
            category_label = int(category)
            category_name = self.categories.decode(category_label)

        rationale = rationale_from_record(r)
        if self.prompt_mode == "classification_reasoning":
            user_text = classification_user_prompt(question)
            assistant_text = rationale or default_supporting_rationale(
                binary, category_name
            )
        else:
            user_text = question
            assistant_text = answer

        return {
            "id": r.get("id"),
            "dataset": r.get("dataset"),
            "image": image,
            "question": question,
            "answer": answer,
            "rationale": rationale,
            "system_prompt": self.system_prompt,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "binary_label": HEAD_IGNORE_INDEX if binary is None else int(binary),
            "category_label": category_label,
        }

    def __getitem__(self, idx):
        n = len(self.records)
        if n == 0:
            raise IndexError("ManifestDataset is empty")
        last_error = None
        for offset in range(n):
            r = self.records[(idx + offset) % n]
            try:
                return self._example_from_record(r)
            except (FileNotFoundError, OSError, Image.UnidentifiedImageError) as exc:
                last_error = exc
                self.skipped_images += 1
                print(
                    f"Skipping missing image ({self.skipped_images}): "
                    f"{r.get('id')} {r.get('image')}"
                )
        raise FileNotFoundError(
            "Every training image was unreadable"
        ) from last_error


class Qwen3VLCollator:
    """
    Builds multimodal Qwen3-VL SFT batches.

    Token loss is applied only to assistant tokens. With classification
    heads enabled, those tokens are the supporting rationale, not the
    yes/no or category labels.
    """

    def __init__(
        self,
        processor,
        max_length=2048,
        include_classification_heads=False,
        system_prompt=None,
    ):
        self.processor = processor
        self.max_length = max_length
        self.include_classification_heads = include_classification_heads
        self.system_prompt = system_prompt

    def __call__(self, examples):
        batch_inputs = []
        examples = [x for x in examples if x is not None]

        for x in examples:
            user_text = x.get("user_text") or x.get("question") or ""
            assistant_text = (
                x.get("assistant_text")
                or x.get("rationale")
                or x.get("answer")
                or ""
            )
            system_prompt = x.get("system_prompt") or self.system_prompt
            messages = build_chat_messages(
                x["image"],
                user_text,
                assistant_text=assistant_text,
                system_prompt=system_prompt,
            )

            encoded = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )

            # Remove singleton batch dimension.
            # image_grid_thw must stay [num_images, 3], not [3].
            sample = {}

            for k, v in encoded.items():
                if torch.is_tensor(v):
                    sample[k] = self._squeeze_feature(k, v)
                else:
                    sample[k] = v

            # ------------------------------------------------
            # Construct labels
            #
            # Better than training on prompt/image tokens.
            # Find answer boundary from prompt-only encoding.
            # ------------------------------------------------

            prompt_messages = build_chat_messages(
                x["image"],
                user_text,
                system_prompt=system_prompt,
            )

            prompt_encoded = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            prompt_len = prompt_encoded["input_ids"].shape[1]
            seq_len = sample["input_ids"].shape[0]

            # Never cut image tokens. That leaves extra pixel_values and
            # crashes with "Image features and image tokens do not match".
            if prompt_len > self.max_length:
                print(
                    f"Skipping example {x.get('id')}: "
                    f"prompt+image is {prompt_len} tokens > max_length={self.max_length}"
                )
                continue

            if seq_len > self.max_length:
                for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
                    if key in sample and torch.is_tensor(sample[key]):
                        sample[key] = sample[key][:self.max_length]

            labels = sample["input_ids"].clone()
            labels[:prompt_len] = -100
            sample["labels"] = labels
            if self.include_classification_heads:
                sample["binary_label"] = int(
                    x.get("binary_label", HEAD_IGNORE_INDEX)
                )
                sample["category_label"] = int(
                    x.get("category_label", HEAD_IGNORE_INDEX)
                )
                sample["head_token_index"] = int(max(prompt_len - 1, 0))
            batch_inputs.append(sample)

        if not batch_inputs:
            raise RuntimeError(
                "Every example in this batch exceeded max_length. "
                "Lower image size or raise --max_length."
            )

        return self._pad(batch_inputs)

    @staticmethod
    def _squeeze_feature(key, value):
        if value.ndim >= 2 and value.shape[0] == 1:
            value = value.squeeze(0)
        if key == "image_grid_thw" and value.ndim == 1:
            value = value.unsqueeze(0)
        return value

    @staticmethod
    def _stack_vision(key, vals):
        if key == "image_grid_thw":
            vals = [
                v.unsqueeze(0) if v.ndim == 1 else v
                for v in vals
            ]
        try:
            return torch.cat(vals, dim=0)
        except Exception:
            return torch.stack(vals)

    def _pad(self, examples):
        max_len = max(x["input_ids"].shape[0] for x in examples)
        max_len = min(max_len, self.max_length)

        pad_id = self.processor.tokenizer.pad_token_id

        input_ids = []
        attention_masks = []
        labels = []
        token_types = []

        for x in examples:
            ids = x["input_ids"][:max_len]
            mask = x["attention_mask"][:max_len]
            lab = x["labels"][:max_len]
            pad_len = max_len - len(ids)

            ids = torch.cat([
                ids,
                torch.full((pad_len,), pad_id, dtype=ids.dtype),
            ])
            mask = torch.cat([
                mask,
                torch.zeros(pad_len, dtype=mask.dtype),
            ])
            lab = torch.cat([
                lab,
                torch.full((pad_len,), -100, dtype=lab.dtype),
            ])

            input_ids.append(ids)
            attention_masks.append(mask)
            labels.append(lab)

            if x.get("mm_token_type_ids") is not None:
                tt = x["mm_token_type_ids"]
                if tt.ndim == 0:
                    tt = tt.unsqueeze(0)
                tt = tt[:max_len]
                tt = torch.cat([
                    tt,
                    torch.zeros(max_len - len(tt), dtype=tt.dtype),
                ])
                token_types.append(tt)

        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }
        if self.include_classification_heads:
            batch["binary_labels"] = torch.tensor(
                [x.get("binary_label", HEAD_IGNORE_INDEX) for x in examples],
                dtype=torch.long,
            )
            batch["category_labels"] = torch.tensor(
                [x.get("category_label", HEAD_IGNORE_INDEX) for x in examples],
                dtype=torch.long,
            )
            batch["head_token_index"] = torch.tensor(
                [x.get("head_token_index", 0) for x in examples],
                dtype=torch.long,
            )
        if token_types:
            batch["mm_token_type_ids"] = torch.stack(token_types)

        # Vision fields are concatenated across images, not batched like text.
        for key in ("pixel_values", "image_grid_thw"):
            vals = [
                x[key]
                for x in examples
                if key in x and x[key] is not None
            ]
            if vals:
                batch[key] = self._stack_vision(key, vals)

        return batch
