# ============================================================
# Classification heads in parallel with rationale token prediction.
#
# Shared pooled hidden state (last prompt token):
#   1. binary  yes/no  → 2 logits (0=no, 1=yes)
#   2. category        → K logits, trained only on yes/positive rows
# LM tokens are a free-text rationale that supports those decisions.
# ============================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100

DEFAULT_POSITIVE_CATEGORIES = (
    "class_1",
    "class_2",
    "class_3",
    "class_4",
    "class_5",
    "class_6",
    "class_7",
)


class CategoryVocab:
    """Maps category names or integer ids onto [0, K)."""

    def __init__(self, names: list[str] | tuple[str, ...] | None = None):
        self.names = [str(n) for n in (names or DEFAULT_POSITIVE_CATEGORIES)]
        if not self.names:
            raise ValueError("CategoryVocab needs at least one name.")
        self.name_to_id = {n.strip().lower(): i for i, n in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def encode(self, value) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            idx = int(value)
            return idx if 0 <= idx < len(self.names) else None
        text = str(value).strip()
        if not text:
            return None
        if text.lstrip("+-").isdigit():
            idx = int(text)
            return idx if 0 <= idx < len(self.names) else None
        return self.name_to_id.get(text.lower())

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.names):
            return self.names[idx]
        return str(idx)


def load_categories(path: str | Path | None = None) -> CategoryVocab:
    if path is None:
        return CategoryVocab()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        names = payload
    else:
        names = payload.get("categories") or payload.get("names")
    if not names:
        raise ValueError(f"No categories list in {path}")
    return CategoryVocab(names)


class ClassificationHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_categories: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_categories < 1:
            raise ValueError("num_categories must be >= 1")
        self.hidden_size = int(hidden_size)
        self.num_categories = int(num_categories)
        self.head_dropout = float(dropout)
        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(self.hidden_size, 2)
        self.category_head = nn.Linear(self.hidden_size, self.num_categories)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.dropout(pooled)
        return self.binary_head(hidden), self.category_head(hidden)

    def config_dict(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_categories": self.num_categories,
            "dropout": self.head_dropout,
            "num_binary": 2,
        }


def last_hidden_state(outputs) -> torch.Tensor:
    hidden = getattr(outputs, "hidden_states", None)
    if hidden is None:
        raise RuntimeError(
            "Model did not return hidden_states. "
            "Forward with output_hidden_states=True."
        )
    if isinstance(hidden, (tuple, list)):
        hidden = hidden[-1]
    if isinstance(hidden, (tuple, list)):
        hidden = hidden[-1]
    return hidden


def pool_hidden_states(
    hidden: torch.Tensor,
    head_token_index: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool one vector per row from [batch, seq, hidden]."""
    batch = hidden.size(0)
    seq = hidden.size(1)
    if head_token_index is not None:
        index = head_token_index.to(device=hidden.device, dtype=torch.long)
        index = index.clamp(min=0, max=max(seq - 1, 0))
        return hidden[torch.arange(batch, device=hidden.device), index]
    if attention_mask is not None:
        index = attention_mask.to(device=hidden.device).long().sum(dim=1) - 1
        index = index.clamp(min=0, max=max(seq - 1, 0))
        return hidden[torch.arange(batch, device=hidden.device), index]
    return hidden[:, -1]


def infer_hidden_size_and_device(model) -> tuple[int, torch.device]:
    base = model
    if hasattr(model, "get_base_model"):
        try:
            base = model.get_base_model()
        except Exception:
            base = getattr(model, "base_model", model)

    hidden = None
    config = getattr(base, "config", None) or getattr(model, "config", None)
    if config is not None:
        hidden = getattr(config, "hidden_size", None)
        text_config = getattr(config, "text_config", None)
        if hidden is None and text_config is not None:
            hidden = getattr(text_config, "hidden_size", None)

    lm_head = getattr(base, "lm_head", None)
    if lm_head is None:
        inner = getattr(base, "model", None)
        lm_head = getattr(inner, "lm_head", None) if inner is not None else None

    if lm_head is not None:
        device = next(lm_head.parameters()).device
        if hidden is None:
            hidden = getattr(lm_head, "in_features", None)
    else:
        device = next(model.parameters()).device

    if hidden is None:
        raise RuntimeError("Could not infer hidden size from the VL model config.")
    return int(hidden), device


def infer_head_device(model) -> torch.device:
    _, device = infer_hidden_size_and_device(model)
    return device


def attach_classification_heads(
    model,
    num_categories: int = 7,
    dropout: float = 0.1,
    dtype: torch.dtype | None = torch.float32,
):
    hidden, device = infer_hidden_size_and_device(model)
    heads = ClassificationHeads(
        hidden_size=hidden,
        num_categories=num_categories,
        dropout=dropout,
    )
    if dtype is not None:
        heads = heads.to(dtype=dtype)
    heads = heads.to(device)
    if hasattr(model, "classification_heads"):
        delattr(model, "classification_heads")
    model.add_module("classification_heads", heads)
    for parameter in model.classification_heads.parameters():
        parameter.requires_grad = True
    return model


def heads_module(model) -> ClassificationHeads:
    current = model.module if hasattr(model, "module") else model
    if hasattr(current, "classification_heads"):
        return current.classification_heads
    base = getattr(current, "base_model", None)
    if base is not None and hasattr(base, "classification_heads"):
        return base.classification_heads
    raise AttributeError("classification_heads is not attached to the model.")


def _zero_like_loss(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.sum() * 0.0


def head_losses(
    binary_logits: torch.Tensor,
    category_logits: torch.Tensor,
    binary_labels: torch.Tensor | None,
    category_labels: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if binary_labels is None:
        loss_binary = _zero_like_loss(binary_logits)
    else:
        labels = binary_labels.to(device=binary_logits.device, dtype=torch.long)
        if (labels == IGNORE_INDEX).all():
            loss_binary = _zero_like_loss(binary_logits)
        else:
            loss_binary = F.cross_entropy(
                binary_logits, labels, ignore_index=IGNORE_INDEX
            )

    if category_labels is None:
        loss_category = _zero_like_loss(category_logits)
    else:
        labels = category_labels.to(device=category_logits.device, dtype=torch.long)
        if (labels == IGNORE_INDEX).all():
            loss_category = _zero_like_loss(category_logits)
        else:
            loss_category = F.cross_entropy(
                category_logits, labels, ignore_index=IGNORE_INDEX
            )
    return loss_binary, loss_category


def combine_losses(
    loss_class: torch.Tensor,
    loss_category: torch.Tensor,
    loss_log_prob: torch.Tensor | None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Single training objective:

        L = alpha * L_class + beta * L_category + gamma * L_log_prob

    L_class is binary yes/no CE.
    L_category is the 7-way CE on yes rows.
    L_log_prob is the rationale token NLL (HuggingFace LM loss = -mean log p).
    Minimizing +gamma * L_log_prob raises token log-probability.
    """
    total = alpha * loss_class + beta * loss_category
    if loss_log_prob is not None:
        total = total + gamma * loss_log_prob
    return total


def save_classification_heads(
    model,
    path: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    heads = heads_module(model)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in heads.state_dict().items()},
        "config": heads.config_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    print("Saved classification heads:", path)
    return path


def read_heads_file(path: str | Path, map_location="cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=map_location, weights_only=False)


def load_classification_heads(
    model,
    path: str | Path,
    map_location="cpu",
) -> dict[str, Any]:
    payload = read_heads_file(path, map_location=map_location)
    heads = heads_module(model)
    config = payload.get("config") or {}
    saved_categories = int(config.get("num_categories", heads.num_categories))
    if saved_categories != heads.num_categories:
        raise RuntimeError(
            f"Heads file has {saved_categories} categories, "
            f"model has {heads.num_categories}."
        )
    heads.load_state_dict(payload["state_dict"])
    print("Loaded classification heads:", path)
    return payload
