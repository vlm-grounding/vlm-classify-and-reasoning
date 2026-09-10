# ============================================================
# Load train/eval experiment JSON. Categories are user-defined.
# ============================================================

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from src.heads import CategoryVocab, load_categories

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM_PROMPT = REPO_ROOT / "workspace" / "system_prompt.md"


def load_experiment_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return payload


def overlay_cli(args, cfg: dict[str, Any], argv: list[str] | None = None):
    """Fill argparse fields from config unless the user passed that flag."""
    argv = argv if argv is not None else sys.argv[1:]
    for key, value in cfg.items():
        if not hasattr(args, key):
            continue
        flag = f"--{key}"
        dashed = flag.replace("_", "-")
        no_flag = f"--no-{key}"
        no_dashed = f"--no-{key.replace('_', '-')}"
        if any(token in argv for token in (flag, dashed, no_flag, no_dashed)):
            continue
        setattr(args, key, value)
    return args


def parse_category_names(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    raise ValueError(f"categories must be a list or comma string, got {type(raw)}")


def vocab_from_config(cfg: dict[str, Any], args=None) -> CategoryVocab:
    names = []
    if args is not None and getattr(args, "categories", None):
        names = parse_category_names(args.categories)
    if not names:
        names = parse_category_names(cfg.get("categories"))
    num = cfg.get("num_categories")
    if args is not None and getattr(args, "num_categories", None) is not None:
        num = args.num_categories

    if names:
        if num is not None and int(num) != len(names):
            raise ValueError(
                f"num_categories={num} does not match categories list "
                f"({len(names)} names): {names}"
            )
        return CategoryVocab(names)

    if num is not None:
        return CategoryVocab([f"class_{i + 1}" for i in range(int(num))])

    categories_file = None
    if args is not None:
        categories_file = getattr(args, "categories_file", None)
    categories_file = categories_file or cfg.get("categories_file")
    if categories_file:
        return load_categories(categories_file)
    return CategoryVocab()


def resolve_system_prompt_path(raw: str | Path | None = None) -> Path:
    if raw is None or str(raw).strip() == "":
        path = DEFAULT_SYSTEM_PROMPT
    else:
        path = Path(str(raw))
        if not path.is_file():
            path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"system_prompt file not found: {raw}")
    return path


def render_system_prompt(
    path: str | Path | None = None,
    category_names=None,
) -> str:
    """Load system_prompt.md and fill {categories} from the experiment config."""
    text = resolve_system_prompt_path(path).read_text(encoding="utf-8").strip()
    names = [str(name).strip() for name in (category_names or []) if str(name).strip()]
    listed = ", ".join(names) if names else "the configured classes"
    return text.replace("{categories}", listed)
