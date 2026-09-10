"""
Local CPU smoke tests. No GPU, no Qwen3-VL-8B, no Colab job.

    python scripts/test_cpu.py
    python scripts/ctl.py cpu-test
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require_torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        print("CPU torch is not installed. In this terminal run:")
        print(
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )
        raise SystemExit(2)


def test_data_and_collator():
    import torch

    from src.data_utils import (
        ManifestDataset,
        Qwen3VLCollator,
        answer_from_record,
        question_from_record,
        resolve_image_path,
    )

    rec_ok = {
        "question": "What color?",
        "answer": "red",
        "image": "/content/drive/MyDrive/classify_and_reasoning_qwen3-vlm8b/gqa/images/x.jpg",
    }
    rec_skip = {"question": None, "answer": None, "image": None, "metadata": {}}
    if question_from_record(rec_ok) != "What color?":
        raise AssertionError("question parse failed")
    if answer_from_record(rec_skip) is not None:
        raise AssertionError("null answer should skip")

    with tempfile.TemporaryDirectory() as tmp:
        from PIL import Image as PILImage

        data_root = Path(tmp) / "data"
        image = data_root / "gqa" / "images" / "x.jpg"
        image.parent.mkdir(parents=True)
        PILImage.new("RGB", (8, 8), (255, 0, 0)).save(image)
        resolved = resolve_image_path(rec_ok["image"], data_root)
        if Path(resolved) != image:
            raise AssertionError(f"path remap failed: {resolved}")

        manifest = Path(tmp) / "tiny.jsonl"
        rows = [
            rec_ok,
            rec_skip,
            {
                "question": "ok",
                "answer": "yes",
                "image": "/content/drive/MyDrive/classify_and_reasoning_qwen3-vlm8b/gqa/images/x.jpg",
            },
        ]
        manifest.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n",
            encoding="utf-8",
        )
        missing = resolve_image_path(
            "/content/drive/MyDrive/classify_and_reasoning_qwen3-vlm8b/chartqa/images/nope.jpg",
            data_root,
        )
        expected_missing = data_root / "chartqa" / "images" / "nope.jpg"
        if Path(missing) != expected_missing:
            raise AssertionError(
                f"missing files should still remap to data/: {missing}"
            )

        rows.append({
            "question": "missing img",
            "answer": "skip",
            "image": "/content/drive/MyDrive/classify_and_reasoning_qwen3-vlm8b/gqa/images/does_not_exist.jpg",
        })
        manifest.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n",
            encoding="utf-8",
        )
        ds = ManifestDataset(manifest, data_root=data_root)
        if len(ds) != 3:
            raise AssertionError(f"expected 3 usable rows, got {len(ds)}")
        sample = ds[2]
        if sample["question"] not in {"What color?", "ok"}:
            raise AssertionError("missing image should skip to a readable neighbor")
        if "binary_label" not in sample or "category_label" not in sample:
            raise AssertionError("examples should include head labels")

    squeezed = Qwen3VLCollator._squeeze_feature(
        "image_grid_thw",
        torch.tensor([[1, 2, 3]]),
    )
    if tuple(squeezed.shape) != (1, 3):
        raise AssertionError(f"image_grid_thw should stay 2D, got {tuple(squeezed.shape)}")
    print("ok  data parse / path remap / collator squeeze")


def test_label_parsing():
    from src.data_utils import (
        HEAD_IGNORE_INDEX,
        binary_from_record,
        category_from_record,
    )
    from src.heads import IGNORE_INDEX, CategoryVocab

    if HEAD_IGNORE_INDEX != -100 or IGNORE_INDEX != -100:
        raise AssertionError("ignore index should be -100")

    vocab = CategoryVocab(["a", "b", "c", "d", "e", "f", "g"])
    yes = binary_from_record({"binary": 1, "question": "q", "answer": "x"})
    no = binary_from_record({"yes_no": "no"})
    from_answer = binary_from_record({"answer": "Yes"})
    missing = binary_from_record({"answer": "red"})
    if yes != 1 or no != 0 or from_answer != 1 or missing is not None:
        raise AssertionError(f"binary parse failed: {yes, no, from_answer, missing}")

    cat = category_from_record({"category": "c"}, vocab)
    cat_id = category_from_record({"category_id": 6}, vocab)
    if cat != 2 or cat_id != 6:
        raise AssertionError(f"category parse failed: {cat, cat_id}")
    print("ok  binary / category label parse")


def test_rationale_supports_classification():
    from src.config_io import render_system_prompt
    from src.data_utils import (
        ManifestDataset,
        build_chat_messages,
        classification_user_prompt,
        rationale_from_record,
    )

    prompt = classification_user_prompt("Is there a finding?", ["a", "b", "c"])
    if prompt != "Is there a finding?":
        raise AssertionError("user turn should be the instance question only")
    system = render_system_prompt(category_names=["a", "b", "c"])
    if "rationale that supports" not in system:
        raise AssertionError("system_prompt.md should ask for supporting rationale")
    if "a, b, c" not in system:
        raise AssertionError("{categories} should be filled from config")
    messages = build_chat_messages("img.jpg", prompt, assistant_text="why", system_prompt=system)
    if messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise AssertionError("SFT chat should be system then user then assistant")

    if rationale_from_record({"answer": "yes"}) != "":
        raise AssertionError("yes/no labels are not rationale")
    if rationale_from_record({"rationale": "Edge irregularity supports class_2."}) != (
        "Edge irregularity supports class_2."
    ):
        raise AssertionError("explicit rationale field should win")
    if rationale_from_record({"answer": "The lesion is round."}) != "The lesion is round.":
        raise AssertionError("non-label answer may be used as rationale")

    with tempfile.TemporaryDirectory() as tmp:
        from PIL import Image as PILImage

        data_root = Path(tmp) / "data"
        image = data_root / "gqa" / "images" / "x.jpg"
        image.parent.mkdir(parents=True)
        PILImage.new("RGB", (8, 8), (255, 0, 0)).save(image)
        manifest = Path(tmp) / "cls.jsonl"
        manifest.write_text(
            json.dumps({
                "image": str(image),
                "binary": 1,
                "category": "class_2",
                "rationale": "The highlighted region matches class_2.",
            })
            + "\n",
            encoding="utf-8",
        )
        ds = ManifestDataset(
            manifest,
            data_root=data_root,
            prompt_mode="classification_reasoning",
        )
        if len(ds) != 1:
            raise AssertionError("classification_reasoning should not require a VQA question")
        sample = ds[0]
        if sample["assistant_text"] != "The highlighted region matches class_2.":
            raise AssertionError("LM target should be the rationale")
        if sample["user_text"]:
            raise AssertionError("no-question example should have an empty user turn")
        if "Write a rationale that supports" not in (sample.get("system_prompt") or ""):
            raise AssertionError("system prompt should request supporting reasoning")
        if sample["binary_label"] != 1:
            raise AssertionError("binary head label missing")
    print("ok  rationale is the token target for classification")


def test_heads_forward_and_loss():
    import torch
    import torch.nn as nn

    from src.heads import (
        ClassificationHeads,
        attach_classification_heads,
        combine_losses,
        head_losses,
        load_classification_heads,
        pool_hidden_states,
        save_classification_heads,
    )

    hidden = torch.randn(4, 6, 8)
    index = torch.tensor([1, 2, 3, 4])
    pooled = pool_hidden_states(hidden, head_token_index=index)
    if pooled.shape != (4, 8):
        raise AssertionError(f"pool shape {tuple(pooled.shape)}")
    if not torch.equal(pooled[0], hidden[0, 1]):
        raise AssertionError("pool did not take head_token_index")

    heads = ClassificationHeads(hidden_size=8, num_categories=7, dropout=0.0)
    binary_logits, category_logits = heads(pooled)
    if tuple(binary_logits.shape) != (4, 2):
        raise AssertionError(f"binary logits {tuple(binary_logits.shape)}")
    if tuple(category_logits.shape) != (4, 7):
        raise AssertionError(f"category logits {tuple(category_logits.shape)}")

    binary_labels = torch.tensor([1, 0, -100, 1])
    category_labels = torch.tensor([3, -100, -100, 0])
    loss_binary, loss_category = head_losses(
        binary_logits, category_logits, binary_labels, category_labels
    )
    if not torch.isfinite(loss_binary) or not torch.isfinite(loss_category):
        raise AssertionError("head losses should be finite")
    total = combine_losses(
        loss_binary,
        loss_category,
        torch.tensor(1.5),
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )
    expected = loss_binary + loss_category + torch.tensor(1.5)
    if abs(float(total.detach()) - float(expected.detach())) > 1e-5:
        raise AssertionError("L should be alpha*class + beta*category + gamma*log_prob")
    total.backward()
    if heads.binary_head.weight.grad is None or heads.category_head.weight.grad is None:
        raise AssertionError("both heads should receive gradients")

    ignored = head_losses(
        binary_logits.detach(),
        category_logits.detach(),
        torch.tensor([-100, -100, -100, -100]),
        torch.tensor([-100, -100, -100, -100]),
    )
    if float(ignored[0]) != 0.0 or float(ignored[1]) != 0.0:
        raise AssertionError("fully ignored labels should yield zero head loss")

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(8, 16, bias=False)

        def get_base_model(self):
            return self

    model = TinyLM()
    attach_classification_heads(model, num_categories=7, dropout=0.0, dtype=torch.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "classification_heads.pt"
        save_classification_heads(model, path, extra={"categories": ["a"] * 7})
        load_classification_heads(model, path)
    print("ok  classification heads forward / loss / save-load")


def test_config_categories_and_ignore_flag():
    from types import SimpleNamespace

    from src.config_io import load_experiment_config, overlay_cli, vocab_from_config
    from src.data_utils import HEAD_IGNORE_INDEX, ManifestDataset
    from src.heads import CategoryVocab

    cfg = load_experiment_config(ROOT / "workspace" / "train_heads.json")
    vocab = vocab_from_config(cfg)
    if list(vocab.names) != ["entailment", "neutral", "contradiction"]:
        raise AssertionError(f"config categories were not used: {vocab.names}")
    if len(vocab) != 3:
        raise AssertionError("e-SNLI-VE mock should be 3-way")
    if cfg.get("system_prompt") != "workspace/system_prompt.md":
        raise AssertionError("train config must point at system_prompt.md")
    grpo_train = load_experiment_config(ROOT / "workspace" / "train_grpo.json")
    grpo_eval = load_experiment_config(ROOT / "workspace" / "eval_grpo.json")
    if grpo_train.get("train_manifest") != cfg.get("train_manifest"):
        raise AssertionError("GRPO train must reuse the SFT train manifest")
    if grpo_eval.get("eval_manifest") != cfg.get("eval_manifest"):
        raise AssertionError("GRPO eval must reuse the SFT eval manifest")
    if not grpo_eval.get("sft_adapter") or not grpo_train.get("sft_adapter"):
        raise AssertionError("GRPO train/eval must name the SFT initial adapter")
    if grpo_eval.get("sft_adapter") != grpo_train.get("sft_adapter"):
        raise AssertionError("GRPO eval SFT adapter must match the train init adapter")

    args = SimpleNamespace(
        learning_rate=9.9,
        alpha=0.0,
        categories=None,
        num_categories=None,
    )
    overlay_cli(args, cfg, argv=["--learning_rate", "9.9"])
    if args.learning_rate != 9.9:
        raise AssertionError("CLI should win over config")
    if args.alpha != cfg["alpha"]:
        raise AssertionError("config should fill missing CLI fields")

    try:
        vocab_from_config({"categories": ["a", "b"], "num_categories": 3})
        raise AssertionError("mismatched num_categories should fail")
    except ValueError:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        from PIL import Image as PILImage

        data_root = Path(tmp) / "data"
        image = data_root / "esnli_ve" / "images" / "x.jpg"
        image.parent.mkdir(parents=True)
        PILImage.new("RGB", (8, 8), (10, 20, 30)).save(image)
        manifest = Path(tmp) / "neg.jsonl"
        manifest.write_text(
            json.dumps({
                "image": "esnli_ve/images/x.jpg",
                "binary": 0,
                "category": "contradiction",
                "rationale": "The image contradicts the hypothesis.",
            })
            + "\n",
            encoding="utf-8",
        )
        ve = CategoryVocab(["entailment", "neutral", "contradiction"])
        ignored = ManifestDataset(
            manifest,
            data_root=data_root,
            categories=ve,
            prompt_mode="classification_reasoning",
            category_ignore_on_negative=True,
        )[0]
        kept = ManifestDataset(
            manifest,
            data_root=data_root,
            categories=ve,
            prompt_mode="classification_reasoning",
            category_ignore_on_negative=False,
        )[0]
        if ignored["category_label"] != HEAD_IGNORE_INDEX:
            raise AssertionError("negatives should ignore category by default")
        if kept["category_label"] != 2:
            raise AssertionError("e-SNLI-VE should train all 3 categories")
    print("ok  config categories / ignore-on-negative")


def test_grpo_reward():
    from src.config_io import load_experiment_config
    from src.grpo_reward import (
        DEFAULT_CATEGORY_KEYWORDS,
        compute_reward,
        content_tokens,
        load_category_keywords,
        weights_from_config,
    )

    if content_tokens("The man is on a street") != ["man", "street"]:
        raise AssertionError("stop words the/a/is/on should be dropped")

    if "shown" not in DEFAULT_CATEGORY_KEYWORDS["entailment"]:
        raise AssertionError("grpo_reward.py must include the category dictionary")
    keywords = load_category_keywords()
    if keywords["contradiction"] != DEFAULT_CATEGORY_KEYWORDS["contradiction"]:
        raise AssertionError("load_category_keywords should default to the in-module dict")

    cfg = load_experiment_config(ROOT / "workspace" / "train_grpo.json")
    weights = weights_from_config(cfg)
    names = ["entailment", "neutral", "contradiction"]
    gold = "The four people are wearing black while walking down the street."
    pred = (
        "Four people are wearing black and walking. The image is consistent "
        "and shown as the same scene."
    )
    scored = compute_reward(
        rationale=pred,
        gold_rationale=gold,
        gold_binary=1,
        pred_binary=1,
        gold_category="entailment",
        pred_category="entailment",
        category_names=names,
        category_keywords=keywords,
        weights=weights,
    )
    expected = (
        weights.alpha * scored["format"]
        + weights.beta * scored["decision"]
        + weights.gamma * scored["category"]
        + weights.delta * scored["keywords"]
    )
    if abs(scored["reward"] - expected) > 1e-6:
        raise AssertionError("R must be α format + β decision + γ category + δ keywords")
    if scored["decision"] != 1.0 or scored["category"] != 1.0:
        raise AssertionError("correct heads should score 1")
    if scored["keywords_dict"] <= 0 or scored["keywords_gt"] <= 0:
        raise AssertionError("both keyword parts should fire")
    if compute_reward(rationale="yes", category_names=names)["format"] != 0.0:
        raise AssertionError("bare yes is not a valid rationale")
    print("ok  GRPO reward alpha format + beta decision + gamma category + delta keywords")


def test_grpo_math():
    import torch

    from src.grpo import (
        completion_token_logprobs,
        group_advantages,
        grpo_loss,
    )

    adv = group_advantages(torch.tensor([1.0, 3.0, 5.0]))
    if abs(float(adv.mean())) > 1e-5:
        raise AssertionError("group advantages should be zero-mean")
    if float(group_advantages(torch.tensor([2.0]))) != 0.0:
        raise AssertionError("a group of one has zero advantage")

    logits = torch.zeros(1, 5, 4)
    logits[0, 2, 1] = 5.0
    input_ids = torch.tensor([[0, 0, 0, 1, 2]])
    logp, mask = completion_token_logprobs(logits, input_ids, prompt_len=3)
    if mask[0, :2].sum() != 0 or mask[0, 2:].sum() != 2:
        raise AssertionError("completion mask should start at prompt_len")
    loss, stats = grpo_loss(
        logp,
        logp.detach(),
        mask,
        torch.tensor([1.0]),
        kl_beta=0.04,
    )
    if not torch.isfinite(loss) or stats["kl"] != 0.0:
        raise AssertionError("identical policy/ref should have zero KL")
    print("ok  GRPO advantages / completion logprobs / loss")


def test_metrics():
    from src.metrics import chartqa_score, docvqa_score, gqa_score, textvqa_score

    if gqa_score("Red.", "red") != 1.0:
        raise AssertionError("gqa normalize/exact match failed")
    if textvqa_score("cat", ["cat", "cat", "cat"]) != 1.0:
        raise AssertionError("textvqa consensus of 3/3 failed")
    if abs(textvqa_score("cat", ["cat", "cat", "dog"]) - (2.0 / 3.0)) > 1e-6:
        raise AssertionError("textvqa consensus of 2/3 failed")
    if chartqa_score("21", "20") != 1.0:
        raise AssertionError("chartqa 5% relaxed match failed")
    if docvqa_score("invoice 12", "invoice 12") != 1.0:
        raise AssertionError("docvqa exact ANLS failed")
    print("ok  metrics")


def test_job_queue_isolated():
    from src.jobs import get_job, new_job, next_queued_job

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg_dir = root / "workspace"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text(
            (ROOT / "workspace" / "config.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        job = new_job("train", root=root)
        if job["status"] != "queued":
            raise AssertionError("new job should be queued")
        loaded = get_job(job["id"], root=root)
        nxt = next_queued_job(root=root)
        if nxt is None or nxt["id"] != job["id"]:
            raise AssertionError("isolated queue did not see the test job")
        if loaded["params"]["output_dir"].endswith("qlora_r8") is False:
            raise AssertionError("train defaults were not applied")
        heads_job = new_job("train_heads", root=root)
        if "mock_heads" not in heads_job["params"]["output_dir"]:
            raise AssertionError("train_heads defaults were not applied")
        if heads_job["params"].get("num_categories") != 3:
            raise AssertionError("train_heads should default to 3 e-SNLI-VE categories")
        grpo_job = new_job("train_grpo", root=root)
        if "mock_grpo" not in grpo_job["params"]["output_dir"]:
            raise AssertionError("train_grpo defaults were not applied")
        if not grpo_job["params"].get("sft_adapter"):
            raise AssertionError("train_grpo must start from the SFT adapter")
    print("ok  job queue (temp dir, not Drive)")


def main() -> int:
    require_torch()
    tests = [
        test_data_and_collator,
        test_label_parsing,
        test_rationale_supports_classification,
        test_heads_forward_and_loss,
        test_config_categories_and_ignore_flag,
        test_grpo_reward,
        test_grpo_math,
        test_metrics,
        test_job_queue_isolated,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    if failed:
        print(f"\n{failed}/{len(tests)} failed")
        return 1
    print(f"\n{len(tests)}/{len(tests)} passed  (CPU only; 8B train still needs Colab)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
