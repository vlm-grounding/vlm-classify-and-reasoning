# ============================================================
# eval_heads.py
# Load QLoRA adapter + classification heads.
# Heads predict yes/no and category; tokens generate supporting rationale.
# ============================================================

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from src.config_io import (
    load_experiment_config,
    overlay_cli,
    parse_category_names,
    render_system_prompt,
    resolve_system_prompt_path,
)
from src.data_utils import (
    answer_from_record,
    binary_from_record,
    build_chat_messages,
    category_from_record,
    classification_user_prompt,
    open_rgb_image,
    question_from_record,
    rationale_from_record,
)
from src.heads import (
    CategoryVocab,
    attach_classification_heads,
    heads_module,
    last_hidden_state,
    load_categories,
    load_classification_heads,
    pool_hidden_states,
    read_heads_file,
)
from src.metrics import METRIC_NAMES, score_prediction
from src.profiler import InferenceTimer, write_json
from src.progress import report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "workspace" / "eval_heads.json"),
        help="Eval paths, category names, and generation settings.",
    )
    parser.add_argument("--model_name", type=str, default="/content/Qwen3-VL-8B-Instruct")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--eval_manifest", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--heads_path", type=str, default=None)
    parser.add_argument("--categories_file", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--num_categories", type=int, default=None)
    parser.add_argument(
        "--category_ignore_on_negative",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Path to system_prompt.md. Must match training.",
    )
    parser.add_argument(
        "--sft_adapter",
        type=str,
        default=None,
        help="Stage 1 SFT adapter. If set, eval runs this baseline before lora_path.",
    )
    parser.add_argument(
        "--sft_output_file",
        type=str,
        default=None,
        help="Predictions for the SFT adapter. Required when --sft_adapter is set.",
    )
    parser.add_argument(
        "--sft_heads_path",
        type=str,
        default=None,
        help="classification_heads.pt next to the SFT adapter.",
    )
    return parser.parse_args()


def default_data_root():
    return str(ROOT / "data")


def resolve_heads_path(lora_path, heads_path):
    if heads_path:
        return Path(heads_path)
    candidate = Path(lora_path) / "classification_heads.pt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"No classification_heads.pt next to {lora_path}. Pass --heads_path."
    )


def resolve_vocab(args, heads_payload, cfg=None):
    cfg = cfg or {}
    names = parse_category_names(getattr(args, "categories", None))
    if not names:
        names = parse_category_names(cfg.get("categories"))
    if names:
        return CategoryVocab(names)
    extra = heads_payload.get("extra") or {}
    names = extra.get("categories")
    if names:
        return CategoryVocab(names)
    if args.categories_file:
        return load_categories(args.categories_file)
    sibling = Path(args.lora_path) / "categories.json"
    if sibling.exists():
        return load_categories(sibling)
    default_file = ROOT / "workspace" / "categories.json"
    if default_file.exists():
        return load_categories(default_file)
    return CategoryVocab()


def resolve_rendered_system_prompt(args, cfg, heads_payload, vocab):
    path = args.system_prompt or cfg.get("system_prompt")
    if path:
        return render_system_prompt(path, vocab.names), str(resolve_system_prompt_path(path))
    extra = heads_payload.get("extra") or {}
    if extra.get("system_prompt"):
        return extra["system_prompt"], extra.get("system_prompt_file")
    sibling = Path(args.lora_path) / "system_prompt.md"
    if sibling.exists():
        return sibling.read_text(encoding="utf-8").strip(), str(sibling)
    return render_system_prompt(None, vocab.names), "workspace/system_prompt.md"


def load_model_and_processor(model_name, lora_path, heads_path, vocab):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, lora_path)
    payload = read_heads_file(heads_path)
    config = payload.get("config") or {}
    num_categories = int(config.get("num_categories", len(vocab)))
    dropout = float(config.get("dropout", 0.1))
    attach_classification_heads(
        model,
        num_categories=num_categories,
        dropout=dropout,
        dtype=torch.float32,
    )
    load_classification_heads(model, heads_path)
    model.eval()
    print("Loaded PEFT adapter from:", lora_path)
    print("Loaded classification heads from:", heads_path)
    return model, processor, payload


def load_records(eval_manifest, max_samples=None):
    records = []
    with open(eval_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    if max_samples is not None:
        records = records[:max_samples]
    print(f"Loaded {len(records):,} evaluation examples.")
    return records


def generate_rationale(
    model,
    processor,
    image_path,
    user_text,
    device,
    max_new_tokens,
    system_prompt=None,
):
    messages = build_chat_messages(
        str(image_path),
        user_text,
        system_prompt=system_prompt,
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    prompt_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[:, prompt_len:]
    return processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def classify_prompt(model, processor, image_path, user_text, device, system_prompt=None):
    messages = build_chat_messages(
        str(image_path),
        user_text,
        system_prompt=system_prompt,
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden = last_hidden_state(outputs)
    pooled = pool_hidden_states(hidden, attention_mask=inputs.get("attention_mask"))
    heads = heads_module(model)
    head_param = next(heads.parameters())
    pooled = pooled.to(device=head_param.device, dtype=head_param.dtype)
    binary_logits, category_logits = heads(pooled)
    binary_id = int(binary_logits.argmax(dim=-1).item())
    category_id = int(category_logits.argmax(dim=-1).item())
    binary_prob = float(torch.softmax(binary_logits, dim=-1)[0, 1].item())
    return binary_id, category_id, binary_prob


def evaluate_adapter(
    *,
    label,
    model_name,
    lora_path,
    heads_path,
    vocab,
    records,
    data_root,
    system_prompt,
    ignore_neg,
    max_new_tokens,
    output_file,
):
    report(phase="setup", message=f"loading {label} adapter: {lora_path}")
    model, processor, _payload = load_model_and_processor(
        model_name,
        lora_path,
        heads_path,
        vocab,
    )
    device = next(model.parameters()).device
    dataset_scores = defaultdict(float)
    dataset_counts = defaultdict(int)
    binary_correct = 0
    binary_total = 0
    category_correct = 0
    category_total = 0
    results = []
    timer = InferenceTimer(method=label)

    with torch.inference_mode():
        for r in tqdm(records, desc=label):
            dataset = (r.get("dataset") or "").lower()
            try:
                _rgb, image_path = open_rgb_image(r.get("image"), data_root)
            except FileNotFoundError:
                continue

            question = question_from_record(r)
            user_text = classification_user_prompt(question)
            gt_rationale = rationale_from_record(r)
            gt = answer_from_record(r)
            if image_path is None:
                continue

            binary_id, category_id, binary_prob = classify_prompt(
                model, processor, image_path, user_text, device, system_prompt
            )
            rationale_pred = timer.time_call(
                lambda: generate_rationale(
                    model,
                    processor,
                    image_path,
                    user_text,
                    device,
                    max_new_tokens,
                    system_prompt,
                )
            )

            score = None
            if gt is not None and dataset:
                try:
                    score = score_prediction(dataset, rationale_pred, gt)
                    dataset_scores[dataset] += score
                    dataset_counts[dataset] += 1
                except ValueError:
                    score = None

            gt_binary = binary_from_record(r)
            gt_category = category_from_record(r, vocab)
            if gt_binary is not None:
                binary_total += 1
                binary_correct += int(gt_binary == binary_id)
            score_category = gt_category is not None and (
                not ignore_neg or gt_binary == 1
            )
            if score_category:
                category_total += 1
                category_correct += int(gt_category == category_id)

            results.append({
                "adapter": label,
                "id": r.get("id"),
                "original_ids": r.get("original_ids"),
                "dataset": dataset,
                "question": question,
                "rationale_gold": gt_rationale or None,
                "rationale_pred": rationale_pred,
                "ground_truth": gt,
                "prediction": rationale_pred,
                "score": None if score is None else float(score),
                "binary_pred": binary_id,
                "binary_prob_yes": binary_prob,
                "binary_gold": gt_binary,
                "category_pred": category_id,
                "category_pred_name": vocab.decode(category_id),
                "category_gold": gt_category,
                "category_gold_name": None if gt_category is None else vocab.decode(gt_category),
            })

            if len(results) == 1 or len(results) % 20 == 0:
                total = max(len(records), 1)
                report(
                    phase="eval",
                    message=f"{label}: scored {len(results):,}/{total:,}",
                    step=len(results),
                    max_steps=total,
                    percent=round(100.0 * len(results) / total, 2),
                )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n")
    print("=" * 72)
    print(f"EVALUATION RESULTS  ({label})")
    print("=" * 72)
    if binary_total:
        print(f"{'binary yes/no':18s} {'Accuracy':18s}: {binary_correct / binary_total:.4f} ({binary_total:,} samples)")
    else:
        print("binary yes/no       no labeled samples")
    category_label = "category (all)" if not ignore_neg else "category (yes)"
    if category_total:
        print(
            f"{category_label:18s} {'Accuracy':18s}: "
            f"{category_correct / category_total:.4f} ({category_total:,} samples)"
        )
    else:
        print(f"{category_label:18s} no labeled samples")

    for dataset in sorted(dataset_counts.keys()):
        n = dataset_counts[dataset]
        mean_score = dataset_scores[dataset] / n
        metric_name = METRIC_NAMES.get(dataset, "Score")
        print(f"{dataset:18s} {metric_name:18s}: {mean_score:.4f} ({n:,} samples)")
    print("=" * 72)

    scores = {
        dataset: dataset_scores[dataset] / dataset_counts[dataset]
        for dataset in dataset_counts
    }
    efficiency_path = str(Path(output_file).with_name(
        Path(output_file).stem + "_efficiency.json"
    ))
    write_json(
        efficiency_path,
        timer.summary({
            "adapter_type": label,
            "lora_path": lora_path,
            "binary_accuracy": None if not binary_total else binary_correct / binary_total,
            "category_accuracy": None if not category_total else category_correct / category_total,
            "category_ignore_on_negative": ignore_neg,
            "scores": scores,
            "counts": dict(dataset_counts),
            "categories": list(vocab.names),
        }),
    )
    print("\nPredictions saved to:")
    print(output_file)
    print("Efficiency report:")
    print(efficiency_path)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "label": label,
        "output_file": output_file,
        "binary_accuracy": None if not binary_total else binary_correct / binary_total,
        "category_accuracy": None if not category_total else category_correct / category_total,
    }


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    overlay_cli(args, cfg)
    if not args.lora_path or not args.eval_manifest or not args.output_file:
        raise SystemExit("lora_path, eval_manifest, and output_file are required.")
    if args.sft_adapter and not args.sft_output_file:
        raise SystemExit("sft_output_file is required when sft_adapter is set.")
    data_root = args.data_root or default_data_root()
    ignore_neg = args.category_ignore_on_negative
    if ignore_neg is None:
        ignore_neg = cfg.get("category_ignore_on_negative", True)

    vocab_heads = resolve_heads_path(
        args.sft_adapter or args.lora_path,
        args.sft_heads_path or args.heads_path,
    )
    heads_payload = read_heads_file(vocab_heads)
    vocab = resolve_vocab(args, heads_payload, cfg)
    system_prompt, system_prompt_path = resolve_rendered_system_prompt(
        args, cfg, heads_payload, vocab
    )
    print("Categories:", list(vocab.names))
    print(f"ignore_category_on_negative={ignore_neg}")
    print(f"system_prompt: {system_prompt_path}")
    if args.sft_adapter:
        print(f"SFT adapter:  {args.sft_adapter}")
    print(f"eval adapter: {args.lora_path}")

    report(phase="data", message="loading eval manifest")
    records = load_records(args.eval_manifest, args.max_samples)
    report(
        phase="eval",
        message=f"eval examples: {len(records):,}",
        max_steps=len(records),
        step=0,
        percent=0,
    )

    summaries = []
    if args.sft_adapter:
        sft_heads = resolve_heads_path(args.sft_adapter, args.sft_heads_path)
        summaries.append(
            evaluate_adapter(
                label="sft",
                model_name=args.model_name,
                lora_path=args.sft_adapter,
                heads_path=sft_heads,
                vocab=vocab,
                records=records,
                data_root=data_root,
                system_prompt=system_prompt,
                ignore_neg=ignore_neg,
                max_new_tokens=args.max_new_tokens,
                output_file=args.sft_output_file,
            )
        )

    grpo_heads = resolve_heads_path(args.lora_path, args.heads_path)
    summaries.append(
        evaluate_adapter(
            label="grpo" if args.sft_adapter else "sft",
            model_name=args.model_name,
            lora_path=args.lora_path,
            heads_path=grpo_heads,
            vocab=vocab,
            records=records,
            data_root=data_root,
            system_prompt=system_prompt,
            ignore_neg=ignore_neg,
            max_new_tokens=args.max_new_tokens,
            output_file=args.output_file,
        )
    )

    if len(summaries) > 1:
        print("\n")
        print("=" * 72)
        print("SFT vs GRPO")
        print("=" * 72)
        for row in summaries:
            bin_acc = row["binary_accuracy"]
            cat_acc = row["category_accuracy"]
            bin_txt = "n/a" if bin_acc is None else f"{bin_acc:.4f}"
            cat_txt = "n/a" if cat_acc is None else f"{cat_acc:.4f}"
            print(f"{row['label']:8s}  binary={bin_txt}  category={cat_txt}  {row['output_file']}")
        print("=" * 72)

    report(
        phase="eval",
        message="evaluation finished",
        step=len(records),
        max_steps=len(records),
        percent=100,
        output_file=args.output_file,
        sft_output_file=args.sft_output_file,
    )


if __name__ == "__main__":
    main()
