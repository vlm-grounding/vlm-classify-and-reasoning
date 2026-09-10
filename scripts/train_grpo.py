# ============================================================
# train_grpo.py
#
# Stage 2 GRPO from the Stage 1 SFT adapter, same classification
# manifests. Sample G rationales, reward
#   R = α·format + β·decision + γ·category + δ·keywords
# Heads are read at the last generated token.
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
from transformers.trainer_utils import get_last_checkpoint

from src.config_io import (
    load_experiment_config,
    overlay_cli,
    render_system_prompt,
    vocab_from_config,
)
from src.data_utils import (
    HEAD_IGNORE_INDEX,
    ManifestDataset,
    build_chat_messages,
)
from src.grpo import (
    completion_token_logprobs,
    group_advantages,
    grpo_loss,
)
from src.grpo_reward import (
    compute_reward,
    load_category_keywords,
    weights_from_config,
)
from src.heads import (
    attach_classification_heads,
    head_losses,
    heads_module,
    last_hidden_state,
    load_classification_heads,
    pool_hidden_states,
    save_classification_heads,
)
from src.progress import report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "workspace" / "train_grpo.json"),
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--sft_adapter", type=str, default=None)
    parser.add_argument("--sft_heads_path", type=str, default=None)
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--eval_manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--category_keywords", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--num_categories", type=int, default=None)
    parser.add_argument(
        "--category_ignore_on_negative",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--kl_beta", type=float, default=0.04)
    parser.add_argument("--head_ce_alpha", type=float, default=0.1)
    parser.add_argument("--head_ce_beta", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--head_learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def default_data_root():
    return str(ROOT / "data")


def resolve_sft_heads(sft_adapter: str, sft_heads_path: str | None) -> Path:
    if sft_heads_path:
        path = Path(sft_heads_path)
        if path.exists():
            return path
    candidate = Path(sft_adapter) / "classification_heads.pt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"No classification_heads.pt for SFT adapter {sft_adapter}"
    )


def to_device(payload: dict, device) -> dict:
    moved = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def lora_snapshot(model) -> dict[str, torch.Tensor]:
    snap = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classification_heads" in name:
            continue
        snap[name] = param.detach().clone()
    return snap


def apply_lora_snapshot(model, snapshot: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    restored = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in snapshot:
                continue
            restored[name] = param.detach().clone()
            param.data.copy_(snapshot[name])
    return restored


def encode_messages(processor, messages, device, add_generation_prompt: bool):
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
    )
    return to_device(encoded, device)


def build_optimizer(model, lr: float, head_lr: float):
    lora_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classification_heads" in name:
            head_params.append(param)
        else:
            lora_params.append(param)
    groups = [
        {"params": lora_params, "lr": lr},
        {"params": head_params, "lr": head_lr},
    ]
    try:
        from bitsandbytes.optim import PagedAdamW8bit

        return PagedAdamW8bit(groups, lr=lr)
    except Exception:
        return torch.optim.AdamW(groups, lr=lr)


def gold_fields(example, vocab, ignore_neg: bool):
    binary = example.get("binary_label", HEAD_IGNORE_INDEX)
    category = example.get("category_label", HEAD_IGNORE_INDEX)
    gold_binary = None if int(binary) == HEAD_IGNORE_INDEX else int(binary)
    gold_category = None
    if int(category) != HEAD_IGNORE_INDEX:
        gold_category = vocab.decode(int(category))
    score_category = gold_category is not None and (
        not ignore_neg or gold_binary == 1
    )
    return gold_binary, gold_category, score_category


def save_policy(model, processor, output_dir, vocab, extra, system_prompt):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    save_classification_heads(
        model,
        output_dir / "classification_heads.pt",
        extra=extra,
    )
    (output_dir / "categories.json").write_text(
        json.dumps({"categories": list(vocab.names)}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "system_prompt.md").write_text(system_prompt + "\n", encoding="utf-8")
    return output_dir


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    overlay_cli(args, cfg)
    if not args.sft_adapter or not args.train_manifest or not args.output_dir:
        raise SystemExit("sft_adapter, train_manifest, and output_dir are required.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = args.data_root or default_data_root()
    vocab = vocab_from_config(cfg, args)
    ignore_neg = args.category_ignore_on_negative
    if ignore_neg is None:
        ignore_neg = cfg.get("category_ignore_on_negative", True)
    system_prompt_path = args.system_prompt or cfg.get("system_prompt")
    system_prompt = render_system_prompt(system_prompt_path, vocab.names)
    keywords_path = args.category_keywords or cfg.get("category_keywords")
    category_keywords = load_category_keywords(keywords_path)
    reward_weights = weights_from_config(cfg)
    sft_heads = resolve_sft_heads(args.sft_adapter, args.sft_heads_path)

    print("Stage 2 GRPO")
    print("SFT adapter:", args.sft_adapter)
    print("Categories:", list(vocab.names))
    print(f"group_size={args.group_size}  kl_beta={args.kl_beta}")
    print(
        "reward R = "
        f"{reward_weights.alpha}*format + {reward_weights.beta}*decision + "
        f"{reward_weights.gamma}*category + {reward_weights.delta}*keywords"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    report(phase="setup", message="loading processor, 4-bit model, and SFT adapter")
    processor = AutoProcessor.from_pretrained(args.model_name)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and isinstance(getattr(image_processor, "size", None), dict):
        image_processor.size["longest_edge"] = min(
            int(image_processor.size.get("longest_edge", 1024 * 1024)),
            1024 * 1024,
        )

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    from peft import prepare_model_for_kbit_training

    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    base.config.use_cache = False

    resume = get_last_checkpoint(args.output_dir) if Path(args.output_dir).exists() else None
    policy_src = resume or args.sft_adapter
    model = PeftModel.from_pretrained(base, policy_src, is_trainable=True)
    attach_classification_heads(
        model,
        num_categories=len(vocab),
        dropout=0.1,
        dtype=torch.float32,
    )
    heads_init = sft_heads
    if resume:
        resume_heads = Path(resume) / "classification_heads.pt"
        if resume_heads.exists():
            heads_init = resume_heads
    load_classification_heads(model, heads_init)
    model.print_trainable_parameters()

    device = next(model.parameters()).device
    ref_weights = lora_snapshot(model)
    print(f"Frozen SFT LoRA snapshot: {len(ref_weights)} tensors")

    report(phase="data", message="loading train manifest (same as SFT)")
    train_dataset = ManifestDataset(
        args.train_manifest,
        data_root=data_root,
        categories=vocab,
        prompt_mode="classification_reasoning",
        category_ignore_on_negative=ignore_neg,
        system_prompt=system_prompt,
    )
    if len(train_dataset) == 0:
        raise SystemExit("Train manifest is empty.")
    loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda rows: rows[0],
    )

    optimizer = build_optimizer(model, args.learning_rate, args.head_learning_rate)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    extra = {
        "stage": "grpo",
        "sft_adapter": args.sft_adapter,
        "categories": list(vocab.names),
        "system_prompt": system_prompt,
        "reward": {
            "alpha": reward_weights.alpha,
            "beta": reward_weights.beta,
            "gamma": reward_weights.gamma,
            "delta": reward_weights.delta,
        },
    }

    step = 0
    accum = 0
    optimizer.zero_grad(set_to_none=True)
    running = []
    iterator = iter(loader)
    max_steps = int(args.max_steps)
    report(phase="train", message="GRPO training started", step=0, max_steps=max_steps)

    while step < max_steps:
        try:
            example = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            example = next(iterator)

        gold_binary, gold_category, score_category = gold_fields(
            example, vocab, ignore_neg
        )
        user_text = example.get("user_text") or ""
        gold_rationale = example.get("rationale") or example.get("assistant_text") or ""
        prompt_messages = build_chat_messages(
            example["image"],
            user_text,
            system_prompt=system_prompt,
        )
        prompt_inputs = encode_messages(
            processor, prompt_messages, device, add_generation_prompt=True
        )
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        if prompt_len >= args.max_length - 8:
            continue

        model.eval()
        model.config.use_cache = True
        sequences = []
        texts = []
        with torch.no_grad():
            for _ in range(int(args.group_size)):
                generated = model.generate(
                    **prompt_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=max(args.temperature, 1e-5),
                    use_cache=True,
                )
                sequences.append(generated)
                texts.append(
                    processor.batch_decode(
                        generated[:, prompt_len:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()
                )
        model.train()
        model.config.use_cache = False

        group_logp = []
        group_logp_ref = []
        group_mask = []
        group_head_loss = []
        rewards = []
        reward_parts = []

        for full_ids, rationale in zip(sequences, texts):
            attn = torch.ones_like(full_ids)
            model_inputs = dict(prompt_inputs)
            model_inputs["input_ids"] = full_ids
            model_inputs["attention_mask"] = attn
            outputs = model(**model_inputs, output_hidden_states=True)
            hidden = last_hidden_state(outputs)
            last_idx = torch.tensor(
                [full_ids.shape[1] - 1],
                device=hidden.device,
                dtype=torch.long,
            )
            pooled = pool_hidden_states(hidden, head_token_index=last_idx)
            heads = heads_module(model)
            head_param = next(heads.parameters())
            pooled = pooled.to(device=head_param.device, dtype=head_param.dtype)
            binary_logits, category_logits = heads(pooled)
            pred_binary = int(binary_logits.argmax(dim=-1).item())
            pred_category = vocab.decode(int(category_logits.argmax(dim=-1).item()))

            scored = compute_reward(
                rationale=rationale,
                gold_rationale=gold_rationale,
                gold_binary=gold_binary,
                pred_binary=pred_binary,
                gold_category=gold_category,
                pred_category=pred_category,
                score_category=score_category,
                category_names=vocab.names,
                category_keywords=category_keywords,
                weights=reward_weights,
            )
            rewards.append(scored["reward"])
            reward_parts.append(scored)

            token_logp, mask = completion_token_logprobs(
                outputs.logits,
                full_ids,
                prompt_len,
                attention_mask=attn,
            )
            group_logp.append(token_logp)
            group_mask.append(mask)

            binary_t = None
            category_t = None
            if gold_binary is not None:
                binary_t = torch.tensor([gold_binary], device=binary_logits.device)
            if score_category and int(example.get("category_label", HEAD_IGNORE_INDEX)) != HEAD_IGNORE_INDEX:
                category_t = torch.tensor(
                    [int(example["category_label"])],
                    device=category_logits.device,
                )
            loss_b, loss_c = head_losses(
                binary_logits, category_logits, binary_t, category_t
            )
            group_head_loss.append(
                args.head_ce_alpha * loss_b + args.head_ce_beta * loss_c
            )

            current = apply_lora_snapshot(model, ref_weights)
            with torch.no_grad():
                ref_out = model(**model_inputs)
            apply_lora_snapshot(model, current)
            token_logp_ref, mask_ref = completion_token_logprobs(
                ref_out.logits,
                full_ids,
                prompt_len,
                attention_mask=attn,
            )
            group_logp_ref.append(token_logp_ref)
            group_mask[-1] = group_mask[-1] * mask_ref

        max_t = max(x.shape[1] for x in group_logp)
        batch = len(group_logp)

        def pad_row(row, width):
            if row.shape[1] == width:
                return row
            pad = row.new_zeros(row.shape[0], width - row.shape[1])
            return torch.cat([row, pad], dim=1)

        token_logp = torch.cat([pad_row(x, max_t) for x in group_logp], dim=0)
        token_logp_ref = torch.cat([pad_row(x, max_t) for x in group_logp_ref], dim=0)
        mask = torch.cat([pad_row(x, max_t) for x in group_mask], dim=0)
        reward_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        advantages = group_advantages(reward_t)
        loss_pg, stats = grpo_loss(
            token_logp,
            token_logp_ref,
            mask,
            advantages,
            args.kl_beta,
        )
        loss_heads = torch.stack(group_head_loss).mean()
        loss = (loss_pg + loss_heads) / max(int(args.grad_accum), 1)
        loss.backward()
        accum += 1

        running.append({
            "reward": float(reward_t.mean()),
            "format": sum(p["format"] for p in reward_parts) / batch,
            "decision": sum(p["decision"] for p in reward_parts) / batch,
            "category": sum(p["category"] for p in reward_parts) / batch,
            "keywords": sum(p["keywords"] for p in reward_parts) / batch,
            "kl": stats["kl"],
            "loss": float(loss.detach()) * max(int(args.grad_accum), 1),
        })

        if accum % max(int(args.grad_accum), 1) == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % max(int(args.logging_steps), 1) == 0 or step == 1:
                window = running[-int(args.logging_steps) :]
                avg = {k: sum(row[k] for row in window) / len(window) for k in window[0]}
                print(
                    f"step {step}/{max_steps}  "
                    f"R={avg['reward']:.3f}  "
                    f"fmt={avg['format']:.2f}  "
                    f"dec={avg['decision']:.2f}  "
                    f"cat={avg['category']:.2f}  "
                    f"kw={avg['keywords']:.2f}  "
                    f"kl={avg['kl']:.4f}  "
                    f"loss={avg['loss']:.4f}"
                )
                report(
                    phase="train",
                    message="GRPO",
                    step=step,
                    max_steps=max_steps,
                    percent=round(100.0 * step / max(max_steps, 1), 2),
                    reward=avg["reward"],
                    kl=avg["kl"],
                    loss=avg["loss"],
                )
            if args.save_steps and step % int(args.save_steps) == 0:
                ckpt = Path(args.output_dir) / f"checkpoint-{step}"
                save_policy(model, processor, ckpt, vocab, extra, system_prompt)
                print("Saved", ckpt)

    final_dir = os.path.join(args.output_dir, "final_adapter")
    save_policy(model, processor, final_dir, vocab, extra, system_prompt)
    report(phase="save", message=f"saved GRPO adapter: {final_dir}", percent=100)
    print("\nSaved GRPO adapter and classification heads:")
    print(final_dir)


if __name__ == "__main__":
    main()
