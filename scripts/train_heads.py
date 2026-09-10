# ============================================================
# train_heads.py
#
# Qwen3-VL-8B-Instruct QLoRA with three parallel outputs:
#   token prediction → rationale that supports the decisions
#   binary head      → yes/no (0/1)
#   category head    → K classes from the train config (e.g. 3 VE labels)
# ============================================================

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from src.config_io import (
    load_experiment_config,
    overlay_cli,
    render_system_prompt,
    vocab_from_config,
)
from src.data_utils import (
    ManifestDataset,
    Qwen3VLCollator,
)
from src.heads import (
    attach_classification_heads,
    combine_losses,
    head_losses,
    heads_module,
    last_hidden_state,
    load_classification_heads,
    pool_hidden_states,
    save_classification_heads,
)
from src.profiler import EfficiencyCallback
from src.progress import report

class DriveProgressCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        step = state.global_step or 0
        max_steps = state.max_steps or 0
        percent = round(100.0 * step / max_steps, 2) if max_steps else 0
        report(
            phase="train",
            message="training started" if step == 0 else f"resumed from step {step}",
            step=step,
            max_steps=max_steps,
            epoch=state.epoch or 0,
            percent=percent,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        max_steps = state.max_steps or 0
        step = state.global_step or 0
        percent = round(100.0 * step / max_steps, 2) if max_steps else None
        report(
            phase="train",
            message="training",
            step=step,
            max_steps=max_steps,
            epoch=logs.get("epoch", state.epoch),
            loss=logs.get("loss"),
            loss_class=logs.get("loss_class"),
            loss_category=logs.get("loss_category"),
            loss_log_prob=logs.get("loss_log_prob"),
            log_prob=logs.get("log_prob"),
            learning_rate=logs.get("learning_rate"),
            eval_loss=logs.get("eval_loss"),
            percent=percent,
        )

    def on_train_end(self, args, state, control, **kwargs):
        report(phase="train", message="training finished", percent=100)


def build_training_args(**kwargs):
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in params:
        kwargs.pop("warmup_steps", None)
    else:
        kwargs.pop("warmup_ratio", None)
    if "eval_strategy" in params:
        kwargs.pop("evaluation_strategy", None)
    else:
        kwargs.pop("eval_strategy", None)
    kwargs = {k: v for k, v in kwargs.items() if k in params and v is not None}
    return TrainingArguments(**kwargs)


class MultiTaskTrainer(Trainer):
    """L = alpha * L_class + beta * L_category + gamma * L_log_prob."""

    def __init__(
        self,
        *args,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        head_save_extra=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.head_save_extra = head_save_extra or {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        binary_labels = inputs.pop("binary_labels", None)
        category_labels = inputs.pop("category_labels", None)
        head_token_index = inputs.pop("head_token_index", None)

        outputs = model(**inputs, output_hidden_states=True)
        hidden = last_hidden_state(outputs)
        pooled = pool_hidden_states(
            hidden,
            head_token_index=head_token_index,
            attention_mask=inputs.get("attention_mask"),
        )

        heads = heads_module(model)
        pooled = pooled.to(device=next(heads.parameters()).device)
        binary_logits, category_logits = heads(pooled.to(dtype=next(heads.parameters()).dtype))
        loss_class, loss_category = head_losses(
            binary_logits,
            category_logits,
            binary_labels,
            category_labels,
        )
        loss_log_prob = outputs.loss
        loss = combine_losses(
            loss_class,
            loss_category,
            loss_log_prob,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )

        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            nll = float(loss_log_prob.detach()) if loss_log_prob is not None else 0.0
            self.log({
                "loss_class": float(loss_class.detach()),
                "loss_category": float(loss_category.detach()),
                "loss_log_prob": nll,
                "log_prob": -nll,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            })

        outputs.loss = loss
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call=_internal_call)
        output_dir = output_dir or self.args.output_dir
        save_classification_heads(
            self.model,
            Path(output_dir) / "classification_heads.pt",
            extra=self.head_save_extra,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "workspace" / "train_heads.json"),
        help="Train/eval hyperparameters and category names.",
    )

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--eval_manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root that contains images. Defaults to <repo>/data.",
    )
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument(
        "--categories_file",
        type=str,
        default=str(ROOT / "workspace" / "categories.json"),
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated positive class names. Overrides --categories_file.",
    )
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight on binary class CE: L = alpha*class + beta*category + gamma*log_prob",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight on category CE (skipped on negatives unless config says otherwise).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Weight on rationale token NLL (-mean log p).",
    )
    parser.add_argument("--num_categories", type=int, default=None)
    parser.add_argument(
        "--category_ignore_on_negative",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Path to system_prompt.md. Defaults to workspace/system_prompt.md.",
    )
    return parser.parse_args()


def default_data_root():
    return str(ROOT / "data")


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    overlay_cli(args, cfg)
    if not args.train_manifest or not args.eval_manifest or not args.output_dir:
        raise SystemExit("train_manifest, eval_manifest, and output_dir are required.")
    data_root = args.data_root or default_data_root()
    vocab = vocab_from_config(cfg, args)
    ignore_neg = args.category_ignore_on_negative
    if ignore_neg is None:
        ignore_neg = cfg.get("category_ignore_on_negative", True)
    system_prompt_path = args.system_prompt or cfg.get("system_prompt")
    system_prompt = render_system_prompt(system_prompt_path, vocab.names)
    print("Categories:", list(vocab.names))
    print(f"num_categories={len(vocab)}  ignore_category_on_negative={ignore_neg}")
    print(f"system_prompt: {system_prompt_path or 'workspace/system_prompt.md'}")
    print(f"L = {args.alpha}*class + {args.beta}*category + {args.gamma}*log_prob")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    report(phase="setup", message="loading processor and 4-bit model")
    processor = AutoProcessor.from_pretrained(args.model_name)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and isinstance(getattr(image_processor, "size", None), dict):
        image_processor.size["longest_edge"] = min(
            int(image_processor.size.get("longest_edge", 1024 * 1024)),
            1024 * 1024,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model.config.use_cache = False

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank * 2
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    attach_classification_heads(
        model,
        num_categories=len(vocab),
        dropout=args.head_dropout,
        dtype=torch.float32,
    )

    print("\nTrainable parameters:")
    model.print_trainable_parameters()
    head_params = sum(p.numel() for p in heads_module(model).parameters())
    print(f"Classification heads: {head_params:,} parameters")

    report(phase="data", message="loading train manifest")
    train_dataset = ManifestDataset(
        args.train_manifest,
        data_root=data_root,
        categories=vocab,
        prompt_mode="classification_reasoning",
        category_ignore_on_negative=ignore_neg,
        system_prompt=system_prompt,
    )
    report(
        phase="data",
        message=f"train examples: {len(train_dataset):,}",
        train_examples=len(train_dataset),
    )

    report(phase="data", message="loading eval manifest")
    eval_dataset = ManifestDataset(
        args.eval_manifest,
        data_root=data_root,
        categories=vocab,
        prompt_mode="classification_reasoning",
        category_ignore_on_negative=ignore_neg,
        system_prompt=system_prompt,
    )
    report(
        phase="data",
        message=f"eval examples: {len(eval_dataset):,}",
        eval_examples=len(eval_dataset),
    )

    collator = Qwen3VLCollator(
        processor,
        max_length=args.max_length,
        include_classification_heads=True,
        system_prompt=system_prompt,
    )

    training_args = build_training_args(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        warmup_steps=0.03,
        weight_decay=0.01,
        logging_steps=args.logging_steps or 10,
        logging_first_step=True,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_total_limit=None,
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
    )

    head_save_extra = {
        "categories": list(vocab.names),
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "loss": "alpha*class + beta*category + gamma*log_prob",
        "system_prompt": system_prompt,
        "system_prompt_file": str(system_prompt_path or "workspace/system_prompt.md"),
    }

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        head_save_extra=head_save_extra,
        callbacks=[
            DriveProgressCallback(),
            EfficiencyCallback(
                os.path.join(args.output_dir, "efficiency_train.json"),
                method="qlora_heads",
            ),
        ],
    )

    last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint:
        heads_path = Path(last_checkpoint) / "classification_heads.pt"
        if heads_path.exists():
            load_classification_heads(model, heads_path)
        print("Resuming from", last_checkpoint)
        report(phase="train", message=f"resuming from {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    save_classification_heads(
        trainer.model,
        Path(final_dir) / "classification_heads.pt",
        extra=head_save_extra,
    )
    (Path(final_dir) / "categories.json").write_text(
        json.dumps({"categories": list(vocab.names)}, indent=2),
        encoding="utf-8",
    )
    (Path(final_dir) / "system_prompt.md").write_text(
        system_prompt + "\n",
        encoding="utf-8",
    )

    report(phase="save", message=f"saved adapter+heads: {final_dir}", percent=100)
    print("\nSaved final adapter and classification heads:")
    print(final_dir)


if __name__ == "__main__":
    main()
