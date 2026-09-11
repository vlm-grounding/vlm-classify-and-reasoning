# Classify and Reason with Qwen3-VL-8B

This project is [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) plus two classification heads and a language rationale with the training framework for QLoRA and GRPO with a customized reward design:

1. **Binary classification** — `no` / `yes`
2. **Categories under the positive class** — if yes, pick one of K classes from the config
3. **Language rationale / reasoning** — the model writes why those decisions are correct

The mock task is e-SNLI-VE (1000 train / 100 eval). The same setup is for later use such as “is there a finding?” and, if yes, which finding, with a written explanation.

**License:** [PolyForm Noncommercial 1.0.0](LICENSE) — you may use, change, and redistribute this software for noncommercial purposes. Commercial use is not permitted.

---

## 1. Model architecture

Qwen3-VL-8B is loaded in 4-bit. LoRA is applied to the usual projection layers. Two linear heads sit on the **same** last-prompt hidden state. The LM head is used only for the rationale tokens.

```text
  system instruction + image + user question
                    │
                    ▼
         Qwen3-VL-8B-Instruct (4-bit QLoRA)
                    │
         last prompt hidden state
           ┌────────┼────────┐
           ▼        ▼        ▼
      binary     category    tokens
     no / yes    K classes   rationale
                  (if yes)
```

**Task structure**

```text
binary = no  →  stop; do not use the category
binary = yes →  category ∈ {class_1, …, class_K}
rationale    →  language that supports the binary (and category) decision
```

The category list is **not** `{no, class_1, …, class_K}`. `no` is only on the binary head. Categories exist under the positive class.

**Chat (standard LoRA SFT)**

| Role | Content |
|---|---|
| `system` | `workspace/system_prompt.md` (`{categories}` filled from config) |
| `user` | image + the instance question |
| `assistant` | rationale / reasoning only |

The assistant target is never `yes`, `no`, or a class name. Those come from the heads.

**Loss**

```text
L = α · L_binary + β · L_category + γ · L_rationale
```

- `L_binary` — 2-way CE (`no` / `yes`)
- `L_category` — K-way CE on positive rows (skipped on `no` when `category_ignore_on_negative` is true)
- `L_rationale` — token NLL of the written reasoning (`-mean log p`)

α, β, γ, K, lr, and LoRA settings are in `workspace/train_heads.json`.

The mock e-SNLI-VE split maps `entailment → yes` and trains all three VE labels on the category head (`category_ignore_on_negative: false`) so the smoke test has three real classes. A findings-style dataset should keep `category_ignore_on_negative: true`: binary first, categories only when positive.

**Stage 2 GRPO reward** (not SFT). After each sampled rationale, heads are read at the last generated token:

```text
R = α · format + β · decision + γ · category + δ · keywords
keywords = ½ · dictionary(gold category) + ½ · overlap(gold rationale)
```

- `format` — 1 if the text is a real rationale (not empty, not just `yes`/`no`/a class name)
- `decision` — 1 if the binary head matches gold, else 0
- `category` — 1 if the category head matches gold (0 when that row should not score K-way)
- `keywords` — (a) hits from `DEFAULT_CATEGORY_KEYWORDS` in `src/grpo_reward.py` (optional overlay: `workspace/category_keywords.json`) for the gold class, plus (b) token F1 vs the gold rationale after dropping `the`, `a`, `an`, and other stop words. Class names are also dropped from the overlap so the model is not paid for saying the label.

Weights are in `workspace/train_grpo.json` → `reward`. They are separate from the Stage 1 SFT α/β/γ.

---

## 2. What this repo implements

| Piece | What it is |
|---|---|
| Backbone | Qwen3-VL-8B-Instruct, 4-bit |
| Adaptation | Standard LoRA / QLoRA (not gated LoRA) |
| Binary head | Linear → 2, last-prompt hidden state |
| Category head | Linear → K, same vector; K classes under **yes** |
| Reasoning | LoRA language modeling of the rationale |
| Stage 2 | GRPO on sampled rationales (`scripts/train_grpo.py`) |
| Config | Category names, K, SFT α/β/γ, GRPO reward α/β/γ/δ, system prompt, lr, LoRA |

What you edit for a new dataset: the jsonl fields `binary`, `category`, `rationale`, plus the train/eval JSON files and `workspace/system_prompt.md`. You should not need to change `src/heads.py`. For a new class set, update `DEFAULT_CATEGORY_KEYWORDS` in `src/grpo_reward.py` (or overlay `workspace/category_keywords.json`).

---

## 3. How to use the codebase

### Layout

```text
src/heads.py                 binary + category heads and the combined loss
src/grpo.py                  group advantages and GRPO loss
src/grpo_reward.py           reward + DEFAULT_CATEGORY_KEYWORDS
src/data_utils.py            manifests, images, SFT collator
src/config_io.py             train/eval JSON and system_prompt.md
scripts/train_heads.py       Stage 1 QLoRA + heads
scripts/train_grpo.py        Stage 2 GRPO from the SFT adapter
scripts/eval_heads.py        head accuracy + generated rationale
scripts/prepare_mock_esnli_ve.py
scripts/test_cpu.py
scripts/ctl.py               queue Colab jobs
workplace/colab_worker.ipynb
workspace/train_heads.json
workspace/eval_heads.json
workspace/train_grpo.json
workspace/eval_grpo.json
workspace/system_prompt.md
workspace/category_keywords.json
data/manifests/              1000 train + 100 eval
data/esnli_ve/images/        synthetic MOCK JPEGs (not Flickr30k)
```

`train_qlora.py` / `eval_qlora.py` are leftover VQA scripts without these heads. Use `*_heads.py`.

### Config

| File | Role |
|---|---|
| `workspace/train_heads.json` | Stage 1 SFT: same mock manifests, `system_prompt`, categories, loss α/β/γ, lr, LoRA |
| `workspace/eval_heads.json` | Stage 1 eval of the SFT adapter |
| `workspace/train_grpo.json` | Stage 2 GRPO: **same** `mock_train.jsonl` / `mock_eval.jsonl`, init from `sft_adapter` |
| `workspace/eval_grpo.json` | Stage 2 eval: `sft_adapter` (baseline) **and** GRPO `lora_path` |
| `workspace/system_prompt.md` | system instruction; keep `{categories}` |
| `workspace/category_keywords.json` | optional overlay of the dict in `src/grpo_reward.py` |

### CPU check (no 8B)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pillow
python scripts/ctl.py cpu-test
```

### Mock examples

| Split | Path | Rows |
|---|---|---|
| train | `data/manifests/mock_train.jsonl` | 1000 |
| eval | `data/manifests/mock_eval.jsonl` | 100 |

Each row has `image`, `question`, `binary`, `category`, `rationale`. Images are generated `MOCK` tiles. Rebuild them with:

```powershell
python scripts/prepare_mock_esnli_ve.py --from-manifests
```

Do not commit Flickr30k photos. See `data/README.md`.

### Train / eval mock (needs GPU)

`max_steps: 30` is a smoke run, not a full epoch.

```powershell
python scripts/train_heads.py --config workspace/train_heads.json
python scripts/eval_heads.py --config workspace/eval_heads.json
```

Writes `checkpoints/mock_heads/final_adapter/` and `results/mock_eval_heads.jsonl`.

Stage 2 uses the **same** mock jsonl. Init from the SFT adapter. Eval scores SFT and GRPO:

```powershell
python scripts/train_grpo.py --config workspace/train_grpo.json
python scripts/eval_heads.py --config workspace/eval_grpo.json
```

`eval_grpo.json` sets `sft_adapter` → `results/mock_eval_sft.jsonl` and GRPO `lora_path` → `results/mock_eval_grpo.jsonl`.

```powershell
python scripts/ctl.py train-grpo
python scripts/ctl.py eval-grpo
```

### Colab A100 (no local GPU)

1. Put the repo on Drive as `classify_and_reasoning_qwen3-vlm8b`.
2. Open `workplace/colab_worker.ipynb`, A100, Run all, leave it running.
3. Locally:

```powershell
python scripts/ctl.py status
python scripts/ctl.py train-heads
python scripts/ctl.py eval-heads
python scripts/ctl.py train-grpo
python scripts/ctl.py eval-grpo
python scripts/ctl.py watch
```

### New dataset

Manifest fields: `image`, `binary` (`0`/`1`), `category` (one of the configured names), `rationale`, optional `question`.

Findings-style (binary first, categories only if positive):

```json
"categories": ["atelectasis", "cardiomegaly", "edema"],
"num_categories": 3,
"category_ignore_on_negative": true
```

### Do not commit

`models/`, `checkpoints/`, `results/`, job logs, Flickr downloads, adapter weights. The mock jsonl and synthetic JPEGs are the exception.

---

## License

Copyright 2026 vlm-grounding.

This software is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0). You may use, modify, and redistribute it for **noncommercial** purposes (including research, personal study, and use by educational or other noncommercial organizations). **Commercial use is not permitted.**

See [LICENSE](LICENSE) for the full terms. Required Notice: Copyright 2026 Qinwu Xu.
