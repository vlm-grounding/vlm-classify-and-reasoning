# Mock e-SNLI-VE split

Published so a clone can run the 1000/100 smoke train without extra downloads.

| Path | What it is |
|---|---|
| `manifests/mock_train.jsonl` | 1000 labeled rows |
| `manifests/mock_eval.jsonl` | 100 labeled rows |
| `esnli_ve/images/*.jpg` | **Synthetic** placeholders, not Flickr photos |

## Do not commit Flickr30k

e-SNLI-VE points at Flickr30k filenames. Those photos are **not** redistributable on GitHub. The JPEGs in this folder are generated color tiles stamped `MOCK`. They keep the original Flickr IDs only so paths stay stable.

To rebuild placeholders from the committed manifests:

```powershell
python scripts/prepare_mock_esnli_ve.py --from-manifests
```

`--try-real-images` is for private local use only. Never add those files to git.

## Labels

- `category`: `entailment` / `neutral` / `contradiction`
- `binary`: 1 if entailment, else 0
- `rationale`: `Explanation_1` from [e-SNLI-VE](https://huggingface.co/datasets/sedrickkeh/e-snli-ve)

Please cite e-SNLI-VE / e-SNLI if you use the text fields.
