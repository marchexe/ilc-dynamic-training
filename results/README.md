# Matched-seed smoke test

Both runs used seed `12345`, 600 optimizer steps, 153,600 training examples,
and 14,848 validation examples. Train and validation worker seeds matched.

| metric | fixed LR | LinUCB |
|---|---:|---:|
| validation accuracy | 0.81964 | 0.81782 |
| validation ROC AUC | 0.93598 | 0.93886 |

The result verifies the controller integration. It is one short run and does
not establish a performance improvement.

The figure was generated with `scripts/plot_training_control.py`.

The compact source data are committed under `matched_seed_12345/`:

- `baseline/metrics.json` — fixed-LR summary;
- `active/metrics.json` — LinUCB summary;
- `active/net_controller.jsonl` — all 19 controller decisions.

Rebuild the committed figure without the original training logs:

```bash
python scripts/plot_training_control.py \
  --baseline-dir results/matched_seed_12345/baseline \
  --active-dir results/matched_seed_12345/active \
  --output results/training_control_comparison_seed12345.png
```
