# Datasets

Large local datasets are not committed. Track lightweight dataset manifests under `datasets/manifests/` so runs can declare the expected local layout, file names, and sizes.

Tracked manifests currently describe:

- `20250218_ilc_nnqq_sgvnew_parquet.json`: compact parquet dataset used by the original guarded fine-tuning configs.
- `20250711_ilc_nnqq_sgv_10m_3cat_parquet.json`: local bb/cc/dd subset converted from the 10m ROOT source, with `train800k`, original `val5k`/`val50k`, and `val1000k` validation levels.
- `20250711_ilc_nnqq_sgv_10m_3cat_tail_proxy_v1.json`: active tail proxy split where `val5k_tail` is the control proxy from the end of `val1000k`, and `val50k_tail` is the disjoint monitor window immediately before it.
