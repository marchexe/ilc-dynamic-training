# train8m provenance and split verification

Specification: `20250711_ilc_nnqq_sgv_10m_3cat_train8m_parquet_v1`; verified 2026-10-12T23:18:31.254885+00:00. Read-only; no conversion.

| Flavour | Label | train8m rows | Unique scalar records | train800k prefix | val1000k overlap | Legacy val50k overlap | Legacy val5k overlap |
|---|---:|---:|---:|---|---:|---:|---:|
| bb | 5 | 7,989,060 | 7,989,060 | exact (798,977) | 0 | 49,928/49,928 | 4,995/4,995 |
| cc | 4 | 7,976,224 | 7,976,224 | exact (797,569) | 0 | 49,861/49,861 | 4,980/4,980 |
| dd | 2 | 7,956,165 | 7,956,165 | exact (795,686) | 0 | 49,729/49,729 | 4,970/4,970 |

Totals: train8m 23,921,449; train800k 2,392,232; val1000k 2,990,125; train8m/val1000k scalar overlap 0.

## Entry windows from the source split macros

- `train8m`: `Entry$<8000000` (divide_energy_10m.C)
- `val500k`: `Entry$>=8000000&&Entry$<8500000` (divide_energy_10m.C)
- `test1500k`: `Entry$>=8500000&&Entry$<10000000` (divide_energy_10m.C)
- `train16m`: `Entry$<16000000` (divide_energy_20m.C)
- `train16m`: `Entry$<16000000` (divide_energy_20m_bb.C)
- `train16m`: `Entry$<16000000` (divide_energy_20m_exb.C)
- `train16m`: `Entry$<16000000` (divide_energy_20m_onlyb.C)
- `val1000k`: `Entry$>=16000000&&Entry$<17000000` (divide_energy_20m.C)
- `val1000k`: `Entry$>=16000000&&Entry$<17000000` (divide_energy_20m_bb.C)
- `val1000k`: `Entry$>=16000000&&Entry$<17000000` (divide_energy_20m_exb.C)
- `val1000k`: `Entry$>=16000000&&Entry$<17000000` (divide_energy_20m_onlyb.C)
- `test3000k`: `Entry$>=17000000&&Entry$<20000000` (divide_energy_20m.C)
- `test3000k`: `Entry$>=17000000&&Entry$<20000000` (divide_energy_20m_bb.C)
- `test3000k`: `Entry$>=17000000&&Entry$<20000000` (divide_energy_20m_exb.C)
- `test3000k`: `Entry$>=17000000&&Entry$<20000000` (divide_energy_20m_onlyb.C)
- `train80k`: `Entry$<80000` (divide_energy_100k.C)
- `val5k`: `Entry$>=80000&&Entry$<85000` (divide_energy_100k.C)
- `test15k`: `Entry$>=85000&&Entry$<100000` (divide_energy_100k.C)

No macro for the original `train800k`/`val50k`/`test150k` ladder is readable in the source directory; their placement is established from content: train800k is the ordered prefix of train8m, and every legacy val50k/val5k record also appears in train8m.

## Consequences

- train8m (Entry$ < 8M of each flavour file) and val1000k (16M-17M) are disjoint windows; no identical scalar record is shared. val50k_tail, val5k_tail and val_holdout are exact windows of val1000k (Stage 2A verified their content and order).
- Legacy `val5k`/`val50k` lie inside train8m and are rejected for train8m selection. `val_holdout`/`val1000k` are reserved for independent evaluation.
- All production train800k rows are contained in train8m (as its first rows per flavour).
- Historical note (outside Stage 2B): legacy `val5k` is also inside train800k (14,945 shared records), legacy `val50k` is disjoint from train800k (0 shared). Current production configurations select on `val50k_tail`.
- ROOT entries are jets, not physical events. No event identifier is stored (`event_nbh` is 0), so event-level lineage cannot be verified; contiguous Entry$ windows keep jets of one event together only if the source stores them in adjacent entries (unverifiable assumption).
- Identity columns: label, jet_e, jet_px, jet_py, jet_pz, jet_phi, jet_theta.
