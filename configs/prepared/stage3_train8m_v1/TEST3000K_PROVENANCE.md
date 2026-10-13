# Sealed test split (test3000k) provenance

Verified 2026-10-13T06:59:51.921914+00:00; read-only. Window `Entry$>=17000000&&Entry$<20000000` of each flavour file (train8m: Entry$ < 8M; val1000k and the Stage 2A holdout: 16M-17M).

| Flavour | Label | Rows | Unique scalar records | train8m | val1000k | pretraining train800k | pretraining val50k |
|---|---:|---:|---:|---:|---:|---:|---:|
| bb | 5 | 2,995,907 | 2,995,907 | 0 | 0 | 0 | 0 |
| cc | 4 | 2,991,086 | 2,991,086 | 0 | 0 | 0 | 0 |
| dd | 2 | 2,983,484 | 2,983,484 | 0 | 0 | 0 | 0 |

Total 8,970,477 rows; zero identical scalar records with any training, selection or pretraining input. No configuration, script, manifest or run of this repository references test3000k before Stage 3 (checked by search). Use by other researchers cannot be verified.
No physical event ID exists, so event-level independence is not provable.
