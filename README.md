# Adaptive Learning-Rate Training for ILC Jet-Flavour Tagging

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://marchexe.github.io/ilc-dynamic-training/)

This repository studies whether learning-rate adaptation during training improves a
Particle Transformer that separates b, c and light-quark jets in simulated ILC
events. A pretrained three-class tagger is fine-tuned under matched conditions with
constant per-member learning rates or with Population Based Training (PBT). The
results are measured as background mistag rates at fixed signal efficiencies, on the
selection validation set and on an independent holdout of 2.8 million jets.

## Findings so far

- **50 epochs, matched.** Windowed PBT beat the fixed-learning-rate population by
  1.26% on the population validation endpoint and by 1.36% on the independent holdout.
- **100 epochs.** Windowed PBT was again lower: 1.69% on validation and 1.44% on the
  holdout. The comparator here is a single fixed-learning-rate member chosen after
  epoch 50, so this comparison is asymmetric.
- **Adaptation frequency: inconclusive.** Every-epoch PBT is ahead of every-five-epochs
  PBT on validation for both seeds, but the order reverses on the holdout.
- **Limits.** One training seed per policy comparison and no event-level confidence
  intervals. The holdout was inspected in earlier diagnostics. All runs share an
  epoch-restarted optimizer contract.
- **Ongoing.** A preregistered comparison on a ten-times larger training set with a
  sealed test set is running. No results from it are reported yet.

## Documentation

- [Overview](https://marchexe.github.io/ilc-dynamic-training/)
- [Methodology](https://marchexe.github.io/ilc-dynamic-training/method.html)
- [Experimental design and evaluation](https://marchexe.github.io/ilc-dynamic-training/evaluation.html)
- [Results and limitations](https://marchexe.github.io/ilc-dynamic-training/results.html)
- [Ongoing research](https://marchexe.github.io/ilc-dynamic-training/ongoing.html)
- [Reproducibility and experiment records](https://marchexe.github.io/ilc-dynamic-training/reproducibility.html)

The documentation sources are in `docs/` (Sphinx). Curated results, figures and
experiment bundles are in `published/`.

## Getting started

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -c requirements-lock.txt
cp configs/site.example.json configs/local/site.json   # machine-specific settings, not committed
.venv/bin/python -m unittest discover -s tests -t .
```

Datasets, checkpoints and run directories are not tracked; their identities are
recorded in `datasets/manifests/` and in the frozen protocols under
`configs/prepared/`. See
[Reproducibility](https://marchexe.github.io/ilc-dynamic-training/reproducibility.html)
for running, verifying and publishing experiments.

## Repository layout

```text
configs/      experiment configurations and frozen study protocols (configs/prepared/)
networks/     Particle Transformer model
scripts/      training engine, verification, campaign launchers, publishing
published/    curated results, figures and experiment bundles
docs/         documentation sources
weaver-core/  vendored and patched Weaver framework
tests/        unit, integration and provenance tests
```
