# Adaptive Learning-Rate Training for ILC Jet-Flavour Tagging

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://marchexe.github.io/ilc-dynamic-training/)

Does learning-rate adaptation with Population Based Training (PBT) improve a pretrained
Particle Transformer b/c/light-jet tagger over standard ParT training? With three training
seeds and the same compute, PBT lowers the composite mistag on an independent 2.8M-jet
holdout by 0.67%, 0.85% and 0.91% (0.81% on average; 95% interval 0.34–1.34% for the
holdout sample). The same comparison on ten times more training data is running. Earlier
gains of 1–2% and 7.7% were measured against fixed learning rates that were too low.

- [Results and limitations](https://marchexe.github.io/ilc-dynamic-training/results.html)
- [Methodology](https://marchexe.github.io/ilc-dynamic-training/method.html) ·
  [Evaluation](https://marchexe.github.io/ilc-dynamic-training/evaluation.html) ·
  [Ongoing study](https://marchexe.github.io/ilc-dynamic-training/ongoing.html)
- [Repository map and commands](https://marchexe.github.io/ilc-dynamic-training/reproducibility.html)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -c requirements-lock.txt
cp configs/site.example.json configs/local/site.json   # machine-specific, untracked
.venv/bin/python -m unittest discover -s tests -t .
```
