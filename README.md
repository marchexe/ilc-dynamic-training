# Adaptive Learning-Rate Training for ILC Jet-Flavour Tagging

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://marchexe.github.io/ilc-dynamic-training/)

Does learning-rate adaptation with Population Based Training (PBT) improve a pretrained
Particle Transformer b/c/light-jet tagger over standard ParT training? With three training
seeds and the same compute, PBT lowers the composite mistag on an independent 2.8M-jet
test sample, Test (2.8M), by 0.67%, 0.85% and 0.91% (0.81% on average; 95% interval 0.34–1.34% for the
Test (2.8M) sample). On ten times more training data (8M jets, one seed) PBT is 12.4% worse than
standard ParT, because it lowers the learning rate too early; cool-down probes and a final
decay reduce this to 2.7%. The model overfits at 800k but not at 8M, which explains the
difference. A prediction ensemble of two separately trained 8M standard ParT models is 5.0% better than
the better single model on each of three training seeds. Earlier gains of 1–2% and 7.7% were
measured against fixed learning rates that were too low.

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
