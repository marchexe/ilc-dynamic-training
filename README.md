# Adaptive Learning-Rate Training for ILC Jet-Flavour Tagging

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://marchexe.github.io/ilc-dynamic-training/)

Does learning-rate adaptation with Population Based Training improve a pretrained
Particle Transformer b/c/light-jet tagger over constant learning rates? Completed
experiments show a gain of about 1–2% in composite mistag on one seed; a preregistered
comparison on ten times more data is running.

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
