"""Minimal documentation build; never import training code or inspect runs."""

project = "Adaptive Learning-Rate Training for ILC Jet-Flavour Tagging"
extensions = [
    "sphinx.ext.githubpages",
]
root_doc = "index"
source_suffix = ".rst"
exclude_patterns = ["_build", ".venv", ".venv-*", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
html_title = project
html_static_path = ["_static"]
