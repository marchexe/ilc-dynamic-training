"""Minimal documentation build; never import training code or inspect runs."""

project = "Adaptive Learning-Rate Training for ILC Jet-Flavour Tagging"
extensions = [
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
]
extlinks = {"repo": ("https://github.com/marchexe/ilc-dynamic-training/blob/main/%s", "%s")}
root_doc = "index"
source_suffix = ".rst"
exclude_patterns = ["_build", ".venv", ".venv-*", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/marchexe/ilc-dynamic-training/",
    "source_branch": "main",
    "source_directory": "docs/",
}
html_title = project
html_static_path = ["_static"]
html_extra_path = ["_extra"]
