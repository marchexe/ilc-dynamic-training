#!/usr/bin/env python3
"""Local site configuration: cluster nodes and private data roots.

Committed code and configs stay portable; deployment details live in an
untracked JSON file, `configs/local/site.json` by default or the file named by
`ILC_SITE_CONFIG`. Copy `configs/site.example.json` to start. Standard library
only, so shell helpers can call it without the project environment.
"""
import json
import os
from pathlib import Path
import re
import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT = PROJECT_DIR / 'configs/local/site.json'
TEMPLATE = 'configs/site.example.json'
ENV = 'ILC_SITE_CONFIG'
VARIABLE = re.compile(r'\$\{([A-Z][A-Z0-9_]*)\}')


class SiteConfigError(ValueError):
    """Missing or invalid local deployment configuration."""


def path():
    return Path(os.environ.get(ENV) or DEFAULT)


def load():
    location = path()
    if not location.is_file():
        raise SiteConfigError(f'Local site configuration not found; copy {TEMPLATE} to configs/local/site.json '
                              f'(or set {ENV}) and fill in your cluster nodes and data roots')
    try:
        site = json.loads(location.read_text())
    except ValueError as error:
        raise SiteConfigError(f'Local site configuration is not valid JSON: {error}') from error
    cluster = site.get('cluster') or {}
    nodes = cluster.get('nodes')
    if (site.get('schema_version') != 1 or not isinstance(nodes, list) or not nodes
            or not all(isinstance(n, str) and n and not n.startswith('-') for n in nodes)
            or len(set(nodes)) != len(nodes) or cluster.get('preferred_node') not in nodes
            or not all(isinstance(v, str) for v in (site.get('paths') or {}).values())):
        raise SiteConfigError('Local site configuration needs schema_version 1, unique cluster.nodes, '
                              'cluster.preferred_node among them, and string paths')
    return site


def cluster():
    site = load()
    return dict(authorized_nodes=list(site['cluster']['nodes']), preferred_node=site['cluster']['preferred_node'])


def with_cluster(allocation):
    """Allocation policy with site nodes; frozen historical policies that list nodes stay as recorded."""
    if 'authorized_nodes' in allocation:
        return dict(allocation)
    return dict(allocation, **cluster())


def expand(value):
    """Resolve `${NAME}` from the environment, then from the site `paths` block."""
    text = str(value)
    names = VARIABLE.findall(text)
    if not names:
        return text
    paths = None
    for name in names:
        if name in os.environ:
            resolved = os.environ[name]
        else:
            if paths is None:
                paths = load().get('paths') or {}
            if name not in paths:
                raise SiteConfigError(f'{name} is not set in the environment or the local site configuration')
            resolved = paths[name]
        text = text.replace('${' + name + '}', resolved)
    return text


def main(argv):
    try:
        if argv == ['nodes']:
            print(' '.join(cluster()['authorized_nodes']))
        elif len(argv) == 2 and argv[0] == 'path':
            print(expand('${' + argv[1] + '}'))
        else:
            print('usage: site_config.py nodes | path NAME', file=sys.stderr)
            return 2
    except SiteConfigError as error:
        print(f'site configuration: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
