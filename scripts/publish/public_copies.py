#!/usr/bin/env python3
"""Write path-free public copies of experiment publication files for the documentation site.

Publication bundles under published/experiments/<slug>/ are kept byte-for-byte as recorded. Some of their
text files contain absolute paths of the checkout that produced them. This tool writes copies to
published/experiments/<slug>/public/ in which that checkout prefix becomes a repository-relative path,
records source and output SHA256 in public/MANIFEST.json, and refuses to write if any absolute path
remains. Values are otherwise unchanged.

    python scripts/publish/public_copies.py --source-root <checkout that produced the bundles>
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

PROJECT_DIR = Path(__file__).resolve().parents[2]
FILES = ('config.yaml', 'metrics.csv', 'provenance.json', 'exploit_history.csv', 'summary.json',
         'lr_history.csv', 'decision_history.csv')
ABSOLUTE = re.compile(r'(?<![\w.~-])/(?:data|home|gpfs|work|scratch|mnt|tmp|Users)/')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def export(bundle, prefix):
    records = []
    for name in FILES:
        source = bundle / name
        if not source.is_file():
            continue
        data = source.read_bytes()
        public = data.replace(prefix.encode(), b'')
        if ABSOLUTE.search(public.decode(errors='replace')):
            raise ValueError(f'{source.relative_to(PROJECT_DIR)}: absolute path remains after redaction')
        target = bundle / 'public' / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(public)
        records.append(dict(file=name, source_sha256=digest(data), public_sha256=digest(public),
                            redacted_occurrences=data.count(prefix.encode())))
    manifest = dict(schema_version=1, rule='absolute checkout prefix removed; paths become repository-relative',
                    files=records)
    (bundle / 'public' / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--source-root', type=Path, default=PROJECT_DIR,
                        help='absolute checkout path recorded inside the bundles')
    args = parser.parse_args(argv)
    prefix = str(args.source_root.resolve()).rstrip('/') + '/'
    for bundle in sorted((PROJECT_DIR / 'published/experiments').iterdir()):
        if bundle.is_dir():
            records = export(bundle, prefix)
            print(f'{bundle.name}: {len(records)} files, {sum(r["redacted_occurrences"] for r in records)} paths redacted')


if __name__ == '__main__':
    main()
