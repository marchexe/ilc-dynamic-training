import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import platform
from importlib.metadata import version

from training.runtime import PROJECT_DIR, sha256, utc_now

PREPARED = PROJECT_DIR / 'configs/prepared/stage2a_holdout_v1'
OUTPUT = PROJECT_DIR / 'runs/eval/stage2a_holdout_v1'
PRIMARY = 'validation_total_reference_mistag_geomean_percent'


def write_json(path, value):
    """Unique staging file supports concurrent, atomic per-checkpoint writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def environment():
    return dict(python=platform.python_version(),packages={name:version(name) for name in
                ('torch','numpy','awkward','pyarrow','scipy','scikit-learn','matplotlib')})


def file_record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path), size_bytes=path.stat().st_size)


def verify_file(record):
    path = Path(record['path'])
    if not path.is_file() or path.stat().st_size != record['size_bytes'] or sha256(path) != record['sha256']:
        raise ValueError(f'Frozen input changed or disappeared: {path}')


def lock_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    return stream


def log(path, message):
    line = f'{utc_now()} {message}'
    print(line, flush=True)
    with Path(path).open('a') as stream:
        stream.write(line + '\n')
