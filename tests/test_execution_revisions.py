"""Revision chains: recorded changes to protocol-pinned files are accepted only as recorded."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.helpers import PROJECT_DIR
from experiments.stage2a.common import sha256, write_json
from experiments.stage3 import protocol as stage3
from validation import verify_stage2a_provenance as stage2a

STAGE3 = PROJECT_DIR / 'configs/prepared/stage3_train8m_v1'


def record(directory, name, payload):
    path = Path(directory) / name
    write_json(path, payload)
    path.with_suffix('.sha256').write_text(sha256(path) + '\n')
    return path


class Stage2ARevisionChainTest(unittest.TestCase):
    def test_later_revisions_supersede_earlier_records_of_the_same_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = record(tmp, 'administrative_revision_1.json', dict(protocol_sha256='p', changed_execution_files=[
                dict(path='a.py', current_sha256='1'), dict(path='b.py', current_sha256='1')]))
            second = record(tmp, 'administrative_revision_2.json', dict(protocol_sha256='p', changed_execution_files=[
                dict(path='a.py', current_sha256='2')]))
            self.assertEqual(stage2a.revision_paths(tmp), [first, second])
            merged = stage2a.load_revisions([first, second])
            self.assertEqual({i['path']: i['current_sha256'] for i in merged['changed_execution_files']}, {'a.py': '2', 'b.py': '1'})
            record(tmp, 'administrative_revision_3.json', dict(protocol_sha256='other', changed_execution_files=[]))
            with self.assertRaisesRegex(ValueError, 'different protocols'):
                stage2a.load_revisions(stage2a.revision_paths(tmp))
            second.write_text(second.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'was edited'):
                stage2a.load_revisions([first, second])


class Stage3ExecutionRevisionTest(unittest.TestCase):
    def test_committed_records_match_the_protocol_git_and_current_files(self):
        frozen = json.loads((STAGE3 / 'protocol_v2.json').read_text())['identity']['execution_files']
        paths = sorted(STAGE3.glob('execution_revision_*.json'), key=lambda p: int(p.stem.rsplit('_', 1)[1]))
        self.assertGreaterEqual(len(paths), 2)
        latest = {}
        for path in paths:
            self.assertEqual(sha256(path), path.with_suffix('.sha256').read_text().strip())
            revision = json.loads(path.read_text())
            self.assertEqual(revision['protocol']['sha256'], (STAGE3 / 'protocol_v2.sha256').read_text().strip())
            for item in revision['changed_execution_files']:
                self.assertEqual(item['frozen_sha256'], frozen[item['path']])
                original = subprocess.run(['git', 'show', f'{item["original_commit"]}:{item["path"]}'], cwd=PROJECT_DIR,
                                          capture_output=True, check=True).stdout
                self.assertEqual(hashlib.sha256(original).hexdigest(), item['frozen_sha256'])
                latest[item['path']] = item
        # A later record supersedes an earlier record of the same file; the latest one matches the file on disk.
        for item in latest.values():
            self.assertEqual(sha256(PROJECT_DIR / item['path']), item['current_sha256'], item['path'])

    def test_drift_accepts_only_exactly_recorded_changes(self):
        protocol = dict(identity=dict(execution_files={'a.py': 'old', 'b.py': 'same'}))
        current = dict(execution_files={'a.py': 'new', 'b.py': 'same'})
        accepted = {'a.py': dict(frozen_sha256='old', current_sha256='new')}
        self.assertEqual(stage3.drift(protocol, current, revisions={}), ['execution_files:a.py'])
        self.assertEqual(stage3.drift(protocol, current, revisions=accepted), [])
        self.assertEqual(stage3.drift(protocol, dict(execution_files={'a.py': 'other', 'b.py': 'same'}), revisions=accepted),
                         ['execution_files:a.py'])

    def test_revisions_apply_only_after_the_campaign_completed(self):
        with tempfile.TemporaryDirectory() as prepared, tempfile.TemporaryDirectory() as output:
            path = record(prepared, 'execution_revision_1.json', dict(changed_execution_files=[
                dict(path='a.py', frozen_sha256='old', current_sha256='new')]))
            self.assertEqual(stage3.execution_revisions(prepared, output), {})
            Path(output, 'status.json').write_text(json.dumps(dict(status='running')))
            self.assertEqual(stage3.execution_revisions(prepared, output), {})
            Path(output, 'status.json').write_text(json.dumps(dict(status='completed')))
            Path(output, 'report').mkdir()
            Path(output, 'report', 'report_integrity.json').write_text('{}')
            self.assertEqual(set(stage3.execution_revisions(prepared, output)), {'a.py'})
            path.write_text(path.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'edited'):
                stage3.execution_revisions(prepared, output)


if __name__ == '__main__':
    unittest.main()
