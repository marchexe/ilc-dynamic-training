"""Stage 2A administrative revision: only a recorded, Git-recoverable change is accepted."""
import copy
import json
import os
from pathlib import Path
import subprocess
import unittest

from tests.helpers import PROJECT_DIR
from validation import verify_stage2a_provenance as provenance

PREPARED = PROJECT_DIR / 'configs/prepared/stage2a_holdout_v1'


class Stage2AProvenanceRevisionTest(unittest.TestCase):
    def setUp(self):
        records = json.loads((PREPARED / 'protocol.json').read_text())['execution_files']
        self.records = [r for r in records if r['path'].endswith(('scripts/cluster/check.sh', 'stage2a/common.py'))]
        self.revision = provenance.load_revision()

    def test_committed_revision_is_pinned_and_recoverable(self):
        self.assertEqual(self.revision['protocol_sha256'], (PREPARED / 'protocol.sha256').read_text().strip())
        checked = provenance.verify_execution_files(self.records, self.revision)
        self.assertEqual({c['path']: c['state'] for c in checked},
                         {'scripts/cluster/check.sh': 'administratively_revised',
                          'scripts/experiments/stage2a/common.py': 'unchanged'})

    def test_any_other_difference_fails(self):
        original = provenance.git_blob
        with self.assertRaisesRegex(ValueError, 'not recoverable'):
            provenance.verify_execution_files(self.records, self.revision, blob=lambda c, p: original(c, p) + b'#')
        changed = copy.deepcopy(self.revision)
        changed['changed_execution_files'][0]['current_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'current content'):
            provenance.verify_execution_files(self.records, changed)
        changed = copy.deepcopy(self.revision)
        changed['changed_execution_files'].append(dict(path='scripts/not_pinned.py'))
        with self.assertRaisesRegex(ValueError, 'not pinned'):
            provenance.verify_execution_files(self.records, changed)
        tampered = copy.deepcopy(self.records)
        tampered[[r['path'].endswith('common.py') for r in tampered].index(True)]['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            provenance.verify_execution_files(tampered, self.revision)
        unrecorded = dict(self.revision, changed_execution_files=[])
        with self.assertRaises(ValueError):
            provenance.verify_execution_files(self.records, unrecorded)


class ClusterScriptSecurityTest(unittest.TestCase):
    script = PROJECT_DIR / 'scripts/cluster/check.sh'

    def test_host_keys_are_verified_and_nodes_are_not_hardcoded(self):
        text = self.script.read_text()
        self.assertIn('StrictHostKeyChecking=yes', text)
        for weak in ('StrictHostKeyChecking=no', 'accept-new', 'UserKnownHostsFile=/dev/null'):
            self.assertNotIn(weak, text)
        self.assertIn('site_config.py" nodes', text)

    def test_missing_node_configuration_fails_without_connecting(self):
        env = dict(os.environ, ILC_SITE_CONFIG='/nonexistent', ML_PFA_CLUSTER_NODES='')
        result = subprocess.run(['bash', str(self.script)], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(result.returncode, 2)
        self.assertIn('site configuration', result.stderr)


if __name__ == '__main__':
    unittest.main()
