"""Local site configuration and public-repository hygiene (portable committed records)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.helpers import PROJECT_DIR
from experiments import site_config
from experiments.stage2b import common as stage2b_common, dataset as data
from experiments.stage3 import common as stage3_common


def write(directory, payload):
    path = Path(directory) / 'site.json'
    path.write_text(json.dumps(payload))
    return str(path)


VALID = dict(schema_version=1, cluster=dict(nodes=['a', 'b'], preferred_node='b'), paths=dict(ILC_TEST_ROOT='/srv/data'))


class SiteConfigTest(unittest.TestCase):
    def test_tests_use_the_placeholder_template(self):
        self.assertEqual(Path(os.environ[site_config.ENV]), PROJECT_DIR / 'configs/site.example.json')
        self.assertEqual(site_config.cluster()['preferred_node'], 'gpu-node-02')

    def test_missing_or_invalid_configuration_fails_explicitly(self):
        with patch.dict(os.environ, {site_config.ENV: '/nonexistent/site.json'}):
            with self.assertRaisesRegex(site_config.SiteConfigError, 'copy configs/site.example.json') as raised:
                site_config.cluster()
            self.assertNotIn('/nonexistent', str(raised.exception))
        with tempfile.TemporaryDirectory() as tmp:
            for broken in (dict(VALID, schema_version=2), dict(VALID, cluster=dict(nodes=['a'], preferred_node='z')),
                           dict(VALID, cluster=dict(nodes=['a', 'a'], preferred_node='a')),
                           dict(VALID, cluster=dict(nodes=['-oProxyCommand=x'], preferred_node='-oProxyCommand=x'))):
                with patch.dict(os.environ, {site_config.ENV: write(tmp, broken)}):
                    with self.assertRaises(site_config.SiteConfigError):
                        site_config.load()
            (Path(tmp) / 'site.json').write_text('{not json')
            with patch.dict(os.environ, {site_config.ENV: str(Path(tmp) / 'site.json')}):
                with self.assertRaises(site_config.SiteConfigError):
                    site_config.load()

    def test_expand_prefers_environment_then_site_and_rejects_unknown(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {site_config.ENV: write(tmp, VALID)}):
            self.assertEqual(site_config.expand('${ILC_TEST_ROOT}/x.root'), '/srv/data/x.root')
            with patch.dict(os.environ, ILC_TEST_ROOT='/override'):
                self.assertEqual(site_config.expand('${ILC_TEST_ROOT}/x.root'), '/override/x.root')
            self.assertEqual(site_config.expand('/absolute/unchanged'), '/absolute/unchanged')
            with self.assertRaisesRegex(site_config.SiteConfigError, 'ILC_MISSING'):
                site_config.expand('${ILC_MISSING}/x')

    def test_policies_take_nodes_from_site_and_digest_only_committed_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            digests = []
            for nodes in (['a', 'b'], ['x', 'y', 'z']):
                payload = dict(VALID, cluster=dict(nodes=nodes, preferred_node=nodes[-1]))
                with patch.dict(os.environ, {site_config.ENV: write(tmp, payload)}):
                    for loader in (stage2b_common.load_policy, stage3_common.load_policy):
                        policy, digest = loader()
                        self.assertEqual(policy['allocation']['authorized_nodes'], nodes)
                        digests.append(digest)
            self.assertEqual(digests[:2], digests[2:])
        historical = dict(authorized_nodes=['old'], preferred_node='old')
        self.assertEqual(site_config.with_cluster(historical), historical)

    def test_shell_helper_and_missing_configuration(self):
        script = PROJECT_DIR / 'scripts/experiments/site_config.py'
        result = subprocess.run([sys.executable, str(script), 'nodes'], capture_output=True, text=True)
        self.assertEqual((result.returncode, result.stdout.split()), (0, ['gpu-node-01', 'gpu-node-02', 'gpu-node-03']))
        result = subprocess.run([sys.executable, str(script), 'nodes'], capture_output=True, text=True,
                                env=dict(os.environ, **{site_config.ENV: '/nonexistent'}))
        self.assertEqual(result.returncode, 2)
        self.assertIn('site configuration', result.stderr)

    def test_launcher_refuses_without_site_configuration(self):
        from scripts.experiments import stage3_campaign as cli
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {site_config.ENV: '/nonexistent'}), \
             patch.object(cli, 'repository_revision', lambda: dict(commit='abc')):
            self.assertEqual(cli.main(['--output', tmp, '--dry-run']), stage3_common.EXIT_INVALID)


class PortableRecordsTest(unittest.TestCase):
    def committed(self, path):
        return json.loads((PROJECT_DIR / path).read_text())

    def test_policies_do_not_list_nodes(self):
        for path in ('configs/prepared/stage2b_train8m_v1/smoke_policy.json', 'configs/prepared/stage3_train8m_v1/policy.json'):
            allocation = self.committed(path)['allocation']
            self.assertNotIn('authorized_nodes', allocation)
            self.assertNotIn('preferred_node', allocation)

    def test_portable_records_hold_no_absolute_paths(self):
        def strings(value):
            if isinstance(value, dict):
                for item in value.values():
                    yield from strings(item)
            elif isinstance(value, list):
                for item in value:
                    yield from strings(item)
            elif isinstance(value, str):
                yield value
        for path in ('configs/prepared/stage3_train8m_v1/protocol.json', 'configs/prepared/stage3_train8m_v1/protocol_v2.json',
                     'configs/prepared/stage3_train8m_v1/test3000k_spec.json',
                     'datasets/manifests/20250711_ilc_nnqq_sgv_10m_3cat_train8m_parquet_v1.json'):
            absolute = [s for s in strings(self.committed(path)) if s.startswith('/')]
            self.assertEqual(absolute, [], path)

    def test_portable_manifest_drops_machine_specific_locations(self):
        full = dict(local_path='datasets/x', weaver_audit=dict(fingerprint='f', total_rows=3, files=[dict(path='/abs/x.parquet')]),
                    files=[dict(path='datasets/x/a.parquet', resolved_path='/abs/a.parquet', sha256='s',
                                source_root_path='/private/a.root', rows=3)])
        portable = data.portable_manifest(full)
        self.assertEqual(portable['files'], [dict(path='datasets/x/a.parquet', sha256='s', rows=3, source_root_file='a.root')])
        self.assertEqual(portable['weaver_audit'], dict(fingerprint='f', total_rows=3))
        self.assertTrue(portable['portable'])

    def test_sealed_test_sources_resolve_through_the_site_configuration(self):
        spec = data.validate_spec(self.committed('configs/prepared/stage3_train8m_v1/test3000k_spec.json'))
        self.assertTrue(all(s['path'].startswith('${ILC_SGV10M_SOURCE_DIR}/') for s in spec['sources']))
        self.assertEqual(str(data.source_path(spec['sources'][0])),
                         '/path/to/20250711_ilc_nnqq_sgv_10m/ml_flavtag_6cat_bb_test3000k.root')

    def test_local_configuration_and_outputs_are_ignored(self):
        for path in ('configs/local/site.json', 'runs/eval/x/status.json', 'logs/x.log', '.env', 'id_ed25519', 'x.pem',
                     'datasets/x/MANIFEST.json'):
            result = subprocess.run(['git', 'check-ignore', '-q', path], cwd=PROJECT_DIR)
            self.assertEqual(result.returncode, 0, path)
        self.assertNotEqual(subprocess.run(['git', 'check-ignore', '-q', 'configs/site.example.json'], cwd=PROJECT_DIR).returncode, 0)


if __name__ == '__main__':
    unittest.main()
