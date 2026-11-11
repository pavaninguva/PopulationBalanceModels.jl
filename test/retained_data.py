"""Offline contracts for explicit, checksum-verified reproduction downloads."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('retained_data', ROOT / 'tools/retained_data.py')
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class RetainedData(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.destination = self.folder / 'checkout'
        self.contents = {'study/trajectory.csv': b'time,number\n0,1\n1,0.5\n',
                         'study/figure.png': b'exact original image bytes',
                         'history/target.zip': b'original archive bytes'}
        self.items = [dict(path=name, group='history' if name.startswith('history/') else 'study',
                           bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
                      for name, content in self.contents.items()]
        self.manifest = dict(schema=1, source=dict(revision='a'*40, archive_prefix='PBM-'+'a'*40,
                                                  archive_url='https://example.invalid/immutable-source.tar.gz'),
                             groups=dict(study='A retained study', history='A historical target'),
                             profiles=dict(docs=['study'], all=['study', 'history']), files=self.items)
        self.archive = self.folder / 'snapshot.zip'
        with zipfile.ZipFile(self.archive, 'w') as archive:
            for name, content in self.contents.items():
                archive.writestr(name, content)

    def tearDown(self):
        self.temporary.cleanup()

    def test_selected_group_restores_exact_bytes_and_original_paths(self):
        items = data.select(self.manifest, profile='docs')
        self.assertEqual(data.fetch(self.manifest, items, self.destination, archive=self.archive), 2)
        self.assertEqual(data.verify(items, self.destination), 2)
        for item in items:
            self.assertEqual((self.destination / item['path']).read_bytes(), self.contents[item['path']])
        self.assertFalse((self.destination / 'history').exists())

    def test_present_verified_files_do_not_download(self):
        data.fetch(self.manifest, self.items, self.destination, archive=self.archive)
        with patch.object(data, 'download', side_effect=AssertionError('No implicit download')):
            self.assertEqual(data.fetch(self.manifest, self.items, self.destination), 0)

    def test_changed_producer_output_is_never_overwritten(self):
        target = self.destination / self.items[0]['path']
        target.parent.mkdir(parents=True)
        target.write_bytes(b'new independent production result')
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            data.fetch(self.manifest, self.items, self.destination, archive=self.archive)
        self.assertEqual(target.read_bytes(), b'new independent production result')

    def test_bad_hash_and_size_leave_no_invalid_output_or_temporary(self):
        for change in (dict(sha256='0'*64), dict(bytes=2), dict(bytes=100000)):
            item = dict(self.items[0], **change)
            with self.assertRaises(ValueError):
                data.fetch(self.manifest, [item], self.destination, archive=self.archive)
            self.assertFalse((self.destination / item['path']).exists())
            self.assertEqual(list(self.destination.rglob('.retained-*')), [])

    def test_missing_and_duplicate_archive_entries_are_visible(self):
        missing = dict(self.items[0], path='study/missing.csv')
        with self.assertRaisesRegex(ValueError, 'missing 1 requested files'):
            data.fetch(self.manifest, [missing], self.destination, archive=self.archive)
        duplicate = self.folder / 'duplicate.zip'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(duplicate, 'w') as archive:
                for _ in range(2):
                    archive.writestr(self.items[0]['path'], self.contents[self.items[0]['path']])
        with self.assertRaisesRegex(ValueError, 'Duplicate archive entry'):
            data.fetch(self.manifest, [self.items[0]], self.destination, archive=duplicate)

    def test_unrequested_unsafe_members_are_not_extracted(self):
        with zipfile.ZipFile(self.archive, 'a') as archive:
            archive.writestr('../escape', b'never installed')
        data.fetch(self.manifest, self.items, self.destination, archive=self.archive)
        self.assertFalse((self.destination.parent / 'escape').exists())
        self.assertEqual(len([p for p in self.destination.rglob('*') if p.is_file()]), 3)

    def test_nonregular_archive_members_are_rejected(self):
        archive_path = self.folder / 'symlink.zip'
        for mode in (0o120777, 0o010644):  # symbolic link and FIFO
            member = zipfile.ZipInfo(self.items[0]['path'])
            member.create_system = 3
            member.external_attr = mode << 16
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr(member, b'/outside')
            with self.assertRaisesRegex(ValueError, 'Non-regular'):
                data.fetch(self.manifest, [self.items[0]], self.destination, archive=archive_path)

    def test_symlink_destination_cannot_escape_checkout(self):
        outside = self.folder / 'outside'
        outside.mkdir()
        self.destination.mkdir()
        try:
            (self.destination / 'study').symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('Platform does not permit creating directory symlinks')
        with self.assertRaisesRegex(ValueError, 'escapes destination'):
            data.fetch(self.manifest, self.items, self.destination, archive=self.archive)
        self.assertEqual(list(outside.iterdir()), [])

    def test_pinned_source_tar_preserves_the_same_contents(self):
        archive_path = self.folder / 'source.tar.gz'
        prefix = self.manifest['source']['archive_prefix']
        with tarfile.open(archive_path, 'w:gz') as archive:
            for name, content in self.contents.items():
                member = tarfile.TarInfo(prefix + '/' + name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        count = data.fetch(self.manifest, self.items, self.destination, archive=archive_path)
        self.assertEqual(count, 3)
        for name, content in self.contents.items():
            self.assertEqual((self.destination / name).read_bytes(), content)

    def test_download_is_explicit_and_reuses_the_cached_source(self):
        self.manifest['source']['archive_url'] = self.archive.as_uri()
        cached = data.download(self.manifest, self.folder / 'cache')
        self.assertEqual(cached.read_bytes(), self.archive.read_bytes())
        with patch.object(data.urllib.request, 'urlopen', side_effect=AssertionError('Cache was not reused')):
            self.assertEqual(data.download(self.manifest, self.folder / 'cache'), cached)

    def test_failed_download_does_not_leave_a_completed_cache_file(self):
        with patch.object(data.urllib.request, 'urlopen', side_effect=OSError('Unavailable source')):
            with self.assertRaisesRegex(OSError, 'Unavailable source'):
                data.download(self.manifest, self.folder / 'cache')
        self.assertEqual(list((self.folder / 'cache').iterdir()), [])

    def test_verify_never_downloads_and_reports_the_required_preparation(self):
        with patch.object(data, 'download', side_effect=AssertionError('No implicit download')):
            with self.assertRaisesRegex(ValueError, 'tools/retained_data.py fetch'):
                data.verify(self.items, self.destination)

    def test_independent_local_archive_round_trip(self):
        data.fetch(self.manifest, self.items, self.destination, archive=self.archive)
        output = self.folder / 'portable.zip'
        sha = data.create_archive(self.manifest, self.items, output, self.destination)
        self.assertEqual(sha, hashlib.sha256(output.read_bytes()).hexdigest())
        with zipfile.ZipFile(output) as archive:
            for name, content in self.contents.items():
                self.assertEqual(archive.read(name), content)
        with self.assertRaises(FileExistsError):
            data.create_archive(self.manifest, self.items, output, self.destination)

    def test_manifest_validation_and_group_selection(self):
        path = self.folder / 'manifest.json'
        path.write_text(json.dumps(self.manifest))
        self.assertEqual(data.load_manifest(path)['files'], self.items)
        self.assertEqual(len(data.select(self.manifest, profile='docs', groups=['history'])), 3)
        for kwargs in (dict(groups=['missing']), dict(profile='missing')):
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                data.select(self.manifest, **kwargs)
        for name in ('../escape', '/absolute', 'C:/outside', 'data\\escape', '.git/config', 'data/../escape'):
            with self.assertRaisesRegex(ValueError, 'Invalid retained-data path'):
                data.relative_path(name)
        self.manifest['files'].append(self.items[0].copy())
        path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            data.load_manifest(path)

    def test_release_inventory_contains_payloads_not_core_or_model_inputs(self):
        manifest = data.load_manifest()
        self.assertEqual(len(manifest['source']['revision']), 40)
        self.assertIn(manifest['source']['revision'], manifest['source']['archive_url'])
        for item in manifest['files']:
            self.assertFalse(item['path'].startswith(('src/', 'ext/', 'python/', 'examples/')))
            self.assertNotEqual(Path(item['path']).suffix, '.jl')
            self.assertFalse(item['path'].startswith('paper/') and '/figures/' in item['path'])
            self.assertFalse(item['path'].startswith(('docs/src/assets/milling_uncertainty/',
                                                     'paper/milling/')))
            self.assertFalse(item['path'].startswith('docs/src/assets/constant_growth/') and
                             Path(item['path']).stem in ('solutions','performance','observed_orders','timings') and
                             Path(item['path']).suffix in ('.png','.pdf','.svg'))
        result = subprocess.run([sys.executable, '-I', str(ROOT / 'tools/retained_data.py'), 'list'],
                                cwd=self.folder, capture_output=True, text=True, check=True)
        self.assertIn('documentation:', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
