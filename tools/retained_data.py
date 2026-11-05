"""Fetch immutable reproduction payloads; never run simulations or change producers.

Uses only Python's standard library (Python 3.9+). Run ``list`` to inspect groups,
then ``fetch --profile docs`` or ``fetch --group paper-protein`` explicitly.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_suffix('.json')
BLOCK = 1024 * 1024


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            result.update(block)
    return result.hexdigest()


def relative_path(name):
    """Manifest/archive names are checkout-relative POSIX paths on every OS."""
    if not isinstance(name, str) or '\\' in name or ':' in name:
        raise ValueError(f'Invalid retained-data path: {name!r}')
    parts = name.split('/')
    if any(part in ('', '.', '..') or part.casefold() == '.git' for part in parts):
        raise ValueError(f'Invalid retained-data path: {name!r}')
    return PurePosixPath(name)


def load_manifest(path=MANIFEST):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('schema') != 1:
        raise ValueError('Unsupported retained-data manifest schema')
    names, folded = set(), set()
    for item in manifest['files']:
        name = str(relative_path(item['path']))
        if name in names or name.casefold() in folded:
            raise ValueError(f'Duplicate retained-data path: {name}')
        names.add(name)
        folded.add(name.casefold())
        sha = item['sha256']
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError(f'Invalid SHA-256 for {name}')
        if not isinstance(item['bytes'], int) or item['bytes'] < 0:
            raise ValueError(f'Invalid byte count for {name}')
        if item['group'] not in manifest['groups']:
            raise ValueError(f'Unknown data group for {name}')
    for profile, groups in manifest['profiles'].items():
        if not groups or any(group not in manifest['groups'] for group in groups):
            raise ValueError(f'Invalid retained-data profile: {profile}')
    return manifest


def select(manifest, *, profile=None, groups=()):
    if profile is not None and profile not in manifest['profiles']:
        raise ValueError(f'Unknown profile {profile!r}; choose from {", ".join(manifest["profiles"])}')
    chosen = set(manifest['profiles'][profile] if profile is not None else ()) | set(groups)
    if not chosen:
        chosen = set(manifest['groups'])
    unknown = chosen - set(manifest['groups'])
    if unknown:
        raise ValueError(f'Unknown retained-data groups: {", ".join(sorted(unknown))}')
    return [item for item in manifest['files'] if item['group'] in chosen]


def target_path(destination, name):
    base = Path(destination).resolve()
    target = base.joinpath(*relative_path(name).parts)
    if not target.resolve().is_relative_to(base) or target.is_symlink():
        raise ValueError(f'Retained-data path escapes destination: {name}')
    return target


def matches(path, item):
    return path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256']


def pending(items, destination):
    missing = []
    for item in items:
        target = target_path(destination, item['path'])
        if target.exists():
            if not matches(target, item):
                raise ValueError(f'Existing file differs; refusing to overwrite {target}')
        else:
            missing.append(item)
    return missing


def install(stream, item, destination):
    """Verify before publishing; an existing producer/output is never replaced."""
    target = target_path(destination, item['path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.retained-', dir=target.parent)
    temporary = Path(temporary)
    try:
        sha, size = hashlib.sha256(), 0
        with os.fdopen(fd, 'wb') as output:
            for block in iter(lambda: stream.read(BLOCK), b''):
                size += len(block)
                if size > item['bytes']:
                    raise ValueError(f'Unexpected size for {item["path"]}')
                sha.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if size != item['bytes'] or sha.hexdigest() != item['sha256']:
            raise ValueError(f'SHA-256/size mismatch for {item["path"]}')
        try:
            # An atomic hard link cannot overwrite a file created concurrently.
            os.link(temporary, target)
        except FileExistsError:
            if not matches(target, item):
                raise ValueError(f'Existing file differs; refusing to overwrite {target}')
    finally:
        temporary.unlink(missing_ok=True)


def extract(archive, items, destination, manifest):
    """Read whitelisted regular files only; never call extractall."""
    expected = {item['path']: item for item in items}
    found = set()

    def accept(name, stream):
        if name in found:
            raise ValueError(f'Duplicate archive entry: {name}')
        found.add(name)
        install(stream, expected[name], destination)

    archive = Path(archive)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                if member.filename not in expected:
                    continue
                mode = member.external_attr >> 16
                if member.is_dir() or (mode & 0o170000) not in (0, 0o100000):
                    raise ValueError(f'Non-regular archive member: {member.filename}')
                with source.open(member) as stream:
                    accept(member.filename, stream)
    else:
        prefix = manifest['source']['archive_prefix'] + '/'
        with tarfile.open(archive, 'r:gz') as source:
            for member in source:
                if not member.name.startswith(prefix):
                    continue
                name = member.name[len(prefix):]
                if name not in expected:
                    continue
                if not member.isfile():
                    raise ValueError(f'Non-regular archive member: {member.name}')
                with source.extractfile(member) as stream:
                    accept(name, stream)
    absent = set(expected) - found
    if absent:
        raise ValueError(f'Archive is missing {len(absent)} requested files, including {min(absent)}')
    return len(found)


def download(manifest, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / (manifest['source']['revision'] + '.tar.gz')
    if archive.is_file():
        return archive
    fd, temporary = tempfile.mkstemp(prefix='.download-', dir=cache)
    temporary = Path(temporary)
    try:
        request = urllib.request.Request(manifest['source']['archive_url'],
                                         headers={'User-Agent': 'PopulationBalanceModels-retained-data'})
        with os.fdopen(fd, 'wb') as output:
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, output, length=BLOCK)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def fetch(manifest, items, destination=ROOT, *, archive=None, cache=None):
    missing = pending(items, destination)
    if not missing:
        return 0
    if archive is None:
        cache = cache or os.environ.get('PBM_RETAINED_CACHE', str(ROOT / '.cache' / 'retained-downloads'))
        archive = download(manifest, cache)
    return extract(archive, missing, destination, manifest)


def verify(items, destination=ROOT):
    missing = pending(items, destination)
    if missing:
        groups = ', '.join(sorted({item['group'] for item in missing}))
        raise ValueError(f'{len(missing)} retained files missing ({groups}). '
                         'Run python tools/retained_data.py fetch with the required --group or --profile.')
    return len(items)


def create_archive(manifest, items, output, destination=ROOT):
    verify(items, destination)
    output = Path(output)
    # Exclusive creation prevents accidental replacement of a previous snapshot.
    with output.open('xb') as handle:
        with zipfile.ZipFile(handle, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as source:
            source.writestr('_retained_data.json', json.dumps(manifest, indent=2, sort_keys=True) + '\n')
            for item in items:
                path = target_path(destination, item['path'])
                compression = zipfile.ZIP_STORED if path.suffix in ('.gz', '.png', '.pdf', '.zip') else zipfile.ZIP_DEFLATED
                source.write(path, item['path'], compress_type=compression)
    return digest(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('list', 'fetch', 'verify', 'archive'))
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--profile')
    parser.add_argument('--group', action='append', default=[])
    parser.add_argument('--destination', type=Path, default=ROOT,
                        help='Checkout root to materialize original relative paths into.')
    parser.add_argument('--archive', type=Path, help='Use a local verified snapshot instead of downloading.')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path, help='Destination ZIP for the archive command.')
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        items = select(manifest, profile=args.profile, groups=args.group)
        if args.command == 'list':
            for group, description in manifest['groups'].items():
                rows = [item for item in items if item['group'] == group]
                if rows:
                    print(f'{group}: {len(rows)} files, {sum(row["bytes"] for row in rows)/2**20:.2f} MiB — {description}')
        elif args.command == 'fetch':
            count = fetch(manifest, items, args.destination, archive=args.archive, cache=args.cache)
            print(f'Installed {count} missing files; {verify(items, args.destination)} payload hashes verified. No numerical calculations run.')
        elif args.command == 'verify':
            print(f'Verified {verify(items, args.destination)} retained payload hashes.')
        elif args.output is None:
            parser.error('archive requires --output')
        else:
            print(f'{create_archive(manifest, items, args.output, args.destination)}  {args.output}')
    except (ValueError, OSError, tarfile.TarError, zipfile.BadZipFile) as failure:
        parser.exit(1, f'Retained data: {failure}\n')


if __name__ == '__main__':
    main()
