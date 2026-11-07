# this_file: src/vexy_audioflexy/migration.py
"""Reviewed, serial, same-volume model relocation; receipts belong in private data."""

import hashlib
import json
import os
from pathlib import Path


_FORBIDDEN = {'.git', '.venv', '.env', 'token', 'stored_tokens', 'credentials', 'input_audio', 'fixtures'}


def _digest(path: Path, symlinks: list[dict] | None = None) -> str:
    """Hash complete regular-file bundles, refusing links and mixed/private trees."""
    digest = hashlib.sha256()
    entries = [path, *sorted(path.rglob('*'))] if path.is_dir() else [path]
    for item in entries:
        if item.is_symlink():
            if symlinks is None:
                raise ValueError(f'Unclassified linked entry: {item}')
            digest.update(json.dumps([str(item.relative_to(path)), 'symlink', os.readlink(item)]).encode() + b'\0')
            continue
        if symlinks is None and item.name in _FORBIDDEN:
            raise ValueError(f'Unclassified or private entry: {item}')
        relative = item.relative_to(path).as_posix()
        digest.update(json.dumps([relative, item.is_dir()]).encode() + b'\0')
        if item.is_file():
            digest.update(str(item.stat().st_size).encode() + b'\0')
            with item.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
        elif not item.is_dir():
            raise ValueError(f'Unsupported filesystem entry: {item}')
    return digest.hexdigest()


def _paths(source: str, target: str) -> tuple[Path, Path]:
    src, dst = Path(source).expanduser().absolute(), Path(target).expanduser().absolute()
    if src.is_symlink() or not src.exists():
        raise ValueError('Source must exist and must not be a symlink')
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    src, dst = src.resolve(strict=True), dst.resolve(strict=False)
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError('Source and target must not overlap')
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    if not dst.parent.is_dir():
        raise ValueError('Create and review the target parent directory first')
    if src.stat().st_dev != dst.parent.stat().st_dev:
        raise ValueError('Cross-volume relocation is unsupported; originals preserved')
    return src, dst


def reviewed_entry(source: str, target: str, *, provider: str,
                   classification: str, reviewed: bool = False) -> dict:
    """Prepare a content-bound entry after human classification and caller review.

    Provider-native caches are registered where they are; this mover cannot merge
    HF snapshots or convert downloaded files into LM Studio managed artifacts.
    """
    if not reviewed or classification != 'model_bundle' or provider != 'local':
        raise ValueError('Only reviewed, standalone, non-provider model bundles may move')
    src, dst = _paths(source, target)
    return {'source': str(src), 'target': str(dst), 'provider': provider,
            'classification': classification, 'reviewed': True, 'sha256': _digest(src)}


def _write_receipt(path: Path, receipt: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def apply_entry(entry: dict, receipt_path: str) -> dict:
    """Apply one reviewed entry with a durable receipt and compatibility symlink.

    Run serially with writers stopped. Existing receipts require manual recovery,
    never automatic retry. A process crash between rename and link can leave the
    source absent; the prepared receipt identifies its intact target for recovery.
    """
    if entry['classification'] == 'runtime_bundle':
        checked = reviewed_runtime_entry(entry['source'], entry['target'],
                                         symlinks=entry['symlinks'], reviewed=entry['reviewed'])
    else:
        checked = reviewed_entry(entry['source'], entry['target'], provider=entry['provider'],
                                 classification=entry['classification'], reviewed=entry['reviewed'])
    if checked != entry:
        raise ValueError('Entry changed or source content differs from reviewed digest')
    src, dst = Path(entry['source']), Path(entry['target'])
    receipt_file = Path(receipt_path).expanduser().absolute()
    resolved_receipt = receipt_file.resolve(strict=False)
    if any(p == resolved_receipt or p in resolved_receipt.parents for p in (src, dst)):
        raise ValueError('Receipt must be outside the model bundle')
    receipt = {**entry, 'state': 'prepared'}
    with receipt_file.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    # shortcut: serial migration requires stopped writers; use OS-specific no-replace
    # rename primitives if concurrent migration is ever supported.
    try:
        if dst.exists() or dst.is_symlink():
            raise FileExistsError(dst)
        src.rename(dst)
    except BaseException:
        _write_receipt(receipt_file, {**receipt, 'state': 'not_moved'})
        raise
    try:
        if _digest(dst, entry.get('symlinks')) != entry['sha256']:
            raise ValueError('Content changed during relocation')
        src.symlink_to(dst, target_is_directory=dst.is_dir())
        receipt['state'] = 'complete'
        _write_receipt(receipt_file, receipt)
    except BaseException:
        if src.is_symlink() and os.readlink(src) == str(dst):
            src.unlink()
        if src.exists() or src.is_symlink():
            _write_receipt(receipt_file, {**receipt, 'state': 'recovery_required'})
            raise RuntimeError('Source occupied; intact target retained; inspect receipt')
        dst.rename(src)
        _write_receipt(receipt_file, {**receipt, 'state': 'rolled_back'})
        raise
    return receipt


def reviewed_duplicate(source: str, canonical: str, recovery: str, *,
                       identity: str, reviewed: bool = False) -> dict:
    """Bind equal files to an existing provider artifact, retaining a recovery copy."""
    if not reviewed or not isinstance(identity, str) or not identity.strip():
        raise ValueError('A reviewed exact provider identity is required')
    src, backup = _paths(source, recovery)
    requested = Path(canonical).expanduser().absolute()
    real = requested.resolve(strict=True)
    if not src.is_file() or not real.is_file() or real in (src, backup):
        raise ValueError('Duplicate consolidation requires distinct regular files')
    if os.path.samefile(src, real):
        raise ValueError('Source already aliases the provider artifact')
    digest = _digest(src)
    if digest != _digest(real):
        raise ValueError('Source and provider artifact differ')
    return {'source': str(src), 'canonical': str(requested),
            'canonical_resolved': str(real), 'recovery': str(backup),
            'identity': identity, 'reviewed': True, 'sha256': digest}


def apply_duplicate(entry: dict, receipt_path: str) -> dict:
    """Retain the duplicate by rename, then link its old path to provider storage.

    Run serially with source/provider writers stopped. Recovery files are retained,
    never deleted; this operation changes lookup topology, not disk usage.
    """
    fresh = reviewed_duplicate(entry['source'], entry['canonical'], entry['recovery'],
                               identity=entry['identity'], reviewed=entry['reviewed'])
    if fresh != entry:
        raise ValueError('Duplicate entry changed since review')
    src, backup = Path(entry['source']), Path(entry['recovery'])
    canonical = Path(entry['canonical'])
    receipt_file = Path(receipt_path).expanduser().absolute()
    receipt_resolved = receipt_file.resolve(strict=False)
    if receipt_resolved in (src, backup, canonical.resolve(strict=True)):
        raise ValueError('Receipt must be distinct from all artifact paths')
    receipt = {**entry, 'state': 'prepared'}
    with receipt_file.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    if backup.exists() or backup.is_symlink():
        raise FileExistsError(backup)
    src.rename(backup)
    try:
        if (canonical.resolve(strict=True) != Path(entry['canonical_resolved'])
                or _digest(canonical.resolve(strict=True)) != entry['sha256']
                or _digest(backup) != entry['sha256']):
            raise ValueError('Source or provider changed during consolidation')
        src.symlink_to(canonical)
        receipt['state'] = 'complete_recovery_retained'
        _write_receipt(receipt_file, receipt)
    except BaseException:
        if src.is_symlink() and os.readlink(src) == str(canonical):
            src.unlink()
        if src.exists() or src.is_symlink():
            _write_receipt(receipt_file, {**receipt, 'state': 'recovery_required'})
            raise RuntimeError('Source occupied; duplicate preserved at recovery path')
        backup.rename(src)
        _write_receipt(receipt_file, {**receipt, 'state': 'rolled_back'})
        raise
    return receipt


def reviewed_hf_seed(source: str, hub: str, *, repo: str, revision: str,
                     filename: str, etag: str, size: int, reviewed: bool = False) -> dict:
    """Bind a verified file to documented HF blob/snapshot cache paths, offline."""
    import re
    from pathlib import PurePosixPath

    if (not reviewed or not isinstance(repo, str)
            or len(repo.split('/')) != 2 or '--' in repo or '..' in repo
            or repo.endswith('.git')
            or any(len(part) > 96 or not re.fullmatch(r'[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?', part)
                   for part in repo.split('/'))
            or not re.fullmatch(r'[0-9a-f]{40}', revision)
            or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', etag)):
        raise ValueError('Reviewed exact HF repo, commit and ETag required')
    relative = PurePosixPath(filename)
    if relative.is_absolute() or '..' in relative.parts or str(relative) != filename or filename == '.':
        raise ValueError('HF filename must be a normalized relative artifact path')
    src = Path(source).expanduser().absolute()
    if src.is_symlink() or not src.is_file() or src.stat().st_size != size:
        raise ValueError('Source must be a regular file matching upstream size')
    src = src.resolve(strict=True)
    cache = Path(hub).expanduser().resolve(strict=True)
    if not cache.is_dir() or cache.stat().st_dev != src.stat().st_dev:
        raise ValueError('HF cache must exist on the source filesystem')
    root = cache / ('models--' + repo.replace('/', '--'))
    blob = root / 'blobs' / etag
    snapshot = root / 'snapshots' / revision / filename
    # Existing cache directory aliases could redirect writes outside this repository.
    if any(p.is_symlink() for p in [root, *blob.parents, *snapshot.parents] if p != cache and cache in p.parents):
        raise ValueError('HF cache parents must not be symlinks')
    sha = hashlib.sha256()
    git = hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
    with src.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
            git.update(block)
    if etag not in (sha.hexdigest(), git.hexdigest()):
        raise ValueError('Source content does not match upstream ETag')
    if blob.is_symlink() or (blob.exists() and (not blob.is_file() or _digest(blob) != _digest(src))):
        raise ValueError('Existing provider blob conflicts')
    if snapshot.exists() or snapshot.is_symlink():
        if not snapshot.is_symlink() or snapshot.resolve(strict=True) != blob:
            raise ValueError('Existing snapshot pointer conflicts')
    if src in (blob, snapshot):
        raise ValueError('Source must be outside the new provider artifact paths')
    return {'source': str(src), 'hub': str(cache), 'repo': repo, 'revision': revision,
            'filename': filename, 'etag': etag, 'size': size, 'reviewed': True,
            'raw_sha256': sha.hexdigest(), 'blob': str(blob), 'snapshot': str(snapshot)}


def apply_hf_seed(entry: dict, receipt_path: str) -> dict:
    """Hardlink verified source bytes into HF cache, preserving the source file."""
    fresh = reviewed_hf_seed(entry['source'], entry['hub'], repo=entry['repo'],
                             revision=entry['revision'], filename=entry['filename'],
                             etag=entry['etag'], size=entry['size'], reviewed=entry['reviewed'])
    if fresh != entry:
        raise ValueError('HF seed entry changed since review')
    src, blob, snapshot = (Path(entry[k]) for k in ('source', 'blob', 'snapshot'))
    receipt_file = Path(receipt_path).expanduser().absolute()
    if receipt_file.resolve(strict=False) in (src, blob, snapshot):
        raise ValueError('Receipt must be separate from artifacts')
    receipt = {**entry, 'state': 'prepared'}
    with receipt_file.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    created_blob = created_snapshot = False
    try:
        blob.parent.mkdir(parents=True, exist_ok=True)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not blob.exists():
            os.link(src, blob)
            created_blob = True
        if not snapshot.is_symlink():
            snapshot.symlink_to(os.path.relpath(blob, snapshot.parent))
            created_snapshot = True
        with blob.open('rb') as stream:
            final_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        if snapshot.resolve(strict=True) != blob or final_hash != entry['raw_sha256']:
            raise ValueError('Provider cache changed during seeding')
        receipt.update(state='complete_source_retained', created_blob=created_blob,
                       created_snapshot=created_snapshot)
        _write_receipt(receipt_file, receipt)
    except BaseException:
        if created_snapshot:
            snapshot.unlink()
        if created_blob:
            blob.unlink()
        _write_receipt(receipt_file, {**receipt, 'state': 'rolled_back_source_retained'})
        raise
    return receipt


def reviewed_runtime_entry(source: str, target: str, *, symlinks: list[dict],
                           reviewed: bool = False) -> dict:
    """Bind a complete runtime checkout, including Git/env bytes and explicit links.

    Relative external links need a separately reviewed rewrite; this first route
    preserves internal relative and stable absolute link semantics unchanged.
    """
    if not reviewed or not isinstance(symlinks, list):
        raise ValueError('Reviewed runtime and explicit link manifest required')
    src, dst = _paths(source, target)
    if not src.is_dir():
        raise ValueError('Runtime bundle must be a directory')
    observed = []
    for path in sorted(src.rglob('*')):
        if not path.is_symlink():
            continue
        raw = os.readlink(path)
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError('Runtime symlinks must resolve to regular files')
        if not Path(raw).is_absolute() and src not in resolved.parents:
            raise ValueError('Relative external runtime link needs reviewed rewriting')
        observed.append({'path': str(path.relative_to(src)), 'target': str(raw)})
    if observed != symlinks:
        raise ValueError('Runtime symlink manifest does not match the complete tree')
    return {'source': str(src), 'target': str(dst), 'provider': 'local',
            'classification': 'runtime_bundle', 'reviewed': True,
            'symlinks': observed, 'sha256': _digest(src, observed)}
