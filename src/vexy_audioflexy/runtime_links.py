# this_file: src/vexy_audioflexy/runtime_links.py
"""Explicit link-aware runtime relocation; receipts and plans must remain private."""

import errno
import hashlib
import json
import os
import stat
from collections.abc import Iterator
from pathlib import Path

from .migration import _paths, _write_receipt


def _raise_walk_error(error: OSError) -> None:
    raise error


def _entries(root: Path) -> Iterator[Path]:
    """Walk physical entries only, including Git and environment contents."""
    yield root
    if root.is_dir():
        for parent, directories, files in os.walk(root, followlinks=False, onerror=_raise_walk_error):
            for name in sorted(directories + files):
                yield Path(parent) / name


def _fingerprint(root: Path, original_links: dict | None = None) -> str:
    digest = hashlib.sha256()
    for path in _entries(root):
        relative = str(path.relative_to(root))
        info = path.lstat()
        kind = stat.S_IFMT(info.st_mode)
        if kind not in (stat.S_IFREG, stat.S_IFDIR, stat.S_IFLNK):
            raise ValueError(f'Unsupported physical runtime entry: {path}')
        # Rename preserves ownership, permissions and bytes, including owner edits.
        digest.update(json.dumps([relative, kind, stat.S_IMODE(info.st_mode),
                                  info.st_uid, info.st_gid]).encode() + b'\0')
        if stat.S_ISLNK(info.st_mode):
            raw = os.readlink(path)
            digest.update((original_links or {}).get(relative, raw).encode() + b'\0')
        elif stat.S_ISREG(info.st_mode):
            digest.update(str(info.st_size).encode() + b'\0')
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
    return digest.hexdigest()


def _meaning(path: Path, root: Path) -> dict:
    try:
        resolved = path.resolve(strict=True)
        kind = stat.S_IFMT(resolved.stat().st_mode)
        location = ('internal:' + str(resolved.relative_to(root))
                    if resolved.is_relative_to(root) else str(resolved))
        return {'resolved': location, 'kind': kind}
    except (OSError, RuntimeError) as error:
        # Existing missing/cyclic pointers are a reviewed baseline, never repaired.
        if isinstance(error, FileNotFoundError):
            return {'error': 'missing'}
        if isinstance(error, RuntimeError) or getattr(error, 'errno', None) == errno.ELOOP:
            return {'error': 'cycle'}
        raise


def reviewed_runtime_link_entry(source: str, target: str, *, symlinks: list[dict],
                                rewrites: dict[str, str], reviewed: bool = False) -> dict:
    """Bind an exact complete link list and explicit old-logical-target rewrites.

    Rewrites may only replace a relative external link with its old absolute
    logical target. They must not pin a resolved provider/interpreter version.
    Standalone regular runtime files require empty link and rewrite manifests.
    """
    if not reviewed or not isinstance(symlinks, list) or not isinstance(rewrites, dict):
        raise ValueError('Reviewed exact link and rewrite manifests required')
    src, dst = _paths(source, target)
    observed, semantics, required = [], {}, set()
    for path in _entries(src):
        if not path.is_symlink():
            continue
        relative, raw = str(path.relative_to(src)), os.readlink(path)
        observed.append({'path': relative, 'target': raw})
        semantics[relative] = _meaning(path, src)
        logical = Path(os.path.abspath(path.parent / raw))
        external = (not logical.is_relative_to(src)
                    or ('resolved' in semantics[relative]
                        and not semantics[relative]['resolved'].startswith('internal:')))
        if not Path(raw).is_absolute() and external:
            required.add(relative)
            if rewrites.get(relative) != str(logical):
                raise ValueError(f'Explicit old logical absolute target required: {relative}')
    observed.sort(key=lambda item: item['path'])
    if observed != symlinks or set(rewrites) != required:
        raise ValueError('Complete exact link manifest and only required rewrites must match')
    return {'source': str(src), 'target': str(dst), 'provider': 'local',
            'classification': 'runtime_bundle' if src.is_dir() else 'runtime_file',
            'reviewed': True, 'symlinks': observed, 'rewrites': dict(rewrites),
            'semantics': semantics, 'sha256': _fingerprint(src)}


def _restore_link_metadata(path: Path, info: os.stat_result) -> None:
    current = path.lstat()
    if (current.st_uid, current.st_gid) != (info.st_uid, info.st_gid):
        os.chown(path, info.st_uid, info.st_gid, follow_symlinks=False)
    if stat.S_IMODE(current.st_mode) != stat.S_IMODE(info.st_mode):
        os.chmod(path, stat.S_IMODE(info.st_mode), follow_symlinks=False)
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns), follow_symlinks=False)


def apply_runtime_link_entry(entry: dict, receipt_path: str) -> dict:
    """Rename one stopped runtime, restoring literal links on handled failure.

    shortcut: stopped writers and serial callers are required; crash recovery
    uses the durable prepared receipt rather than automatic replay.
    """
    fresh = reviewed_runtime_link_entry(entry['source'], entry['target'],
                                        symlinks=entry['symlinks'], rewrites=entry['rewrites'],
                                        reviewed=entry['reviewed'])
    if fresh != entry:
        raise ValueError('Runtime entry changed since review')
    src, dst = Path(entry['source']), Path(entry['target'])
    receipt_file = Path(receipt_path).expanduser().absolute()
    resolved = receipt_file.resolve(strict=False)
    if any(resolved == p or p in resolved.parents for p in (src, dst)):
        raise ValueError('Receipt must be outside the runtime')
    receipt = {**entry, 'state': 'prepared',
               'pre_raw_targets': {x['path']: x['target'] for x in entry['symlinks']}}
    receipt['post_raw_targets'] = {**receipt['pre_raw_targets'], **entry['rewrites']}
    with receipt_file.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    metadata = {relative: (src / relative).lstat() for relative in entry['rewrites']}
    try:
        src.rename(dst)
    except BaseException:
        _write_receipt(receipt_file, {**receipt, 'state': 'not_moved'})
        raise
    rewritten = []
    try:
        # Old logical targets must already work when rewrites are installed.
        src.symlink_to(dst, target_is_directory=dst.is_dir())
        for relative, raw in entry['rewrites'].items():
            path = dst / relative
            path.unlink()
            rewritten.append(relative)
            path.symlink_to(raw)
            _restore_link_metadata(path, metadata[relative])
        if _fingerprint(dst, receipt['pre_raw_targets']) != entry['sha256']:
            raise ValueError('Runtime bytes or metadata changed during relocation')
        post = {str(p.relative_to(dst)): os.readlink(p)
                for p in _entries(dst) if p.is_symlink()}
        if post != receipt['post_raw_targets']:
            raise ValueError('Runtime literal link map changed during relocation')
        meanings = {relative: _meaning(dst / relative, dst) for relative in post}
        if meanings != entry['semantics']:
            raise ValueError('Runtime link semantics changed during relocation')
        receipt.update(state='complete', post_semantics=meanings)
        _write_receipt(receipt_file, receipt)
    except BaseException:
        try:
            for relative in reversed(rewritten):
                path = dst / relative
                if path.is_symlink():
                    path.unlink()
                elif path.exists():
                    raise RuntimeError('Rewritten link occupied; inspect recovery receipt')
                path.symlink_to(receipt['pre_raw_targets'][relative])
                _restore_link_metadata(path, metadata[relative])
            if src.is_symlink() and os.readlink(src) == str(dst):
                src.unlink()
            if src.exists() or src.is_symlink():
                raise RuntimeError('Source occupied; inspect recovery receipt')
            dst.rename(src)
        except BaseException:
            _write_receipt(receipt_file, {**receipt, 'state': 'recovery_required'})
            raise
        _write_receipt(receipt_file, {**receipt, 'state': 'rolled_back'})
        raise
    return receipt
