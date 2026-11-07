# this_file: src/vexy_audioflexy/overlays.py
"""Writable caller overlays over immutable provider artifacts and retained originals."""

import ctypes
import errno
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath

from .migration import _digest, _paths, _write_receipt


def reviewed_overlay_entry(source: str, overlay: str, recovery: str, *,
                           provider_files: dict[str, str], reviewed: bool = False,
                           writable_provider: bool = False) -> dict:
    """Bind a whole download root to provider links and a retained recovery tree.

    Original links require separate review; this route handles regular files and
    directories only. Unmapped writable metadata/derivatives are independent copies.
    """
    if not isinstance(writable_provider, bool):
        raise ValueError('writable_provider must be a boolean')
    if not reviewed or not isinstance(provider_files, dict) or not provider_files:
        raise ValueError('Reviewed provider file mapping required')
    src, dst = _paths(source, overlay)
    _, backup = _paths(source, recovery)
    if dst == backup or dst in backup.parents or backup in dst.parents:
        raise ValueError('Overlay and recovery paths must not overlap')
    if not src.is_dir():
        raise ValueError('Overlay source must be a directory')
    if writable_provider:
        _probe_clone(dst.parent)
    nodes = []
    for path in [src, *sorted(src.rglob('*'))]:
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('Original overlay trees must contain only regular files/directories')
        info = path.stat()
        nodes.append({'path': str(path.relative_to(src)), 'directory': path.is_dir(),
                      'mode': info.st_mode, 'mtime_ns': info.st_mtime_ns,
                      'size': info.st_size, 'inode': info.st_ino})
    providers = {}
    for name, canonical in provider_files.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or str(rel) != name or name == '.':
            raise ValueError('Provider keys must be normalized relative file paths')
        artifact = Path(canonical).expanduser().absolute()
        real = artifact.resolve(strict=True)
        if src == real or src in real.parents or dst == real or dst in real.parents or backup == real or backup in real.parents:
            raise ValueError('Provider artifacts must be outside source/overlay/recovery')
        if writable_provider and real.stat().st_dev != dst.parent.stat().st_dev:
            raise ValueError('APFS provider clones must remain on the same filesystem')
        if not (src / name).is_file() or not real.is_file():
            raise ValueError('Provider mapping must identify existing regular files')
        digest = _digest(src / name, [])
        if digest != _digest(real, []):
            raise ValueError('Provider artifact differs from original source')
        providers[name] = {'path': str(artifact), 'resolved': str(real), 'sha256': digest}
    entry = {'source': str(src), 'overlay': str(dst), 'recovery': str(backup),
             'provider_files': providers, 'nodes': nodes, 'sha256': _digest(src, []),
             'reviewed': True}
    if writable_provider:
        entry['provider_mode'] = 'apfs_clone'
    return entry


def apply_overlay(entry: dict, receipt_path: str) -> dict:
    """Stage writable directories, retain originals by rename, then link old root.

    Serial stopped-writer operation. No original bytes are deleted. Provider files
    are linked individually so new compiled outputs and atomic replacements land
    in writable overlay directories rather than immutable snapshot directories.
    """
    fresh = reviewed_overlay_entry(entry['source'], entry['overlay'], entry['recovery'],
                                   provider_files={k: v['path'] for k, v in entry['provider_files'].items()},
                                   reviewed=entry['reviewed'],
                                   writable_provider=entry.get('provider_mode') == 'apfs_clone')
    if fresh != entry:
        raise ValueError('Overlay source/provider/metadata changed since review')
    src, dst, backup = [Path(entry[k]) for k in ('source', 'overlay', 'recovery')]
    receipt_file = Path(receipt_path).expanduser().absolute()
    receipt_real = receipt_file.resolve(strict=False)
    if any(p == receipt_real or p in receipt_real.parents for p in (src, dst, backup)):
        raise ValueError('Receipt must be outside source/overlay/recovery')
    receipt = {**entry, 'state': 'prepared'}
    with receipt_file.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    created_overlay = moved = False
    try:
        dst.mkdir()
        created_overlay = True
        for node in entry['nodes'][1:]:
            relative = node['path']
            path = dst / relative
            if node['directory']:
                path.mkdir()
            else:
                canonical = entry['provider_files'].get(relative)
                if canonical:
                    if entry.get('provider_mode') == 'apfs_clone':
                        real = Path(canonical['resolved'])
                        _clone_file(real, path)
                        if path.is_symlink() or path.samefile(real) or _digest(path, []) != canonical['sha256']:
                            raise ValueError('Provider clone must have independent verified bytes')
                        path.chmod(stat.S_IMODE(node['mode']) | stat.S_IRUSR | stat.S_IWUSR)
                    else:
                        path.symlink_to(Path(canonical['path']))
                else:
                    shutil.copy2(src / relative, path)
                    path.chmod(stat.S_IMODE(node['mode']) | stat.S_IRUSR | stat.S_IWUSR)
        # Restore directory attributes after staging has created all children.
        for node in reversed(entry['nodes']):
            if node['directory']:
                shutil.copystat(src / node['path'], dst / node['path'])
                (dst / node['path']).chmod(stat.S_IMODE(node['mode']) | stat.S_IRWXU)
        if backup.exists() or backup.is_symlink():
            raise FileExistsError(backup)
        src.rename(backup)
        moved = True
        src.symlink_to(dst, target_is_directory=True)
        if _digest(backup, []) != entry['sha256']:
            raise ValueError('Original tree changed during overlay adoption')
        for name, canonical in entry['provider_files'].items():
            real = Path(canonical['resolved'])
            if entry.get('provider_mode') == 'apfs_clone':
                if (src / name).is_symlink() or (src / name).samefile(real) or _digest(src / name, []) != canonical['sha256']:
                    raise ValueError('Writable provider clone changed during adoption')
            elif (src / name).resolve(strict=True) != real:
                raise ValueError('Provider pointer changed during overlay adoption')
            if _digest(real, []) != canonical['sha256']:
                raise ValueError('Canonical provider bytes changed during adoption')
        receipt['state'] = 'complete_original_tree_retained'
        _write_receipt(receipt_file, receipt)
    except BaseException:
        if moved:
            if src.is_symlink() and os.readlink(src) == str(dst):
                src.unlink()
            if src.exists() or src.is_symlink():
                _write_receipt(receipt_file, {**receipt, 'state': 'recovery_required'})
                raise RuntimeError('Original source occupied; recovery and overlay retained')
            backup.rename(src)
        if created_overlay:
            shutil.rmtree(dst)
        _write_receipt(receipt_file, {**receipt, 'state': 'rolled_back'})
        raise
    return receipt


def _clone_file(source: Path, target: Path) -> None:
    """Invoke clonefile(2) directly: cp -c silently falls back to full copies."""
    if sys.platform != 'darwin':
        raise OSError(errno.ENOTSUP, 'Writable provider views require APFS clonefile support')
    library = ctypes.CDLL(None, use_errno=True)
    clone = library.clonefile
    clone.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32)
    clone.restype = ctypes.c_int
    if clone(os.fsencode(source), os.fsencode(target), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def _probe_clone(parent: Path) -> None:
    """Prove the actual destination filesystem isolates clone writes."""
    with tempfile.TemporaryDirectory(prefix='.vafx-clone-probe-', dir=parent) as temporary:
        root = Path(temporary)
        original, clone = root / 'original', root / 'clone'
        original.write_bytes(b'original clone probe')
        _clone_file(original, clone)
        if original.samefile(clone):
            raise ValueError('Filesystem clone unexpectedly shares the original inode')
        clone.write_bytes(b'changed independent clone')
        if original.read_bytes() != b'original clone probe':
            raise ValueError('Filesystem clone failed copy-on-write isolation')
