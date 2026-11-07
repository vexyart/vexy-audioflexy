# this_file: tests/test_overlays.py
"""Real filesystem overlay transactions on synthetic provider bytes."""
from pathlib import Path

import pytest

from vexy_audioflexy.overlays import apply_overlay, reviewed_overlay_entry


def prepare(tmp_path):
    source = tmp_path / 'legacy';source.mkdir()
    (source / 'weights').write_bytes(b'provider bytes')
    (source / '.cache').mkdir()
    (source / '.cache/metadata').write_bytes(b'original metadata')
    (source / 'compiled').write_bytes(b'compiled derivative')
    provider = tmp_path / 'provider';provider.write_bytes(b'provider bytes')
    entry = reviewed_overlay_entry(str(source), str(tmp_path / 'overlay'),
                                   str(tmp_path / 'recovery'),
                                   provider_files={'weights': str(provider)}, reviewed=True)
    return source, provider, entry


def test_overlay_preserves_originals_and_writable_caller_directories(tmp_path):
    source, provider, entry = prepare(tmp_path)
    result = apply_overlay(entry, str(tmp_path / 'receipt.json'))
    assert source.is_symlink() and source.resolve() == tmp_path / 'overlay'
    assert (source / 'weights').resolve() == provider
    assert (source / '.cache/metadata').read_bytes() == b'original metadata'
    assert (source / 'compiled').read_bytes() == b'compiled derivative'
    (source / 'new-compiled').write_bytes(b'new output')
    replacement = source / 'new-weights';replacement.write_bytes(b'new revision')
    replacement.replace(source / 'weights')
    assert provider.read_bytes() == b'provider bytes'
    assert (tmp_path / 'recovery/weights').read_bytes() == b'provider bytes'
    assert (source / 'weights').read_bytes() == b'new revision'
    assert result['state'] == 'complete_original_tree_retained'


@pytest.mark.parametrize('failure', ['stage_link', 'legacy_link', 'receipt'])
def test_overlay_failures_restore_every_original_byte(tmp_path, monkeypatch, failure):
    import vexy_audioflexy.overlays as overlays
    source, provider, entry = prepare(tmp_path)
    original_link = Path.symlink_to
    original_receipt = overlays._write_receipt
    def link(path, *args, **kwargs):
        if failure == 'stage_link' or (failure == 'legacy_link' and path == source):
            raise PermissionError('link disabled')
        return original_link(path, *args, **kwargs)
    def receipt(path, value):
        if failure == 'receipt' and value['state'] == 'complete_original_tree_retained':
            raise OSError('receipt unavailable')
        return original_receipt(path, value)
    monkeypatch.setattr(Path, 'symlink_to', link)
    monkeypatch.setattr(overlays, '_write_receipt', receipt)
    with pytest.raises((PermissionError, OSError)):
        apply_overlay(entry, str(tmp_path / 'receipt.json'))
    assert source.is_dir() and not source.is_symlink()
    assert (source / 'weights').read_bytes() == provider.read_bytes() == b'provider bytes'
    assert (source / '.cache/metadata').read_bytes() == b'original metadata'
    assert not (tmp_path / 'overlay').exists() and not (tmp_path / 'recovery').exists()


@pytest.mark.parametrize('conflict', ['source', 'provider', 'overlay', 'recovery'])
def test_overlay_conflicts_fail_without_moving_original_tree(tmp_path, conflict):
    source, provider, entry = prepare(tmp_path)
    if conflict == 'source':(source / 'weights').write_bytes(b'changed')
    elif conflict == 'provider':provider.write_bytes(b'changed')
    else:Path(entry[conflict]).mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        apply_overlay(entry, str(tmp_path / 'receipt.json'))
    assert source.is_dir() and not source.is_symlink()


def test_overlay_rejects_traversal_overlaps_and_original_symlinks(tmp_path):
    source, provider, entry = prepare(tmp_path)
    for mapping in ({'../outside': str(provider)}, {'/absolute': str(provider)}):
        with pytest.raises(ValueError):
            reviewed_overlay_entry(str(source), entry['overlay'], entry['recovery'],
                                   provider_files=mapping, reviewed=True)
    with pytest.raises(ValueError):
        reviewed_overlay_entry(str(source), entry['overlay'], entry['overlay'],
                               provider_files={'weights': str(provider)}, reviewed=True)
    (source / 'alias').symlink_to('weights')
    with pytest.raises(ValueError):
        reviewed_overlay_entry(str(source), entry['overlay'], entry['recovery'],
                               provider_files={'weights': str(provider)}, reviewed=True)


def test_overlay_is_writable_while_recovery_preserves_original_modes(tmp_path):
    import stat
    source, provider, entry = prepare(tmp_path)
    (source / '.cache').chmod(0o500)
    (source / '.cache/metadata').chmod(0o400)
    entry = reviewed_overlay_entry(str(source), entry['overlay'], entry['recovery'],
                                   provider_files={'weights': str(provider)}, reviewed=True)
    apply_overlay(entry, str(tmp_path / 'receipt.json'))
    assert stat.S_IMODE((tmp_path / 'recovery/.cache').stat().st_mode) == 0o500
    assert stat.S_IMODE((source / '.cache').stat().st_mode) == 0o700
    (source / '.cache/new-output').write_bytes(b'writable')
    (source / '.cache/metadata').write_bytes(b'writable copy')
    assert stat.S_IMODE((tmp_path / 'recovery/.cache/metadata').stat().st_mode) == 0o400
    assert (tmp_path / 'recovery/.cache/metadata').read_bytes() == b'original metadata'


def test_overlay_in_place_metadata_and_compiled_writes_preserve_recovery(tmp_path):
    source, provider, entry = prepare(tmp_path)
    apply_overlay(entry, str(tmp_path / 'receipt.json'))
    metadata = source / '.cache/metadata'
    compiled = source / 'compiled'
    assert not metadata.is_symlink() and not compiled.is_symlink()
    metadata.write_bytes(b'new metadata')
    compiled.write_bytes(b'new compiled derivative')
    assert (tmp_path / 'recovery/.cache/metadata').read_bytes() == b'original metadata'
    assert (tmp_path / 'recovery/compiled').read_bytes() == b'compiled derivative'
    assert metadata.stat().st_ino != (tmp_path / 'recovery/.cache/metadata').stat().st_ino
    assert provider.read_bytes() == b'provider bytes'


def test_apfs_view_cached_revision_copyfile_does_not_modify_provider_or_recovery(tmp_path):
    import shutil
    import sys
    source, provider, entry = prepare(tmp_path)
    args = dict(source=str(source), overlay=entry['overlay'], recovery=entry['recovery'],
                provider_files={'weights': str(provider)}, reviewed=True, writable_provider=True)
    if sys.platform != 'darwin':
        with pytest.raises(OSError):
            reviewed_overlay_entry(**args)
        assert provider.read_bytes() == b'provider bytes'
        return
    entry = reviewed_overlay_entry(**args)
    apply_overlay(entry, str(tmp_path / 'receipt.json'))
    assert not (source / 'weights').is_symlink() and not (source / 'weights').samefile(provider)
    cached = tmp_path / 'new-cached-revision';cached.write_bytes(b'new cached weights')
    shutil.copyfile(cached, source / 'weights')
    assert (source / 'weights').read_bytes() == b'new cached weights'
    assert provider.read_bytes() == (tmp_path / 'recovery/weights').read_bytes() == b'provider bytes'


def test_apfs_probe_failure_has_no_full_copy_fallback(tmp_path, monkeypatch):
    import vexy_audioflexy.overlays as overlays
    source, provider, entry = prepare(tmp_path)
    def unsupported(*args):
        raise OSError('clone unsupported')
    def forbidden_fallback(*args, **kwargs):
        raise AssertionError('full-copy fallback forbidden')
    monkeypatch.setattr(overlays, '_clone_file', unsupported)
    monkeypatch.setattr(overlays.shutil, 'copy2', forbidden_fallback)
    with pytest.raises(OSError, match='clone unsupported'):
        reviewed_overlay_entry(str(source), entry['overlay'], entry['recovery'],
                               provider_files={'weights': str(provider)}, reviewed=True, writable_provider=True)
    assert source.is_dir() and not source.is_symlink()
    assert not Path(entry['overlay']).exists() and not Path(entry['recovery']).exists()


def test_apfs_stage_clone_failure_rolls_back_before_source_move(tmp_path, monkeypatch):
    import sys
    import vexy_audioflexy.overlays as overlays
    source, provider, entry = prepare(tmp_path)
    if sys.platform != 'darwin':
        with pytest.raises(OSError):
            reviewed_overlay_entry(str(source), entry['overlay'], entry['recovery'],
                                   provider_files={'weights': str(provider)}, reviewed=True, writable_provider=True)
        return
    entry = reviewed_overlay_entry(str(source), entry['overlay'], entry['recovery'],
                                   provider_files={'weights': str(provider)}, reviewed=True, writable_provider=True)
    original = overlays._clone_file
    def fail_provider(src, dst):
        if src == provider:
            raise OSError('provider clone failed')
        original(src, dst)
    monkeypatch.setattr(overlays, '_clone_file', fail_provider)
    with pytest.raises(OSError, match='provider clone failed'):
        apply_overlay(entry, str(tmp_path / 'receipt.json'))
    assert source.is_dir() and not source.is_symlink()
    assert provider.read_bytes() == (source / 'weights').read_bytes() == b'provider bytes'
    assert not Path(entry['overlay']).exists() and not Path(entry['recovery']).exists()


def test_apfs_rejects_hardlink_substitute_without_changing_provider(tmp_path, monkeypatch):
    import os
    import sys
    import vexy_audioflexy.overlays as overlays
    source, provider, entry = prepare(tmp_path)
    if sys.platform != 'darwin':
        with pytest.raises(OSError):
            reviewed_overlay_entry(str(source), entry['overlay'], entry['recovery'],
                                   provider_files={'weights': str(provider)}, reviewed=True, writable_provider=True)
        return
    entry = reviewed_overlay_entry(str(source), entry['overlay'], entry['recovery'],
                                   provider_files={'weights': str(provider)}, reviewed=True, writable_provider=True)
    original = overlays._clone_file
    def bad_clone(src, dst):
        if src == provider:
            os.link(src, dst)
        else:
            original(src, dst)
    monkeypatch.setattr(overlays, '_clone_file', bad_clone)
    with pytest.raises(ValueError, match='independent verified bytes'):
        apply_overlay(entry, str(tmp_path / 'receipt.json'))
    assert source.is_dir() and not source.is_symlink()
    assert provider.read_bytes() == b'provider bytes'
    assert not Path(entry['overlay']).exists()


def test_overlay_rollback_preserves_owner_changed_literal_alias(tmp_path, monkeypatch):
    import os
    import vexy_audioflexy.overlays as overlays
    source, provider, entry = prepare(tmp_path)
    write = overlays._write_receipt
    raw = str(tmp_path) + '/./overlay/'
    def fail_after_owner_change(path, receipt):
        if receipt['state'] == 'complete_original_tree_retained':
            source.unlink();os.symlink(raw, source)
            raise OSError('receipt failure after owner changed alias')
        write(path, receipt)
    monkeypatch.setattr(overlays, '_write_receipt', fail_after_owner_change)
    with pytest.raises(RuntimeError, match='occupied'):
        apply_overlay(entry, str(tmp_path / 'receipt.json'))
    assert os.readlink(source) == raw
    assert (tmp_path / 'recovery/weights').read_bytes() == b'provider bytes'
    assert provider.read_bytes() == b'provider bytes'
