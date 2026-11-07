# this_file: tests/test_runtime_links.py
"""Synthetic filesystem acceptance for explicit runtime link relocation."""

import json
import os
from pathlib import Path

import pytest
from vexy_audioflexy import runtime_links as links


def prepare(tmp_path):
    src = tmp_path / 'old' / 'runtime'
    src.mkdir(parents=True)
    parent = tmp_path / 'new'
    parent.mkdir()
    return src, parent / 'runtime', tmp_path / 'receipt.json'


def plan(src, dst, rewrites=None):
    manifest = sorted([{'path': str(p.relative_to(src)), 'target': os.readlink(p)}
                       for p in links._entries(src) if p.is_symlink()],
                      key=lambda x: x['path'])
    return links.reviewed_runtime_link_entry(str(src), str(dst), symlinks=manifest,
                                             rewrites=rewrites or {}, reviewed=True)


def test_move_preserves_bytes_owner_edits_modes_and_link_shapes(tmp_path):
    src, dst, receipt = prepare(tmp_path)
    (src / '.git').mkdir()
    (src / '.git' / 'owner').write_bytes(b'owner edits\x00')
    (src / '.venv').mkdir()
    (src / '.venv' / 'python').write_bytes(b'executable')
    (src / '.venv' / 'python').chmod(0o751)
    (src / 'internal-dir').symlink_to('.venv', target_is_directory=True)
    (src / 'missing').symlink_to('absent')
    (src / 'cycle-a').symlink_to('cycle-b')
    (src / 'cycle-b').symlink_to('cycle-a')
    external = tmp_path / 'external'
    external.mkdir()
    (external / 'private').write_bytes(b'must not hash or copy')
    (src / 'external').symlink_to(external, target_is_directory=True)
    entry = plan(src, dst)
    result = links.apply_runtime_link_entry(entry, str(receipt))
    assert result['state'] == 'complete'
    assert src.is_symlink() and src.resolve() == dst
    assert (dst / '.git' / 'owner').read_bytes() == b'owner edits\x00'
    assert (dst / '.venv' / 'python').stat().st_mode & 0o777 == 0o751
    assert result['pre_raw_targets'] == result['post_raw_targets']
    assert not any('private' in str(p.relative_to(dst)) for p in links._entries(dst))
    assert result['post_semantics'] == entry['semantics']


def test_relative_external_requires_logical_not_resolved_rewrite(tmp_path):
    src, dst, receipt = prepare(tmp_path)
    external = tmp_path / 'python-version'
    external.write_bytes(b'python')
    (src / 'python').symlink_to(external)
    (src / 'python3').symlink_to('python')
    with pytest.raises(ValueError, match='old logical'):
        plan(src, dst)
    with pytest.raises(ValueError, match='old logical'):
        plan(src, dst, {'python3': str(external)})
    entry = plan(src, dst, {'python3': str(src / 'python')})
    result = links.apply_runtime_link_entry(entry, str(receipt))
    assert (dst / 'python3').readlink() == src / 'python'
    assert (src / 'python3').resolve() == external
    assert result['post_raw_targets']['python3'] == str(src / 'python')


def test_relative_external_directory_and_missing_target_rebase(tmp_path):
    src, dst, receipt = prepare(tmp_path)
    (src.parent / 'sibling').mkdir()
    (src / 'directory').symlink_to('../sibling')
    (src / 'broken').symlink_to('../absent')
    rewrites = {name: str(src.parent / target)
                for name, target in [('directory', 'sibling'), ('broken', 'absent')]}
    entry = plan(src, dst, rewrites)
    links.apply_runtime_link_entry(entry, str(receipt))
    assert (dst / 'directory').resolve() == src.parent / 'sibling'
    assert not (dst / 'broken').exists()


@pytest.mark.parametrize('failure', ['rewrite', 'verify', 'receipt'])
def test_failure_restores_original_literals_source_and_bytes(tmp_path, monkeypatch, failure):
    src, dst, receipt = prepare(tmp_path)
    external = tmp_path / 'interpreter'
    external.write_bytes(b'external')
    (src / 'python').symlink_to(external)
    (src / 'python3').symlink_to('python')
    (src / 'owner').write_bytes(b'unchanged')
    entry = plan(src, dst, {'python3': str(src / 'python')})
    if failure == 'rewrite':
        original = Path.symlink_to
        def fail(self, target, **kwargs):
            if self == dst / 'python3' and str(target) != 'python':
                raise OSError('injected rewrite failure')
            return original(self, target, **kwargs)
        monkeypatch.setattr(Path, 'symlink_to', fail)
    elif failure == 'verify':
        original = links._fingerprint
        def fail(root, original_links=None):
            if root == dst:
                raise OSError('injected verification failure')
            return original(root, original_links)
        monkeypatch.setattr(links, '_fingerprint', fail)
    else:
        original = links._write_receipt
        def fail(path, data):
            if data['state'] == 'complete':
                raise OSError('injected receipt failure')
            return original(path, data)
        monkeypatch.setattr(links, '_write_receipt', fail)
    with pytest.raises(OSError, match='injected'):
        links.apply_runtime_link_entry(entry, str(receipt))
    assert not src.is_symlink() and src.is_dir() and not dst.exists()
    assert (src / 'python3').readlink() == Path('python')
    assert (src / 'owner').read_bytes() == b'unchanged'
    assert json.loads(receipt.read_text())['state'] == 'rolled_back'


def test_standalone_runtime_file_preserves_executable_and_bytes(tmp_path):
    src, dst, receipt = prepare(tmp_path)
    src.rmdir()
    src.write_bytes(b'#!/bin/sh\necho synthetic\n')
    src.chmod(0o750)
    entry = plan(src, dst)
    assert entry['classification'] == 'runtime_file'
    links.apply_runtime_link_entry(entry, str(receipt))
    assert src.is_symlink() and dst.read_bytes() == b'#!/bin/sh\necho synthetic\n'
    assert dst.stat().st_mode & 0o777 == 0o750


def test_changed_manifest_content_and_collisions_refuse_without_move(tmp_path):
    src, dst, receipt = prepare(tmp_path)
    (src / 'owner').write_text('first')
    (src / 'link').symlink_to('owner')
    with pytest.raises(ValueError, match='manifest'):
        links.reviewed_runtime_link_entry(str(src), str(dst), symlinks=[], rewrites={}, reviewed=True)
    entry = plan(src, dst)
    (src / 'owner').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        links.apply_runtime_link_entry(entry, str(receipt))
    entry = plan(src, dst)
    receipt.write_text('existing')
    with pytest.raises(FileExistsError):
        links.apply_runtime_link_entry(entry, str(receipt))
    receipt.unlink()
    dst.symlink_to('missing')
    with pytest.raises(FileExistsError):
        links.apply_runtime_link_entry(entry, str(receipt))
    assert src.is_dir() and not src.is_symlink()


def test_fifo_physical_entry_rejected_without_following_external(tmp_path):
    import os
    src, dst, _ = prepare(tmp_path)
    os.mkfifo(src / 'pipe')
    with pytest.raises(ValueError, match='Unsupported physical'):
        plan(src, dst)


def test_absolute_internal_alias_and_rewritten_link_metadata(tmp_path):
    src, dst, receipt = prepare(tmp_path)
    (src / 'owner').write_bytes(b'owner')
    original_inode = (src / 'owner').stat().st_ino
    (src / 'absolute').symlink_to(src / 'owner')
    external = tmp_path / 'interpreter'
    external.write_bytes(b'interpreter')
    (src / 'python').symlink_to(external)
    (src / 'python3').symlink_to('python')
    original = (src / 'python3').lstat()
    entry = plan(src, dst, {'python3': str(src / 'python')})
    links.apply_runtime_link_entry(entry, str(receipt))
    assert (dst / 'absolute').resolve() == dst / 'owner'
    assert (dst / 'owner').stat().st_ino == original_inode
    after = (dst / 'python3').lstat()
    assert (after.st_uid, after.st_gid, after.st_mode, after.st_mtime_ns) == (
        original.st_uid, original.st_gid, original.st_mode, original.st_mtime_ns)


def test_obstructed_recovery_retains_target_and_explicit_receipt(tmp_path, monkeypatch):
    src, dst, receipt = prepare(tmp_path)
    (src / 'owner').write_bytes(b'owner')
    entry = plan(src, dst)
    original = links._fingerprint
    def fail(root, original_links=None):
        if root == dst:
            src.unlink()
            src.write_bytes(b'concurrent obstruction')
            raise OSError('injected failure')
        return original(root, original_links)
    monkeypatch.setattr(links, '_fingerprint', fail)
    with pytest.raises(RuntimeError, match='Source occupied'):
        links.apply_runtime_link_entry(entry, str(receipt))
    assert src.read_bytes() == b'concurrent obstruction'
    assert (dst / 'owner').read_bytes() == b'owner'
    assert json.loads(receipt.read_text())['state'] == 'recovery_required'


def test_walk_error_rejects_plan_and_apply_before_any_move(tmp_path, monkeypatch):
    src, dst, receipt = prepare(tmp_path)
    (src / 'owner').write_bytes(b'complete bytes')
    entry = plan(src, dst)
    def inaccessible(root, *, followlinks, onerror):
        assert followlinks is False
        onerror(PermissionError('injected unreadable subtree'))
        yield root, [], []
    monkeypatch.setattr(links.os, 'walk', inaccessible)
    with pytest.raises(PermissionError, match='unreadable'):
        plan(src, dst)
    with pytest.raises(PermissionError, match='unreadable'):
        links.apply_runtime_link_entry(entry, str(receipt))
    assert (src / 'owner').read_bytes() == b'complete bytes'
    assert not src.is_symlink() and not dst.exists() and not receipt.exists()


def test_unreadable_subtree_cannot_hide_owner_bytes_or_external_links(tmp_path, monkeypatch):
    import os
    src, dst, receipt = prepare(tmp_path)
    hidden = src / 'unreadable'
    hidden.mkdir()
    (hidden / 'owner').write_bytes(b'owner edit')
    external = tmp_path / 'external'
    external.write_bytes(b'external')
    (hidden / 'pointer').symlink_to('../../../external')
    entry = plan(src, dst, {'unreadable/pointer': str(external)})
    hidden.chmod(0)
    try:
        if os.geteuid() == 0:
            # Root bypasses chmod; exercise the identical scanner error callback.
            def inaccessible(root, *, followlinks, onerror):
                onerror(PermissionError('root CI simulated chmod denial'))
                yield root, [], []
            monkeypatch.setattr(links.os, 'walk', inaccessible)
        with pytest.raises(PermissionError):
            links.reviewed_runtime_link_entry(str(src), str(dst), symlinks=[],
                                               rewrites={}, reviewed=True)
        with pytest.raises(PermissionError):
            links.apply_runtime_link_entry(entry, str(receipt))
        assert not src.is_symlink() and not dst.exists() and not receipt.exists()
    finally:
        hidden.chmod(0o755)
    assert (hidden / 'owner').read_bytes() == b'owner edit'
    assert (hidden / 'pointer').readlink() == Path('../../../external')


@pytest.mark.parametrize('raw', ['./target/', 'target/', './target'])
def test_runtime_accepts_and_preserves_os_raw_directory_target(tmp_path, raw):
    src, dst, receipt = prepare(tmp_path)
    (src / 'target').mkdir()
    (src / 'link').symlink_to(raw)
    inode = (src / 'link').lstat().st_ino
    entry = links.reviewed_runtime_link_entry(
        str(src), str(dst), symlinks=[{'path': 'link', 'target': raw}],
        rewrites={}, reviewed=True)
    result = links.apply_runtime_link_entry(entry, str(receipt))
    assert os.readlink(dst / 'link') == raw
    assert (dst / 'link').lstat().st_ino == inode
    assert result['pre_raw_targets'] == result['post_raw_targets'] == {'link': raw}


def test_runtime_lexical_link_drift_changes_digest_and_rejects_apply(tmp_path):
    src, dst, receipt = prepare(tmp_path)
    (src / 'target').mkdir()
    (src / 'link').symlink_to('./target/')
    entry = plan(src, dst)
    before = links._fingerprint(src)
    (src / 'link').unlink()
    (src / 'link').symlink_to('target')
    assert links._fingerprint(src) != before
    with pytest.raises(ValueError, match='manifest'):
        links.apply_runtime_link_entry(entry, str(receipt))
    assert src.is_dir() and not src.is_symlink()
    assert not dst.exists() and not receipt.exists()


def test_runtime_rollback_restores_exact_external_link_literal(tmp_path, monkeypatch):
    src, dst, receipt = prepare(tmp_path)
    external = src.parent / 'external'
    external.mkdir()
    (src / 'owner').write_bytes(b'owner')
    raw = './../external/'
    (src / 'link').symlink_to(raw)
    entry = plan(src, dst, {'link': str(external)})
    original = links._fingerprint
    def fail(root, original_links=None):
        if root == dst:
            raise OSError('injected after raw rewrite')
        return original(root, original_links)
    monkeypatch.setattr(links, '_fingerprint', fail)
    with pytest.raises(OSError, match='raw rewrite'):
        links.apply_runtime_link_entry(entry, str(receipt))
    assert os.readlink(src / 'link') == raw
    assert (src / 'owner').read_bytes() == b'owner'
    assert not src.is_symlink() and not dst.exists()
    assert json.loads(receipt.read_text())['state'] == 'rolled_back'


def test_runtime_rollback_keeps_owner_changed_lexical_source_alias(tmp_path, monkeypatch):
    src, dst, receipt = prepare(tmp_path)
    (src / 'owner').write_bytes(b'owner')
    entry = plan(src, dst)
    owner_literal = str(dst) + '/.'
    original = links._fingerprint
    def fail(root, original_links=None):
        if root == dst:
            src.unlink()
            src.symlink_to(owner_literal)
            raise OSError('injected after owner alias change')
        return original(root, original_links)
    monkeypatch.setattr(links, '_fingerprint', fail)
    with pytest.raises(RuntimeError, match='Source occupied'):
        links.apply_runtime_link_entry(entry, str(receipt))
    assert os.readlink(src) == owner_literal
    assert (dst / 'owner').read_bytes() == b'owner'
    assert json.loads(receipt.read_text())['state'] == 'recovery_required'
