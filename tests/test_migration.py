# this_file: tests/test_migration.py
"""Exercise relocation on real temporary files without accessing real models."""
import json
from pathlib import Path

import pytest

from vexy_audioflexy.migration import apply_entry, reviewed_entry


def bundle(tmp_path):
    source = tmp_path / 'source with spaces'
    source.mkdir()
    (source / 'weights.bin').write_bytes(bytes(range(256)))
    return source, tmp_path / 'target'


def prepare(source, target):
    return reviewed_entry(str(source), str(target), provider='local',
                          classification='model_bundle', reviewed=True)


def test_apply_preserves_bytes_and_compatibility(tmp_path):
    source, target = bundle(tmp_path)
    entry = prepare(source, target)
    result = apply_entry(entry, str(tmp_path / 'receipt.json'))
    assert source.is_symlink() and source.resolve() == target
    assert (source / 'weights.bin').read_bytes() == bytes(range(256))
    assert result['state'] == 'complete'


def test_link_failure_rolls_back(tmp_path, monkeypatch):
    source, target = bundle(tmp_path)
    entry = prepare(source, target)
    def fail(*args, **kwargs):
        raise PermissionError('links disabled')
    monkeypatch.setattr(Path, 'symlink_to', fail)
    with pytest.raises(PermissionError):
        apply_entry(entry, str(tmp_path / 'receipt.json'))
    assert source.is_dir() and not source.is_symlink() and not target.exists()
    assert (source / 'weights.bin').read_bytes() == bytes(range(256))
    assert json.loads((tmp_path / 'receipt.json').read_text())['state'] == 'rolled_back'


@pytest.mark.parametrize('kind', ['collision', 'recursive', 'link', 'private'])
def test_rejects_unsafe_paths(tmp_path, kind):
    source, target = bundle(tmp_path)
    if kind == 'collision':
        target.mkdir()
    elif kind == 'recursive':
        target = source / 'inside'
    elif kind == 'link':
        (source / 'cycle').symlink_to(source)
    else:
        (source / '.venv').mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        prepare(source, target)
    assert (source / 'weights.bin').exists()


def test_rejects_changed_content_and_existing_receipt(tmp_path):
    source, target = bundle(tmp_path)
    entry = prepare(source, target)
    (source / 'weights.bin').write_bytes(b'changed')
    with pytest.raises(ValueError):
        apply_entry(entry, str(tmp_path / 'receipt.json'))
    entry = prepare(source, target)
    (tmp_path / 'receipt.json').write_text('prior operation')
    with pytest.raises(FileExistsError):
        apply_entry(entry, str(tmp_path / 'receipt.json'))
    assert source.exists() and not target.exists()


@pytest.mark.parametrize('provider,classification,reviewed', [
    ('huggingface', 'model_bundle', True), ('lmstudio', 'model_bundle', True),
    ('local', 'mixed', True), ('local', 'model_bundle', False)])
def test_requires_reviewed_nonprovider_bundle(tmp_path, provider, classification, reviewed):
    source, target = bundle(tmp_path)
    with pytest.raises(ValueError):
        reviewed_entry(str(source), str(target), provider=provider,
                       classification=classification, reviewed=reviewed)


def test_completion_receipt_failure_rolls_back(tmp_path, monkeypatch):
    import vexy_audioflexy.migration as migration
    source, target = bundle(tmp_path)
    entry = prepare(source, target)
    original = migration._write_receipt
    def fail_complete(path, receipt):
        if receipt['state'] == 'complete':
            raise OSError('receipt storage unavailable')
        original(path, receipt)
    monkeypatch.setattr(migration, '_write_receipt', fail_complete)
    with pytest.raises(OSError):
        apply_entry(entry, str(tmp_path / 'receipt.json'))
    assert source.is_dir() and not source.is_symlink() and not target.exists()
    assert (source / 'weights.bin').read_bytes() == bytes(range(256))


def test_receipt_inside_bundle_rejected(tmp_path):
    source, target = bundle(tmp_path)
    with pytest.raises(ValueError):
        apply_entry(prepare(source, target), str(source / 'receipt.json'))
    assert source.exists() and not target.exists()


def test_dangling_requested_target_is_rejected_without_mutation(tmp_path):
    source, target = bundle(tmp_path)
    redirected = tmp_path / 'unrelated destination'
    target.symlink_to(redirected)
    with pytest.raises(FileExistsError):
        prepare(source, target)
    assert source.is_dir() and not source.is_symlink()
    assert (source / 'weights.bin').read_bytes() == bytes(range(256))
    assert target.is_symlink() and target.readlink() == redirected
    assert not redirected.exists()


def duplicate(tmp_path):
    from vexy_audioflexy.migration import reviewed_duplicate
    source, canonical, recovery = [tmp_path / n for n in ('source', 'canonical', 'recovery')]
    source.write_bytes(b'unchanged model')
    canonical.write_bytes(source.read_bytes())
    entry = reviewed_duplicate(str(source), str(canonical), str(recovery),
                               identity='publisher/model@revision/file', reviewed=True)
    return source, canonical, recovery, entry


def test_duplicate_retains_recovery_and_provider_bytes(tmp_path):
    from vexy_audioflexy.migration import apply_duplicate
    source, canonical, recovery, entry = duplicate(tmp_path)
    result = apply_duplicate(entry, str(tmp_path / 'receipt.json'))
    assert source.is_symlink() and source.resolve() == canonical
    assert source.read_bytes() == recovery.read_bytes() == canonical.read_bytes() == b'unchanged model'
    assert result['state'] == 'complete_recovery_retained'


def test_duplicate_link_failure_rolls_back_without_provider_change(tmp_path, monkeypatch):
    from vexy_audioflexy.migration import apply_duplicate
    source, canonical, recovery, entry = duplicate(tmp_path)
    def fail(*args, **kwargs):
        raise PermissionError('no links')
    monkeypatch.setattr(Path, 'symlink_to', fail)
    with pytest.raises(PermissionError):
        apply_duplicate(entry, str(tmp_path / 'receipt.json'))
    assert source.read_bytes() == canonical.read_bytes() == b'unchanged model'
    assert not source.is_symlink() and not recovery.exists()


@pytest.mark.parametrize('change', ['provider_bytes', 'recovery_collision', 'canonical_retarget'])
def test_duplicate_rejects_changed_entry_before_mutation(tmp_path, change):
    from vexy_audioflexy.migration import apply_duplicate
    source, canonical, recovery, entry = duplicate(tmp_path)
    if change == 'provider_bytes':
        canonical.write_bytes(b'changed')
    elif change == 'recovery_collision':
        recovery.write_bytes(b'owner data')
    else:
        blob = tmp_path / 'blob'
        blob.write_bytes(canonical.read_bytes())
        canonical.unlink()
        canonical.symlink_to(blob)
    with pytest.raises((ValueError, FileExistsError)):
        apply_duplicate(entry, str(tmp_path / 'receipt.json'))
    assert source.read_bytes() == b'unchanged model' and not source.is_symlink()


def test_duplicate_accepts_provider_snapshot_symlink(tmp_path):
    from vexy_audioflexy.migration import reviewed_duplicate, apply_duplicate
    source, canonical, recovery, entry = duplicate(tmp_path)
    snapshot = tmp_path / 'snapshot-file'
    snapshot.symlink_to(canonical)
    entry = reviewed_duplicate(str(source), str(snapshot), str(recovery),
                               identity='publisher/model@revision/file', reviewed=True)
    apply_duplicate(entry, str(tmp_path / 'receipt.json'))
    assert source.readlink() == snapshot and source.resolve() == canonical
    assert canonical.read_bytes() == recovery.read_bytes() == b'unchanged model'


def test_duplicate_receipt_failure_rolls_back_and_preserves_canonical(tmp_path, monkeypatch):
    import vexy_audioflexy.migration as migration
    source, canonical, recovery, entry = duplicate(tmp_path)
    original = migration._write_receipt
    def fail_complete(path, receipt):
        if receipt['state'] == 'complete_recovery_retained':
            raise OSError('receipt unavailable')
        original(path, receipt)
    monkeypatch.setattr(migration, '_write_receipt', fail_complete)
    with pytest.raises(OSError):
        migration.apply_duplicate(entry, str(tmp_path / 'receipt.json'))
    assert source.read_bytes() == canonical.read_bytes() == b'unchanged model'
    assert not source.is_symlink() and not recovery.exists()
    assert json.loads((tmp_path / 'receipt.json').read_text())['state'] == 'rolled_back'


def hf_seed(tmp_path):
    import hashlib
    from vexy_audioflexy.migration import reviewed_hf_seed
    source = tmp_path / 'downloaded-model'
    source.write_bytes(b'verified HF bytes')
    hub = tmp_path / 'hub'
    hub.mkdir()
    return reviewed_hf_seed(str(source), str(hub), repo='publisher/model',
                            revision='a' * 40, filename='nested/weights.bin',
                            etag=hashlib.sha256(source.read_bytes()).hexdigest(),
                            size=source.stat().st_size, reviewed=True)


def test_hf_seed_preserves_source_and_installs_native_topology(tmp_path):
    from vexy_audioflexy.migration import apply_hf_seed
    entry = hf_seed(tmp_path)
    result = apply_hf_seed(entry, str(tmp_path / 'receipt.json'))
    source, blob, snapshot = [Path(entry[k]) for k in ('source', 'blob', 'snapshot')]
    assert source.samefile(blob) and snapshot.resolve() == blob
    assert not source.is_symlink() and source.read_bytes() == b'verified HF bytes'
    assert result['state'] == 'complete_source_retained'


def test_hf_seed_pointer_failure_rolls_back_created_blob(tmp_path, monkeypatch):
    from vexy_audioflexy.migration import apply_hf_seed
    entry = hf_seed(tmp_path)
    def fail(*args, **kwargs):
        raise PermissionError('no symlinks')
    monkeypatch.setattr(Path, 'symlink_to', fail)
    with pytest.raises(PermissionError):
        apply_hf_seed(entry, str(tmp_path / 'receipt.json'))
    assert Path(entry['source']).read_bytes() == b'verified HF bytes'
    assert not Path(entry['blob']).exists() and not Path(entry['snapshot']).is_symlink()


@pytest.mark.parametrize('conflict', ['blob', 'snapshot', 'source'])
def test_hf_seed_rejects_conflicts_before_mutation(tmp_path, conflict):
    from vexy_audioflexy.migration import apply_hf_seed
    entry = hf_seed(tmp_path)
    path = Path(entry[conflict]);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'owner content')
    with pytest.raises(ValueError):
        apply_hf_seed(entry, str(tmp_path / 'receipt.json'))
    assert path.read_bytes() == b'owner content'


def test_hf_seed_pointer_failure_preserves_existing_provider_blob(tmp_path, monkeypatch):
    from vexy_audioflexy.migration import apply_hf_seed
    entry = hf_seed(tmp_path)
    blob = Path(entry['blob']);blob.parent.mkdir(parents=True)
    blob.write_bytes(Path(entry['source']).read_bytes())
    def fail(*args, **kwargs):
        raise PermissionError('no symlinks')
    monkeypatch.setattr(Path, 'symlink_to', fail)
    with pytest.raises(PermissionError):
        apply_hf_seed(entry, str(tmp_path / 'receipt.json'))
    assert blob.read_bytes() == Path(entry['source']).read_bytes() == b'verified HF bytes'


def test_hf_seed_receipt_failure_rolls_back_only_new_cache_entries(tmp_path, monkeypatch):
    import vexy_audioflexy.migration as migration
    entry = hf_seed(tmp_path)
    original = migration._write_receipt
    def fail_complete(path, receipt):
        if receipt['state'] == 'complete_source_retained':
            raise OSError('receipt unavailable')
        original(path, receipt)
    monkeypatch.setattr(migration, '_write_receipt', fail_complete)
    with pytest.raises(OSError):
        migration.apply_hf_seed(entry, str(tmp_path / 'receipt.json'))
    assert Path(entry['source']).read_bytes() == b'verified HF bytes'
    assert not Path(entry['blob']).exists() and not Path(entry['snapshot']).is_symlink()


def test_hf_seed_rejects_unbound_metadata_and_traversal(tmp_path):
    from vexy_audioflexy.migration import reviewed_hf_seed
    entry = hf_seed(tmp_path)
    fields = {k: entry[k] for k in ('repo', 'revision', 'filename', 'etag', 'size', 'reviewed')}
    for overrides in ({'etag': 'b' * 64}, {'filename': '../outside'}, {'revision': 'main'},
                      {'repo': '../model'}, {'reviewed': False}):
        with pytest.raises(ValueError):
            reviewed_hf_seed(entry['source'], entry['hub'], **{**fields, **overrides})


def test_runtime_moves_git_environment_bytes_and_internal_link(tmp_path):
    from vexy_audioflexy.migration import reviewed_runtime_entry, apply_entry
    source = tmp_path / 'runtime'
    (source / '.git').mkdir(parents=True)
    (source / '.git/config').write_bytes(b'owner configuration')
    (source / '.venv').mkdir()
    (source / '.venv/tool').write_bytes(b'#!/original/path/python\n')
    (source / 'library.3.dylib').write_bytes(b'native bytes')
    (source / 'library.dylib').symlink_to('library.3.dylib')
    links = [{'path': 'library.dylib', 'target': 'library.3.dylib'}]
    entry = reviewed_runtime_entry(str(source), str(tmp_path / 'target'), symlinks=links, reviewed=True)
    apply_entry(entry, str(tmp_path / 'receipt.json'))
    assert source.is_symlink()
    assert (source / 'library.dylib').read_bytes() == b'native bytes'
    assert (source / '.git/config').read_bytes() == b'owner configuration'
    assert (source / '.venv/tool').read_bytes() == b'#!/original/path/python\n'


def test_runtime_rejects_unlisted_or_external_relative_links(tmp_path):
    from vexy_audioflexy.migration import reviewed_runtime_entry
    source = tmp_path / 'runtime';source.mkdir()
    external = tmp_path / 'external';external.write_bytes(b'owner bytes')
    (source / 'link').symlink_to('../external')
    with pytest.raises(ValueError):
        reviewed_runtime_entry(str(source), str(tmp_path / 'target'),
                               symlinks=[{'path': 'link', 'target': '../external'}], reviewed=True)
    (source / 'link').unlink();(source / 'file').write_bytes(b'bytes')
    (source / 'link').symlink_to('file')
    with pytest.raises(ValueError):
        reviewed_runtime_entry(str(source), str(tmp_path / 'target'), symlinks=[], reviewed=True)


def test_runtime_compatibility_link_failure_rolls_back_tree_links(tmp_path, monkeypatch):
    from vexy_audioflexy.migration import reviewed_runtime_entry, apply_entry
    source = tmp_path / 'runtime';source.mkdir()
    (source / 'real.dylib').write_bytes(b'library')
    (source / 'alias.dylib').symlink_to('real.dylib')
    entry = reviewed_runtime_entry(str(source), str(tmp_path / 'target'),
                                   symlinks=[{'path': 'alias.dylib', 'target': 'real.dylib'}], reviewed=True)
    def fail(*args, **kwargs):
        raise PermissionError('no compatibility link')
    monkeypatch.setattr(Path, 'symlink_to', fail)
    with pytest.raises(PermissionError):
        apply_entry(entry, str(tmp_path / 'receipt.json'))
    assert source.is_dir() and not source.is_symlink()
    assert (source / 'alias.dylib').readlink() == Path('real.dylib')
    assert (source / 'alias.dylib').read_bytes() == b'library'


@pytest.mark.parametrize('repo', ['owner--other/model', 'owner/other--model', 'owner/..',
                                 '--/model', 'owner/mödel', 'owner/' + 'a' * 97,
                                 'owner/model.git', '-owner/model', 'owner/model.'])
def test_hf_seed_rejects_invalid_repo_namespace_without_cache_mutation(tmp_path, repo):
    from vexy_audioflexy.migration import reviewed_hf_seed
    entry = hf_seed(tmp_path)
    fields = {k: entry[k] for k in ('revision', 'filename', 'etag', 'size', 'reviewed')}
    before = list(Path(entry['hub']).iterdir())
    with pytest.raises(ValueError):
        reviewed_hf_seed(entry['source'], entry['hub'], repo=repo, **fields)
    assert list(Path(entry['hub']).iterdir()) == before
    assert Path(entry['source']).read_bytes() == b'verified HF bytes'


def test_runtime_rejects_fifo_symlink_without_opening_special_file(tmp_path):
    import os
    from vexy_audioflexy.migration import reviewed_runtime_entry
    fifo = tmp_path / 'external-fifo'
    os.mkfifo(fifo)
    source = tmp_path / 'runtime';source.mkdir()
    (source / 'fifo-link').symlink_to(fifo)
    with pytest.raises(ValueError, match='regular files'):
        reviewed_runtime_entry(str(source), str(tmp_path / 'target'),
                               symlinks=[{'path': 'fifo-link', 'target': str(fifo)}], reviewed=True)
    assert (source / 'fifo-link').is_symlink()
    assert not (tmp_path / 'target').exists()


@pytest.mark.parametrize('repo', ['owner/' + 'a' * 96, 'a' * 96 + '/model',
                                 'a' * 96 + '/' + 'b' * 96])
def test_hf_seed_accepts_valid_96_character_components(tmp_path, repo):
    from vexy_audioflexy.migration import reviewed_hf_seed
    entry = hf_seed(tmp_path)
    fields = {k: entry[k] for k in ('revision', 'filename', 'etag', 'size', 'reviewed')}
    result = reviewed_hf_seed(entry['source'], entry['hub'], repo=repo, **fields)
    assert result['repo'] == repo
    assert not Path(result['blob']).exists()


def test_runtime_literal_link_change_rejects_reviewed_entry(tmp_path):
    import os
    from vexy_audioflexy.migration import reviewed_runtime_entry, _digest
    source, target = bundle(tmp_path)
    link = source / 'alias'
    os.symlink('./weights.bin', link)
    manifest = [{'path': 'alias', 'target': './weights.bin'}]
    entry = reviewed_runtime_entry(str(source), str(target), symlinks=manifest, reviewed=True)
    before = _digest(source, manifest)
    link.unlink()
    os.symlink('weights.bin', link)
    assert _digest(source, manifest) != before
    with pytest.raises(ValueError):
        apply_entry(entry, str(tmp_path / 'receipt.json'))
    assert not target.exists() and os.readlink(link) == 'weights.bin'


def test_digest_preserves_dot_and_trailing_slash_literal(tmp_path):
    import os
    from vexy_audioflexy.migration import _digest
    source = tmp_path / 'tree';source.mkdir()
    (source / 'target').mkdir()
    link = source / 'link'
    os.symlink('./target/', link)
    before = _digest(source, [])
    link.unlink();os.symlink('target', link)
    assert _digest(source, []) != before


@pytest.mark.parametrize('duplicate', [False, True])
def test_rollback_preserves_owner_changed_literal_alias(tmp_path, monkeypatch, duplicate):
    import os
    import vexy_audioflexy.migration as migration
    source, target = bundle(tmp_path)
    if duplicate:
        source = source / 'weights.bin'
        target.write_bytes(source.read_bytes())
        recovery = tmp_path / 'recovery'
        entry = migration.reviewed_duplicate(str(source), str(target), str(recovery),
                                             identity='test/model', reviewed=True)
        apply = migration.apply_duplicate
    else:
        entry = prepare(source, target)
        apply = apply_entry
    write = migration._write_receipt
    raw = str(target.parent) + '/./' + target.name
    def fail_after_owner_change(path, receipt):
        if receipt['state'].startswith('complete'):
            source.unlink();os.symlink(raw, source)
            raise OSError('receipt failure after owner changed alias')
        write(path, receipt)
    monkeypatch.setattr(migration, '_write_receipt', fail_after_owner_change)
    with pytest.raises(RuntimeError, match='occupied'):
        apply(entry, str(tmp_path / 'receipt.json'))
    assert os.readlink(source) == raw
    assert target.exists()
