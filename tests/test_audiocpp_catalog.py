# this_file: tests/test_audiocpp_catalog.py
"""Shared audio.cpp catalog: internal consistency, generator routing and registry wiring, without inference."""
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools"
CATALOG = TOOLS / "catalogs/audiocpp-83b1354.json"
REGISTRIES = {"tts_all": "tts/tts_all.py", "asr_all": "asr/asr_all.py", "aud_all": "fx/aud_all.py"}
GEN = runpy.run_path(str(TOOLS / "gen_audiocpp_catalog.py"))


def load_registry(name):
    path = TOOLS / REGISTRIES[name]
    for extra in (path.parent / "tools", path.parent):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    spec = importlib.util.spec_from_file_location(f"ac83_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def catalog():
    return json.loads(CATALOG.read_text())


@pytest.fixture(scope="module")
def registries():
    return {name: load_registry(name).MODELS for name in REGISTRIES}


def test_catalog_when_loaded_then_internally_consistent(catalog):
    assert len(catalog["source_revision"]) == 40
    families = [f["family"] for f in catalog["families"]]
    assert len(families) == len(set(families)), "duplicate family"
    for family in catalog["families"]:
        ids = {p["id"] for p in family["packages"]}
        assert family["status"] == "pending runtime"
        assert family["selected_package"] in ids | {None}, family["family"]
        assert {c.split(":", 1)[1] for c in family["covered_by"]} <= ids, family["family"]
        for route in family["routes"]:
            assert route["status"] in {"pending runtime", "verified"}
            assert (route["status"] == "verified") == ("verification" in route), family["family"]
            assert route["route"] in {"tts_all", "asr_all", "aud_all", "separate_all", "aud_analyze", "aud_generate", "covered", "pending adapter"}
            if route["route"] in REGISTRIES:
                assert route["package"] in ids, family["family"]
                assert set(route["langs"]) <= {"en", "pl"} and route["langs"]
            elif route["route"] == "pending adapter":
                assert route["reason"], family["family"]


def test_catalog_when_engine_present_then_matches_real_specs(catalog):
    root = Path(os.environ.get("AUDIOCPP_AC83_ROOT") or GEN["default_specs"]().parent)
    specs = root / "model_specs"
    if not specs.is_dir():
        pytest.skip("pinned audio.cpp engine not installed")
    real = {json.loads(p.read_text())["family"]: json.loads(p.read_text()) for p in specs.glob("*.json")}
    for family in catalog["families"]:
        assert family["family"] in real
        assert {p["id"] for p in family["packages"]} == {p["id"] for p in real[family["family"]].get("packages", [])}


def test_generator_when_fixture_specs_then_routes_by_task(tmp_path):
    def spec(family, tasks, request=(), packages=("p_q8_0",), **extra):
        return {"family": family, "tasks": list(tasks), "options": {"request": list(request)},
                "package_defaults": {"download": {"repo": "org/repo"}},
                "packages": [{"id": f"{family}_{p}", "precision": p.split("_", 1)[1]} for p in packages], **extra}
    fixtures = [
        spec("fx_voice", ["tts"], ui={"default_voice": "v1"}),
        spec("fx_clone", ["tts", "clone"], [{"name": "reference_text", "required": True}]),
        spec("fx_asr", ["asr"], [{"name": "language"}], languages=["pl"]),
        spec("fx_vc", ["vc"]),
        spec("fx_music", ["music"], [{"name": "lyrics", "required": True}]),
        spec("fx_ja", ["tts"], languages=["ja"]),
        spec("fx_need", ["tts"], [{"name": "style", "required": True}]),
        spec("fx_empty", ["tts"], packages=()),
        spec("fx_pick", ["asr"], packages=("x_f32", "x_q4_k")),
    ]
    for item in fixtures:
        (tmp_path / f"{item['family']}.json").write_text(json.dumps(item))
    built = {f["family"]: f for f in GEN["build"](tmp_path, "0" * 40, tmp_path)["families"]}
    route = lambda fam: built[fam]["routes"][0]
    assert route("fx_voice")["route"] == "tts_all" and route("fx_voice")["params"]["voice_id"] == "v1"
    assert {"mode": "clone", "task": "clon", "ref": True}.items() <= route("fx_clone")["params"].items()
    assert route("fx_asr")["params"]["lang_flag"] and route("fx_asr")["langs"] == ["pl"]
    assert route("fx_vc")["params"] == {"op": "vc", "task": "vc", "source": "speech", "target": "spkb"}
    assert route("fx_music")["route"] == "aud_generate" and route("fx_music")["params"] == {"task": "gen"}
    for fam in ("fx_ja", "fx_need", "fx_empty"):
        assert route(fam)["route"] == "pending adapter", fam
    assert route("fx_pick")["package"] == "fx_pick_x_q4_k"
    assert built["fx_voice"]["packages"][0]["revision"] is None


@pytest.mark.parametrize("name", list(REGISTRIES))
def test_registry_when_catalog_loaded_then_unique_names_and_pending_rows(name, registries, catalog):
    models = registries[name]
    names = [m["name"] for m in models]
    assert len(names) == len(set(names)), "duplicate registry name"
    expected = {r["package"] for f in catalog["families"] for r in f["routes"] if r["route"] == name}
    rows = [m for m in models if m["name"].startswith("ac83-")]
    assert {m["package"] for m in rows} == expected
    assert all(m["status"] in {"pending runtime", "verified"} and m["env"]["AUDIOCPP_ROOT"] for m in rows)
    assert all((TOOLS / REGISTRIES[name]).parent.joinpath(m["script"]).is_file() for m in rows)


def test_registry_when_family_covered_then_not_duplicated(registries, catalog):
    for family in catalog["families"]:
        for route in family["routes"]:
            if route["route"] == "covered" and route["registry"] in REGISTRIES:
                ids = {p["id"] for p in family["packages"]}
                generated = [m for m in registries[route["registry"]] if m["name"].startswith("ac83-") and m["package"] in ids]
                assert not generated, family["family"]


def test_registry_when_required_inputs_then_supplied(registries, catalog):
    by_package = {m["package"]: m for m in registries["tts_all"] if m["name"].startswith("ac83-")}
    for family in catalog["families"]:
        for route in family["routes"]:
            if route["route"] != "tts_all":
                continue
            args = by_package[route["package"]]["args_fn"]("hello", "ref.wav", "out.wav", "en")
            if "reference_text" in family["required_options"]:
                assert "--ref" in args and "--ref-text" in args, family["family"]
            if "instruct" in family["required_options"]:
                assert "--instruct" in args, family["family"]
            if route["params"]["voice_id"]:
                assert args[args.index("--voice-id") + 1] == route["params"]["voice_id"]


def test_verified_ledger_when_loaded_then_every_run_matches_a_routed_package(catalog):
    """Each ledger record names a registry:package the catalog routes, at the catalog's engine revision."""
    ledger = json.loads((TOOLS / "catalogs" / "audiocpp-verified.json").read_text())
    assert ledger["source_revision"] == catalog["source_revision"]
    routed = {}
    for family in catalog["families"]:
        for r in family["routes"]:
            routed[f"{r['route']}:{r.get('package')}"] = r
            routed[f"{r['route']}:{r.get('package')}:{r['tasks'][0]}"] = r  # task-suffixed keys
    for key, run in ledger["runs"].items():
        assert key in routed, f"ledger entry {key} is not a catalog route"
        assert routed[key]["status"] == "verified", key
        assert run["date"] and run["check"] and run["result"], key


def test_apply_verification_when_revision_differs_then_rows_stay_pending(tmp_path):
    """A ledger written for another engine revision must not mark rows verified."""
    gen = runpy.run_path(str(TOOLS / "gen_audiocpp_catalog.py"))
    (tmp_path / "catalogs").mkdir()
    (tmp_path / "catalogs" / "audiocpp-verified.json").write_text(json.dumps(
        {"source_revision": "a" * 40, "runs": {"asr_all:p": {"date": "d", "check": "c", "result": "r"}}}))
    families = [{"routes": [{"route": "asr_all", "package": "p", "status": "pending runtime"}]}]
    gen["apply_verification"](families, "b" * 40, tmp_path)
    assert families[0]["routes"][0]["status"] == "pending runtime"
    gen["apply_verification"](families, "a" * 40, tmp_path)
    assert families[0]["routes"][0]["status"] == "verified"


def test_installed_from_when_copy_fallback_then_prep_rerun_is_idempotent(tmp_path):
    provider = runpy.run_path(str(TOOLS / "_audiocpp_provider.py"))
    blob = tmp_path / "blob"
    blob.write_bytes(b"w" * 32)
    output = tmp_path / "model.gguf"
    import shutil
    shutil.copy2(blob, output)  # what link_model_file does across volumes
    assert provider["installed_from"](output, blob), "a copy2 of the blob must count as installed"
    output.write_bytes(b"x" * 33)
    assert not provider["installed_from"](output, blob), "a different file must not count as installed"


def test_link_model_file_when_copy_fails_then_no_partial_left(tmp_path):
    provider = runpy.run_path(str(TOOLS / "_audiocpp_provider.py"))
    with pytest.raises(OSError):
        provider["link_model_file"](tmp_path / "missing-blob", tmp_path / "model.gguf")
    assert not list(tmp_path.glob(".*.partial")), "failed link must clean up its temp file"


def test_separate_all_when_catalog_has_sep_routes_then_rows_are_accepted():
    module = runpy.run_path(str(TOOLS / "fx/separate_all.py"))
    rows = module["CATALOG_ROWS"]
    catalog = json.loads(CATALOG.read_text())
    sep = {r["package"] for f in catalog["families"] for r in f["routes"] if r["route"] == "separate_all"}
    assert sep and {row["package"] for row in rows.values()} == sep, "every sep route must reach separate_all"
    assert all(row["params"] == {"task": "sep"} for row in rows.values())


def test_analysis_tasks_when_installable_then_route_to_analyze_command():
    catalog = json.loads(CATALOG.read_text())
    routes = {(f["family"], r["tasks"][0]): r for f in catalog["families"] for r in f["routes"] if r["registry"] is None}
    assert routes[("firered_vad", "vad")]["route"] == "aud_analyze"
    assert routes[("mms_forced_aligner", "align")]["params"] == {"task": "align", "language": "eng"}
    assert routes[("sortformer_diar_v2", "diar")]["package"].endswith("_local"), "local-conversion packages are staged"
    assert routes[("sherpa_kws", "wake_word")]["route"] == "aud_analyze", "embedded keyword rows need no extra input"


def test_generation_tasks_when_routable_then_gen_or_midi_and_held_families_pending():
    catalog = json.loads(CATALOG.read_text())
    routes = {(f["family"], r["tasks"][0]): r for f in catalog["families"] for r in f["routes"] if r["registry"] is None}
    assert routes[("stable_audio", "music")]["params"] == {"task": "gen"}
    assert routes[("muscriptor", "midi")]["params"] == {"task": "midi"}
    assert routes[("mulacover", "music")]["route"] == "pending adapter", "MIDI-input families stay pending"
    yue = routes[("yue2", "music")]
    assert yue["package"] == "yue2_main_q8_0" and "style=" in yue["params"]["request_options"]
    assert routes[("lfm_d1_omni", "classify")]["params"]["request_options"].startswith("questions=")


def test_verification_when_shared_package_then_task_suffix_selects_route(tmp_path):
    ledger = {"source_revision": "r", "runs": {"aud_generate:pkg:sfx": {"result": "ok"}}}
    (tmp_path / "catalogs").mkdir()
    (tmp_path / "catalogs/audiocpp-verified.json").write_text(json.dumps(ledger))
    families = [{"routes": [{"route": "aud_generate", "package": "pkg", "tasks": [t], "status": "pending runtime"} for t in ("music", "sfx")]}]
    GEN["apply_verification"](families, "r", tmp_path)
    assert [r["status"] for r in families[0]["routes"]] == ["pending runtime", "verified"]


def test_apply_verification_when_bare_key_and_package_has_two_tasks_then_neither_verified(tmp_path):
    (tmp_path / "catalogs").mkdir()
    ledger = {"source_revision": "r1", "runs": {"aud_generate:pkg": {"check": "x"}}}
    (tmp_path / "catalogs" / "audiocpp-verified.json").write_text(json.dumps(ledger))
    routes = [{"route": "aud_generate", "package": "pkg", "tasks": [t], "status": "pending runtime"} for t in ("music", "midi")]
    GEN["apply_verification"]([{"routes": routes}], "r1", tmp_path)
    assert [r["status"] for r in routes] == ["pending runtime"] * 2, "a bare key must not verify a multi-task package"


def test_tts_overrides_when_registry_builds_args_then_options_use_equals_form(registries):
    """VeVo2's --extra starts with '--' and AuK needs companion packages; both must survive Fire parsing."""
    by_package = {m["package"]: m for m in registries["tts_all"] if m["name"].startswith("ac83-")}
    vevo2 = by_package["vevo2_q8_0"]["args_fn"]("Hi.", "ref.wav", "o.wav", "en")
    assert "--extra=--task-route zero_shot_tts" in vevo2 and vevo2[vevo2.index("--ref") + 1] == "ref.wav"
    assert "--ref-text" not in vevo2, "a reference transcript makes VeVo2 speak it too"
    auk = by_package["auk_flash_q8_0"]["args_fn"]("Hi.", "", "o.wav", "en")
    assert "--deps=auk_qwen_q8_0,auk_vae_f32" in auk
    assert any(a.startswith("--request_options=") and "duration_sec=" in a for a in auk)
    assert by_package["auk_flash_q8_0"]["langs"] == ["en"]


def test_tts_audiocpp_prep_when_fire_passes_deps_tuple_then_each_dep_installed(monkeypatch):
    spec = importlib.util.spec_from_file_location("tts_audiocpp_mod", TOOLS / "tts/tools/tts-audiocpp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = []
    monkeypatch.setattr(module, "_require_engine", lambda: None)
    monkeypatch.setattr(module, "_installed", lambda pkg: seen.append(pkg) or True)
    module.prep("auk_flash_q8_0", deps=("auk_qwen_q8_0", "auk_vae_f32"))
    assert seen == ["auk_qwen_q8_0", "auk_vae_f32", "auk_flash_q8_0"]


def test_music_generators_when_inputs_known_then_routed_with_their_options(catalog):
    fams = {f["family"]: f for f in catalog["families"]}
    gen = lambda fam: next(r for r in fams[fam]["routes"] if r["route"] == "aud_generate")
    ace = gen("ace_step")
    assert ace["package"] == "ace_step_turbo_q8_0" and ace["params"]["task_route"] == "text2music"
    assert ace["params"]["lyrics"] and ace["params"]["duration_seconds"] > 0
    assert "lyrics=" in gen("minimax_music3")["params"]["request_options"]
    kws = next(r for r in fams["sherpa_kws"]["routes"] if r["tasks"] == ["wake_word"])
    assert kws["route"] == "aud_analyze" and kws["status"] == "verified"


def _aud_module(monkeypatch, tmp_path, calls, writes=None):
    spec = importlib.util.spec_from_file_location("aud_audiocpp_mod", TOOLS / "fx/tools/aud-audiocpp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_require_engine", lambda: tmp_path / "audiocpp_cli")
    monkeypatch.setattr(module, "_resolve", lambda pkg: {"family": "fx"})
    module._real_installed = module._installed
    monkeypatch.setattr(module, "_installed", lambda pkg: True)
    monkeypatch.setattr(module, "_model_path", lambda pkg, fam="": tmp_path / "model.gguf")

    class Done:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if writes:
            Path(cmd[cmd.index("--out") + 1]).write_bytes(b"RIFF")
        return Done()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return module


def test_aud_audiocpp_run_when_lyrics_and_duration_then_cli_flags_passed(monkeypatch, tmp_path):
    calls = []
    module = _aud_module(monkeypatch, tmp_path, calls, writes=True)
    module.run(package="p", task="gen", text="folk", lyrics="la la", duration_seconds=20, output=str(tmp_path / "a.wav"))
    module.run(package="p", task="gen", text="folk", output=str(tmp_path / "b.wav"))
    with_flags, plain = calls
    assert with_flags[with_flags.index("--lyrics") + 1] == "la la"
    assert with_flags[with_flags.index("--duration-seconds") + 1] == "20"
    assert "--lyrics" not in plain and "--duration-seconds" not in plain


def test_aud_audiocpp_analyze_when_no_output_then_empty_only_for_wake_word(monkeypatch, tmp_path):
    module = _aud_module(monkeypatch, tmp_path, [])
    module.analyze(str(tmp_path / "in.wav"), "p", "wake_word", output=str(tmp_path / "kws.json"))
    assert json.loads((tmp_path / "kws.json").read_text()) == []
    with pytest.raises(SystemExit):
        module.analyze(str(tmp_path / "in.wav"), "p", "vad", output=str(tmp_path / "vad.json"))


def test_generation_options_when_set_then_match_run_keywords():
    import inspect
    spec = importlib.util.spec_from_file_location("aud_audiocpp_sig", TOOLS / "fx/tools/aud-audiocpp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = set(inspect.signature(module.run).parameters)
    for family, options in GEN["GENERATION_OPTIONS"].items():
        assert set(options) <= names, family


def test_aud_audiocpp_stage_when_converted_file_then_linked_with_marker(monkeypatch, tmp_path):
    module = _aud_module(monkeypatch, tmp_path, [])
    monkeypatch.setattr(module, "MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(module, "_installed", lambda pkg: False)
    monkeypatch.setattr(module, "_resolve", lambda pkg: {
        "family": "fx", "target_directory": "Fx-local", "strip_prefix": "Fx-local",
        "files": ["Fx-local/fx-f16.gguf"]})
    src = tmp_path / "converted.gguf"
    src.write_bytes(b"GGUF")
    module.stage("fx_local", str(src))
    dest = tmp_path / "models/Fx-local/fx-f16.gguf"
    assert dest.is_symlink() and dest.resolve() == src.resolve()
    marker = json.loads((tmp_path / "models/Fx-local/.audiocpp-package-fx_local.json").read_text())
    assert marker["files"]["Fx-local/fx-f16.gguf"]["size"] == 4
    assert module._real_installed("fx_local"), "a staged package satisfies the real install check"
    module.stage("fx_local", str(dest.resolve()), force=True)
    assert src.read_bytes() == b"GGUF", "re-staging the same file must not delete it"
    wrong = tmp_path / "converted.bin"
    wrong.write_bytes(b"x")
    with pytest.raises(SystemExit):
        module.stage("fx_local", str(wrong), force=True)
    with pytest.raises(SystemExit):
        module.stage("fx_local", str(tmp_path / "missing.gguf"))


def test_aud_audiocpp_run_when_yue2_then_vae_companion_prepped(monkeypatch, tmp_path):
    module = _aud_module(monkeypatch, tmp_path, [])
    prepped = []
    monkeypatch.setattr(module, "_installed", lambda pkg: False)
    monkeypatch.setattr(module, "prep", prepped.append)
    try:
        module.run("in.wav", package="yue2_main_q8_0", task="gen", output=str(tmp_path / "o.wav"))
    except SystemExit:
        pass
    assert prepped[:2] == ["yue2_main_q8_0", "yue2_vae_f16"]
