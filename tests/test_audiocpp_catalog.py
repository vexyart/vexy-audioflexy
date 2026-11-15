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
            assert route["route"] in {"tts_all", "asr_all", "aud_all", "covered", "pending adapter"}
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
    assert route("fx_music")["route"] == "pending adapter"
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
    routed = {f"{r['route']}:{r.get('package')}": r for f in catalog["families"] for r in f["routes"]}
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
