#!/usr/bin/env -S uv run -s
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "numpy", "soundfile", "platformdirs>=4"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/build_dashboard.py
"""Build ``out/index.html`` — the interactive sample browser — from ``out/report.json``.

The report JSON is the accumulated record across runs; this renders it as a single
self-contained page (no CDN, no build step). Every card carries the players a
listener needs to judge the operation: the SOURCE audio that went in, the TARGET
voice reference where one was used, and the RESULT that came out, so a voice
conversion can be heard as a before/after rather than a single mystery clip.

Run directly to rebuild the dashboard without re-running any inference:

    ./tools/build_dashboard.py
"""

from __future__ import annotations


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _model_index() -> dict[str, dict]:
    """Static metadata per model name, from the batch runner's own registry."""
    from aud_all import IGNORED_ENGINES, MODELS, model_op, model_quant, model_stack

    return {
        m["name"]: {
            "repo": m.get("repo", ""),
            "stack": model_stack(m),
            "quant": model_quant(m),
            "op": model_op(m),
            "family": m.get("family", ""),
            "route": m.get("route", ""),
            "ignored": m["name"] in IGNORED_ENGINES,
            "script": m.get("script", ""),
        }
        for m in MODELS
    }


def _stage_input(path: str, outdir: Path) -> str:
    """Copy an input fixture into ``out/inputs/`` so the page can play it.

    The page must stay openable straight from ``out/``; linking to absolute paths
    outside it breaks under file:// in some browsers, so inputs are staged once.
    """
    if not path:
        return ""
    src = Path(path)
    if not src.exists():
        return ""
    dest_dir = outdir / "inputs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if not dest.exists() or dest.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dest)
    return f"inputs/{dest.name}"


def _num(value: object) -> float | None:
    """A metric as a float, or None when it was never measured."""
    return float(value) if isinstance(value, (int, float)) else None


def _rows(report: dict, index: dict[str, dict], outdir: Path) -> list[dict]:
    results = report.get("results", {})
    results = list(results.values()) if isinstance(results, dict) else results
    rows: list[dict] = []
    for entry in results:
        name = entry.get("model", "")
        meta = index.get(name, {})
        out_name = Path(entry.get("output_path") or "").name
        rows.append({
            "model": name,
            "op": entry.get("op") or meta.get("op", ""),
            "success": bool(entry.get("success")),
            "file": out_name,
            # A recorded run can point at a clip that is no longer on disk; that
            # gets a card without a player rather than a silent 404.
            "have_file": bool(out_name) and (outdir / out_name).exists(),
            "source": _stage_input(entry.get("source_path", ""), outdir),
            "target": _stage_input(entry.get("target_path", ""), outdir),
            "rms": float(entry.get("rms") or 0.0),
            "rtf": float(entry.get("rtf") or 0.0),
            "gen_s": float(entry.get("duration") or 0.0),
            "audio_s": float(entry.get("audio_duration_s") or 0.0),
            "src_s": float(entry.get("source_duration_s") or 0.0),
            "rate": int(float(entry.get("sample_rate") or 0)),
            "src_rate": int(float(entry.get("source_sample_rate") or 0)),
            "size_kb": float(entry.get("file_size_kb") or 0.0),
            "error": (entry.get("error") or "").strip(),
            "stack": entry.get("stack") or meta.get("stack", ""),
            "quant": entry.get("quant") or meta.get("quant", ""),
            "family": entry.get("family") or meta.get("family", ""),
            "route": entry.get("route") or meta.get("route", ""),
            "repo": meta.get("repo", ""),
            # Objective scores; a listening dashboard that only says "not
            # silence" cannot tell a working denoiser from a passthrough.
            "mos": _num((entry.get("metrics") or {}).get("dnsmos.ovrl")),
            "src_mos": _num((entry.get("source_metrics") or {}).get("dnsmos.ovrl")),
            "utmos": _num((entry.get("metrics") or {}).get("utmos")),
            "hf8k": _num((entry.get("metrics") or {}).get("hf8k_ratio")),
            "stoi": _num((entry.get("metrics") or {}).get("stoi")),
            "corr": _num((entry.get("metrics") or {}).get("input_corr")),
            "ignored": bool(entry.get("ignored")) or meta.get("ignored", False),
        })
    rows.sort(key=lambda r: (not r["success"], r["op"], r["model"]))
    return rows


CSS = """
:root{color-scheme:light dark;
 --bg:#f6f7f9;--panel:#fff;--ink:#12141a;--muted:#5b6270;--line:#e3e6ec;
 --ok:#1f8a4c;--bad:#c0392b;--warn:#b8860b;--accent:#2f6feb;--chip:#eef1f6}
@media (prefers-color-scheme:dark){:root{
 --bg:#0e1116;--panel:#161a22;--ink:#e6e9ef;--muted:#9aa3b2;--line:#252b36;
 --ok:#3fb950;--bad:#f85149;--warn:#d29922;--accent:#589bff;--chip:#1e2430}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:28px 24px 18px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{margin:0 0 4px;font-size:22px;letter-spacing:-.01em}
.sub{color:var(--muted)}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px}
.stat{background:var(--chip);border-radius:8px;padding:8px 12px;min-width:92px}
.stat b{display:block;font-size:18px}
.stat span{color:var(--muted);font-size:12px}
.controls{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:10px;
 align-items:center;padding:12px 24px;background:var(--panel);border-bottom:1px solid var(--line)}
input[type=search],select{background:var(--bg);color:var(--ink);border:1px solid var(--line);
 border-radius:8px;padding:7px 10px;font:inherit}
input[type=search]{flex:1;min-width:200px}
label.sw{display:flex;align-items:center;gap:6px;color:var(--muted)}
main{padding:18px 24px 60px}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(360px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
.card.fail{border-color:color-mix(in srgb,var(--bad) 40%,var(--line))}
.card.silent{border-color:color-mix(in srgb,var(--warn) 45%,var(--line))}
.name{font-weight:650;word-break:break-word}
.name a{color:inherit;text-decoration:none;border-bottom:1px dotted var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.chip{background:var(--chip);border-radius:999px;padding:2px 9px;font-size:12px;color:var(--muted)}
.chip.ok{color:var(--ok)}.chip.bad{color:var(--bad)}.chip.warn{color:var(--warn)}
.chip.op{color:var(--accent)}
.track{margin-top:10px}
.track b{display:block;font-size:11px;letter-spacing:.06em;text-transform:uppercase;
 color:var(--muted);margin-bottom:2px}
.track.result b{color:var(--accent)}
audio{width:100%}
.metrics{display:flex;flex-wrap:wrap;gap:12px;color:var(--muted);font-size:12px;margin-top:10px}
pre.err{white-space:pre-wrap;word-break:break-word;background:var(--chip);border-radius:8px;
 padding:8px;margin:8px 0 0;font-size:12px;color:var(--muted);max-height:9em;overflow:auto}
.empty{color:var(--muted);padding:40px 0;text-align:center}
footer{color:var(--muted);padding:0 24px 40px;font-size:12px}
"""

JS = """
const rows = ROWS;
const $ = (s) => document.querySelector(s);
const grid = $('#grid');

function fill(sel, values, label){
  sel.innerHTML = '<option value="">' + label + '</option>' +
    values.map(v => `<option>${v}</option>`).join('');
}
const uniq = (k) => [...new Set(rows.map(r => r[k]).filter(Boolean))].sort();
fill($('#op'), uniq('op'), 'All operations');
fill($('#family'), uniq('family'), 'All families');
fill($('#quant'), uniq('quant'), 'All quantizations');

const esc = (s) => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function player(label, src, cls){
  if (!src) return '';
  return `<div class="track ${cls||''}"><b>${label}</b>
    <audio controls preload="none" src="${src}"></audio></div>`;
}

function render(){
  const q = $('#q').value.trim().toLowerCase();
  const op = $('#op').value, family = $('#family').value, quant = $('#quant').value;
  const status = $('#status').value, sort = $('#sort').value;

  let list = rows.filter(r =>
    (!op || r.op === op) && (!family || r.family === family) &&
    (!quant || r.quant === quant) &&
    (status === '' ||
      (status === 'ok' && r.success && r.rms > 0) ||
      (status === 'silent' && r.success && !r.rms) ||
      (status === 'fail' && !r.success)) &&
    (!q || (r.model + ' ' + r.repo + ' ' + r.route + ' ' + r.quant).toLowerCase().includes(q)));

  const by = {
    name: (a,b) => a.model.localeCompare(b.model),
    op:   (a,b) => a.op.localeCompare(b.op) || a.model.localeCompare(b.model),
    rtf:  (a,b) => (a.rtf||1e9) - (b.rtf||1e9),
    gen:  (a,b) => a.gen_s - b.gen_s,
  }[sort];
  list = [...list].sort(by);

  $('#count').textContent = list.length + ' of ' + rows.length + ' runs';
  if (!list.length){ grid.innerHTML = '<p class="empty">No runs match these filters.</p>'; return; }

  grid.innerHTML = list.map(r => {
    const silent = r.success && !r.rms;
    const repo = r.repo
      ? `<a href="https://huggingface.co/${r.repo}" target="_blank" rel="noreferrer">${r.repo}</a>`
      : '<span class="chip">no repo recorded</span>';
    const chips = [
      `<span class="chip op">${r.op}</span>`,
      r.family && `<span class="chip">${r.family}</span>`,
      r.route && `<span class="chip">${r.route}</span>`,
      r.quant && `<span class="chip">${r.quant}</span>`,
      r.stack && `<span class="chip">${r.stack}</span>`,
      r.success && !silent ? '<span class="chip ok">ok</span>' : '',
      silent ? '<span class="chip warn">silent output</span>' : '',
      !r.success ? '<span class="chip bad">failed</span>' : '',
      r.ignored ? '<span class="chip">ignored</span>' : '',
    ].filter(Boolean).join('');

    const players =
      player(`source in${r.src_rate ? ' · ' + r.src_rate + ' Hz' : ''}`, r.source) +
      player(r.op === 'aec' ? 'far-end loudspeaker reference' : 'target voice', r.target) +
      (r.success && r.have_file
        ? player(`result out${r.rate ? ' · ' + r.rate + ' Hz' : ''}`, r.file, 'result')
        : (r.success ? '<div class="chips"><span class="chip">clip no longer in out/</span></div>' : ''));

    const metrics = r.success
      ? `<div class="metrics"><span>gen ${r.gen_s.toFixed(1)}s</span>
         <span>out ${r.audio_s.toFixed(1)}s</span><span>in ${r.src_s.toFixed(1)}s</span>
         <span>RMS ${r.rms.toFixed(4)}</span>
         ${r.mos != null ? `<span>DNSMOS ${r.mos.toFixed(2)}${r.src_mos != null ? ' (' + (r.mos - r.src_mos >= 0 ? '+' : '') + (r.mos - r.src_mos).toFixed(2) + ' vs in)' : ''}</span>` : ''}
         ${r.utmos != null ? `<span>UTMOS ${r.utmos.toFixed(2)}</span>` : ''}
         ${r.stoi != null ? `<span>STOI ${r.stoi.toFixed(3)}</span>` : ''}
         ${r.hf8k != null ? `<span>&gt;8 kHz ${(r.hf8k * 100).toFixed(1)}%</span>` : ''}
         ${r.corr != null ? `<span>corr(in) ${r.corr.toFixed(3)}</span>` : ''}
         <span>${r.size_kb ? Math.round(r.size_kb) + ' KB' : ''}</span></div>`
      : (r.error ? `<pre class="err">${esc(r.error)}</pre>` : '');

    return `<article class="card ${r.success ? (silent ? 'silent' : '') : 'fail'}">
      <div class="name">${r.model}</div>
      <div class="sub" style="font-size:12px">${repo}</div>
      <div class="chips">${chips}</div>
      ${players}${metrics}
    </article>`;
  }).join('');
}
document.querySelectorAll('.controls input,.controls select').forEach(el => {
  el.addEventListener('input', render);
});
render();
"""


def build(outdir: Path | None = None) -> Path:
    """Render out/index.html from out/report.json. Returns the written path."""
    outdir = Path(outdir) if outdir else Path.cwd() / "out"
    report_path = outdir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"no report to render at {report_path}; run ./aud_all.py first")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _rows(report, _model_index(), outdir)
    summary = report.get("summary", {})
    ops = len({r["op"] for r in rows if r["op"]})
    families = len({r["family"] for r in rows if r["family"]})
    ok = sum(1 for r in rows if r["success"] and r["rms"] > 0)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fx — audio enhancement and voice conversion</title>
<style>{CSS}</style></head><body>
<header>
  <h1>fx — audio enhancement &amp; voice conversion</h1>
  <div class="sub">Every clip on this page was processed locally on Apple Silicon by
  the backend recorded on each card. Each card plays the source that went in, the
  target voice or far-end loudspeaker reference where one was used, and the result that came out.
  Generated {html.escape(str(report.get("generated", "")))}.</div>
  <div class="stats">
    <div class="stat"><b>{len(rows)}</b><span>runs</span></div>
    <div class="stat"><b>{ok}</b><span>non-silent</span></div>
    <div class="stat"><b>{len(rows) - ok}</b><span>failed or silent</span></div>
    <div class="stat"><b>{families}</b><span>model families</span></div>
    <div class="stat"><b>{ops}</b><span>operations</span></div>
    <div class="stat"><b>{summary.get("batch_elapsed_s", 0):.0f}s</b><span>last batch</span></div>
  </div>
</header>
<div class="controls">
  <input type="search" id="q" placeholder="Search model, route or quantization…">
  <select id="op"></select><select id="family"></select><select id="quant"></select>
  <select id="status"><option value="">All runs</option><option value="ok">Succeeded</option>
    <option value="silent">Silent output</option><option value="fail">Failed</option></select>
  <select id="sort"><option value="op">Sort by operation</option>
    <option value="name">Sort by name</option>
    <option value="rtf">Fastest (RTF)</option>
    <option value="gen">Shortest processing</option></select>
  <label class="sw" id="count"></label>
</div>
<main><div class="grid" id="grid"></div></main>
<footer>Filenames encode the run:
<code>{{op}}__{{model}}__{{quant}}__{{stack}}__{{source}}.wav</code>.
Rebuild this page from the recorded results with <code>./tools/build_dashboard.py</code>.</footer>
<script>const ROWS = {json.dumps(rows)};{JS}</script>
</body></html>
"""
    out_path = outdir / "index.html"
    out_path.write_text(page, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    print(f"Dashboard written: {build()}")
