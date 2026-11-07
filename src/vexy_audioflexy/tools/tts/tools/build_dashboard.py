#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/build_dashboard.py
"""Build ``out/index.html`` — the interactive sample browser — from ``out/report.json``.

The report JSON is the accumulated record across runs; this renders it as a single
self-contained page (no CDN, no build step) with per-clip audio players and live
filtering by language, inference stack, cloning mode and status.

Run directly to rebuild the dashboard without re-running any inference:

    ./tools/build_dashboard.py
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from urllib.parse import quote
from _paths import runtime_path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _model_index() -> dict[str, dict]:
    """Static metadata per model name, from the batch runner's own registry."""
    from tts_all import IGNORED_ENGINES, MODELS, model_mode, model_quant, model_stack

    index: dict[str, dict] = {}
    for model in MODELS:
        index[model["name"]] = {
            "repo": model.get("repo", ""),
            "stack": model_stack(model),
            "quant": model_quant(model),
            "mode": model_mode(model),
            "langs": model.get("langs", []),
            "ignored": model["name"] in IGNORED_ENGINES,
            "script": model.get("script", ""),
        }
    return index


def _transcripts(outdir: Path) -> dict[str, str]:
    """What each clip was heard to say, from tools/verify-samples.sh.

    A sample that plays is not a sample that speaks, and the difference is not
    visible from a waveform — so the dashboard shows the transcript beside the
    player rather than leaving it in a side file.
    """
    path = outdir / "transcripts.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _rows(report: dict, index: dict[str, dict], outdir: Path) -> list[dict]:
    heard = _transcripts(outdir)
    results = report.get("results", [])
    results = list(results.values()) if isinstance(results, dict) else results
    rows: list[dict] = []
    for entry in results:
        # A voice-design run is recorded as "model:desiN"; the registry knows "model".
        name = entry.get("model", "")
        base = name.split(":", 1)[0]
        meta = index.get(base, {})
        # Historical reports carry absolute paths from earlier checkouts; out/ is flat.
        path = entry.get("output_path") or ""
        name_only = Path(path).name if path else ""
        rows.append({
            "model": name,
            "lang": entry.get("lang", ""),
            "success": bool(entry.get("success")),
            "file": name_only,
            # Older successful runs can point at clips that are no longer on disk;
            # those get a card without a player rather than a silent 404.
            "have_file": bool(name_only) and (outdir / name_only).exists(),
            "heard": heard.get(name_only, ""),
            "rtf": float(entry.get("rtf") or 0.0),
            "gen_s": float(entry.get("duration") or 0.0),
            "audio_s": float(entry.get("audio_duration_s") or 0.0),
            "size_kb": float(entry.get("file_size_kb") or 0.0),
            "error": (entry.get("error") or "").strip(),
            "stack": entry.get("stack") or meta.get("stack", ""),
            "quant": entry.get("quant") or meta.get("quant", ""),
            "mode": entry.get("mode") or meta.get("mode", ""),
            "repo": meta.get("repo", ""),
            "ignored": meta.get("ignored", False),
        })
    rows.sort(key=lambda r: (not r["success"], r["model"], r["lang"]))
    return rows


CSS = """
:root{color-scheme:light dark;
 --bg:#f6f7f9;--panel:#fff;--ink:#12141a;--muted:#5b6270;--line:#e3e6ec;
 --ok:#1f8a4c;--bad:#c0392b;--accent:#2f6feb;--chip:#eef1f6}
@media (prefers-color-scheme:dark){:root{
 --bg:#0e1116;--panel:#161a22;--ink:#e6e9ef;--muted:#9aa3b2;--line:#252b36;
 --ok:#3fb950;--bad:#f85149;--accent:#589bff;--chip:#1e2430}}
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
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(330px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
.card.fail{border-color:color-mix(in srgb,var(--bad) 40%,var(--line))}
.name{font-weight:650;word-break:break-word}
.name a{color:inherit;text-decoration:none;border-bottom:1px dotted var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.chip{background:var(--chip);border-radius:999px;padding:2px 9px;font-size:12px;color:var(--muted)}
.chip.ok{color:var(--ok)}.chip.bad{color:var(--bad)}
.chip.lang{color:var(--accent)}
audio{width:100%;margin-top:6px}
.metrics{display:flex;gap:14px;color:var(--muted);font-size:12px;margin-top:8px}
.heard{margin-top:8px;padding:7px 9px;background:var(--chip);border-radius:8px;
 font-size:12.5px;line-height:1.45;color:var(--ink)}
.heard.bad{color:var(--bad)}
pre.err{white-space:pre-wrap;word-break:break-word;background:var(--chip);border-radius:8px;
 padding:8px;margin:8px 0 0;font-size:12px;color:var(--muted);max-height:9em;overflow:auto}
.empty{color:var(--muted);padding:40px 0;text-align:center}
footer{color:var(--muted);padding:0 24px 40px;font-size:12px}
"""

JS = r"""
const rows = ROWS;
const $ = (s) => document.querySelector(s);
const grid = $('#grid');

function fill(sel, values, label){
  sel.innerHTML = '<option value="">' + label + '</option>' +
    values.map(v => `<option>${v}</option>`).join('');
}
const uniq = (k) => [...new Set(rows.map(r => r[k]).filter(Boolean))].sort();
fill($('#lang'), uniq('lang'), 'All languages');
fill($('#stack'), uniq('stack'), 'All stacks');
fill($('#mode'), uniq('mode'), 'All modes');

function render(){
  const q = $('#q').value.trim().toLowerCase();
  const lang = $('#lang').value, stack = $('#stack').value, mode = $('#mode').value;
  const status = $('#status').value, sort = $('#sort').value;

  let list = rows.filter(r =>
    (!lang || r.lang === lang) && (!stack || r.stack === stack) &&
    (!mode || r.mode === mode) &&
    (status === '' ||
      (status === 'notspeech' ? /^\[/.test(r.heard || '') : (status === 'ok') === r.success)) &&
    (!q || (r.model + ' ' + r.repo + ' ' + r.quant + ' ' + (r.heard || ''))
             .toLowerCase().includes(q)));

  const by = {
    name: (a,b) => a.model.localeCompare(b.model) || a.lang.localeCompare(b.lang),
    rtf:  (a,b) => (a.rtf||1e9) - (b.rtf||1e9),
    audio:(a,b) => b.audio_s - a.audio_s,
    gen:  (a,b) => a.gen_s - b.gen_s,
  }[sort];
  list = [...list].sort(by);

  $('#count').textContent = list.length + ' of ' + rows.length + ' runs';
  if (!list.length){ grid.innerHTML = '<p class="empty">No runs match these filters.</p>'; return; }

  grid.innerHTML = list.map(r => {
    const repo = r.repo
      ? `<a href="https://huggingface.co/${r.repo}" target="_blank" rel="noreferrer">${r.repo}</a>`
      : '<span class="chip">no repo recorded</span>';
    const chips = [
      `<span class="chip lang">${r.lang}</span>`,
      r.quant && `<span class="chip">${r.quant}</span>`,
      r.stack && `<span class="chip">${r.stack}</span>`,
      r.mode && `<span class="chip">${r.mode}</span>`,
      r.success ? '<span class="chip ok">ok</span>' : '<span class="chip bad">failed</span>',
      r.ignored ? '<span class="chip">ignored</span>' : '',
    ].filter(Boolean).join('');
    const player = r.success && r.have_file
      ? `<audio controls preload="none" src="${r.file}"></audio>`
      : (r.success ? '<div class="chips"><span class="chip">clip no longer in out/</span></div>' : '');
    const heard = r.heard
      ? `<div class="heard ${/^\[/.test(r.heard) ? 'bad' : ''}">${
          r.heard.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}</div>`
      : '';
    const metrics = r.success
      ? `<div class="metrics"><span>RTF ${r.rtf ? r.rtf.toFixed(3) : '—'}</span>
         <span>gen ${r.gen_s.toFixed(1)}s</span><span>audio ${r.audio_s.toFixed(1)}s</span>
         <span>${r.size_kb ? Math.round(r.size_kb) + ' KB' : ''}</span></div>`
      : (r.error ? `<pre class="err">${r.error.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}</pre>` : '');
    return `<article class="card ${r.success ? '' : 'fail'}">
      <div class="name">${r.model}</div>
      <div class="sub" style="font-size:12px">${repo}</div>
      <div class="chips">${chips}</div>
      ${player}${heard}${metrics}
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
    outdir = Path(outdir) if outdir else runtime_path('out')
    report_path = outdir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"no report to render at {report_path}; run ./tts_all.py first")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _rows(report, _model_index(), outdir)
    summary = report.get("summary", {})
    models = len({r["model"].split(":", 1)[0] for r in rows})
    stacks = len({r["stack"] for r in rows if r["stack"]})
    ok = sum(1 for r in rows if r["success"])
    children = sorted(p for p in outdir.iterdir() if p.is_dir() and (p / "report.json").is_file() and (p / "index.html").is_file())
    run_links = " · ".join(f'<a href="{quote(p.name, safe="")}/index.html">{html.escape(p.name)}</a>' for p in children)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PVC — TTS model comparison</title>
<style>{CSS}</style></head><body>
<header>
  <h1>Personal Voice Clone — TTS model comparison</h1>
  <div class="sub">Every clip on this page was generated locally on Apple Silicon from the
  same two prompts, one English and one Polish. Generated {html.escape(str(report.get("generated", "")))}.</div>
  {f'<div class="sub">Separate batch dashboards: {run_links}</div>' if run_links else ''}
  <div class="stats">
    <div class="stat"><b>{models}</b><span>models</span></div>
    <div class="stat"><b>{len(rows)}</b><span>runs</span></div>
    <div class="stat"><b>{ok}</b><span>succeeded</span></div>
    <div class="stat"><b>{len(rows) - ok}</b><span>failed</span></div>
    <div class="stat"><b>{stacks}</b><span>inference stacks</span></div>
    <div class="stat"><b>{summary.get("batch_elapsed_s", 0):.0f}s</b><span>last batch</span></div>
  </div>
</header>
<div class="controls">
  <input type="search" id="q" placeholder="Search model, repo or quantization…">
  <select id="lang"></select><select id="stack"></select><select id="mode"></select>
  <select id="status"><option value="">All runs</option><option value="ok">Succeeded</option>
    <option value="fail">Failed</option><option value="notspeech">Not speech</option></select>
  <select id="sort"><option value="name">Sort by name</option>
    <option value="rtf">Fastest (RTF)</option>
    <option value="gen">Shortest generation</option>
    <option value="audio">Longest audio</option></select>
  <label class="sw" id="count"></label>
</div>
<main><div class="grid" id="grid"></div></main>
<footer>Filenames encode the run: <code>{{lang}}__{{model}}__{{quant}}__{{stack}}__{{mode}}.mp3</code>.
Each card shows what the clip was heard to say — run <code>./tools/verify-samples.sh</code>
to refresh that, and filter to <em>Not speech</em> to find clips that play but say nothing.
Rebuild this page from the recorded results with <code>./tools/build_dashboard.py</code>.</footer>
<script>const ROWS = {json.dumps(rows)};{JS}</script>
</body></html>
"""
    out_path = outdir / "index.html"
    out_path.write_text(page, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    print(f"Dashboard written: {build(args.outdir)}")
