#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/build_dashboard.py
"""Build ``out/index.html`` — the reference-vs-hypothesis browser — from report.json.

Kept separate from ``asr_all.py`` so the dashboard can be regenerated from a recorded
run without re-doing inference. Every scored clip is shown with its ground truth and
the model's output side by side, word-diffed, so a human can see *which* words the
model got wrong rather than only how many.
"""

from __future__ import annotations

import html
import json
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

VERSION = "1.0.0"


def _norm_words(text: str) -> list[str]:
    """Same normalization policy as asr_all.normalize, tokenized on whitespace."""
    text = unicodedata.normalize("NFC", text).casefold()
    text = "".join(" " if unicodedata.category(c)[0] in {"P", "S"} else c for c in text)
    return text.split()


def _diff_html(reference: str, hypothesis: str) -> tuple[str, str]:
    """Word-level diff: deletions marked in the reference, insertions in the hypothesis."""
    ref, hyp = _norm_words(reference), _norm_words(hypothesis)
    matcher = SequenceMatcher(a=ref, b=hyp, autojunk=False)
    left: list[str] = []
    right: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        r = html.escape(" ".join(ref[i1:i2]))
        h = html.escape(" ".join(hyp[j1:j2]))
        if tag == "equal":
            left.append(r)
            right.append(h)
        else:
            if r:
                left.append(f"<mark class=del>{r}</mark>")
            if h:
                right.append(f"<mark class=ins>{h}</mark>")
    return " ".join(left), " ".join(right)


def _clip_rows(record: dict) -> str:
    rows: list[str] = []
    for clip in record.get("clips", []):
        name = html.escape(str(clip.get("clip", "")))
        if not clip.get("success"):
            rows.append(
                f"<tr class=fail><td>{name}</td><td colspan=4>"
                f"{html.escape(str(clip.get('error', 'failed'))[:200])}</td></tr>"
            )
            continue
        left, right = _diff_html(clip.get("reference", ""), clip.get("hypothesis", ""))
        if clip.get("degenerate"):
            right = (
                f"<em class=degen>no words — model emitted "
                f"{html.escape(repr(clip.get('hypothesis', '')))}</em>"
            )
        # A word error rate above 1.0 means more errors than there are reference
        # words, which only happens when the model inserts — it is generating
        # rather than mis-hearing. Hviske does this on Polish, emitting fluent
        # Danish. That is a different failure from a bad score and is worth saying.
        wer = clip.get("wer") or 0.0
        runaway = wer > 1.0
        row_class = " class=runaway" if runaway else ""
        if runaway:
            right = f"<em class=degen>generating, not transcribing</em> {right}"
        rows.append(
            f"<tr{row_class}><td class=clip>{name}</td>"
            f"<td class=wer>{clip.get('wer', 0) * 100:.1f}%</td>"
            f"<td class=wer>{clip.get('cer', 0) * 100:.1f}%</td>"
            f"<td class=txt><span class=lbl>ref</span> {left}</td>"
            f"<td class=txt><span class=lbl>hyp</span> {right}</td></tr>"
        )
    return "".join(rows)


def build(outdir: str | Path = "out") -> Path | None:
    """Render *outdir*/index.html from *outdir*/report.json. Returns the path."""
    out = Path(outdir)
    report = out / "report.json"
    if not report.exists():
        return None
    data = json.loads(report.read_text(encoding="utf-8"))
    results = data.get("results", {})

    sections: list[str] = []
    blocked: list[str] = []
    for key in sorted(results):
        rec = results[key]
        model = html.escape(str(rec.get("model", "")))
        lang = html.escape(str(rec.get("lang", "")))
        if rec.get("blocked"):
            blocked.append(
                f"<li><b>{model}</b> — {html.escape(str(rec.get('reason', '')))}</li>")
            continue
        if not rec.get("clips"):
            continue
        head = (
            f"{model} <span class=tag>{lang}</span> "
            f"<span class=tag>{html.escape(str(rec.get('stack', '')))}</span> "
            f"<span class=tag>{html.escape(str(rec.get('quant', '')))}</span> "
            f"WER <b>{rec.get('wer', 0) * 100:.1f}%</b> · "
            f"CER <b>{rec.get('cer', 0) * 100:.1f}%</b> · "
            f"RTF <b>{rec.get('rtf', 0):.3f}</b> · "
            f"{rec.get('clips_ok', 0)}/{rec.get('clips_total', 0)} clips"
        )
        sections.append(
            f"<details open><summary>{head}</summary><table>"
            f"<thead><tr><th>Clip<th>WER<th>CER<th>Reference<th>Hypothesis</tr></thead>"
            f"<tbody>{_clip_rows(rec)}</tbody></table></details>"
        )

    blocked_html = (
        f"<h2>Blocked engines</h2><ul class=blocked>{''.join(sorted(set(blocked)))}</ul>"
        if blocked else ""
    )
    page = f"""<!doctype html><meta charset=utf-8><title>ASR dashboard</title>
<style>
:root{{color-scheme:light dark}}
body{{font:14px/1.6 system-ui,sans-serif;margin:2rem auto;max-width:1200px;padding:0 1rem}}
h1{{margin-bottom:.2rem}} .sub{{color:#888;margin-top:0}}
details{{border:1px solid #8883;border-radius:8px;margin:1rem 0;padding:.5rem 1rem}}
summary{{font-size:1.05rem;cursor:pointer;padding:.3rem 0}}
.tag{{background:#8882;border-radius:4px;padding:.05rem .4rem;font-size:.8rem;margin:0 .1rem}}
table{{border-collapse:collapse;width:100%;margin-top:.5rem}}
td,th{{padding:.35rem .5rem;border-bottom:1px solid #8883;vertical-align:top;text-align:left}}
td.clip{{font-family:ui-monospace,monospace;font-size:.78rem;white-space:nowrap}}
td.wer{{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}}
td.txt{{width:36%}} .lbl{{color:#888;font-size:.72rem;text-transform:uppercase;margin-right:.3rem}}
mark.del{{background:#ffd9d9;color:#900}} mark.ins{{background:#d9f0ff;color:#046}}
tr.fail td{{color:#b00}} .degen{{color:#b60}}
tr.runaway td.wer{{color:#b3261e;font-weight:700}} ul.blocked li{{margin:.4rem 0;color:#777}}
@media(prefers-color-scheme:dark){{mark.del{{background:#5a1f1f;color:#ffb4b4}}
mark.ins{{background:#123a52;color:#a8dcff}}}}
</style>
<h1>ASR dashboard</h1>
<p class=sub>Generated {html.escape(str(data.get('generated', '')))} ·
{data.get('summary', {}).get('successes', 0)}/{data.get('summary', {}).get('total_runs', 0)}
runs succeeded · <a href="report.html">sortable summary table</a></p>
<p class=sub>Text is shown after scoring normalization (case- and punctuation-folded,
diacritics kept). <mark class=del>Red</mark> = in the reference but missed;
<mark class=ins>blue</mark> = produced but not in the reference.</p>
{''.join(sections) or '<p>No scored results yet.</p>'}
{blocked_html}
"""
    path = out / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


if __name__ == "__main__":
    import fire

    fire.Fire(build)
