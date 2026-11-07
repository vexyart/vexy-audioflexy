#!/usr/bin/env -S uv run
# /// script
# dependencies = ["fire", "rich", "jiwer", "jieba==0.42.1", "sacrebleu", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/translate_all.py
"""Index-Echo speech-translation benchmarks with distinct source and target scores."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

from asr_all import aggregate, score

ROOT = Path(__file__).resolve().parent


def main(manifest: str, outdir: str, targets: str = "en,es,ja", force: bool = False) -> None:
    """Run pinned fixtures sequentially; score target only when a reference exists."""
    source = Path(manifest).resolve()
    fixture = json.loads(source.read_text())
    output = Path(outdir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"manifest": str(source), "manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dataset_revision": fixture["revision"], "targets": {}}
    selected = targets.split(",")
    if any(target not in ["en", "es", "ja"] for target in selected):
        raise SystemExit("Target languages must be en/es/ja")
    for target in selected:
        records = []
        for row in fixture["fixtures"]:
            audio = source.parent / row["audio"]
            if hashlib.sha256(audio.read_bytes()).hexdigest() != row["wav_sha256"]:
                raise SystemExit("Fixture bytes differ from pinned manifest")
            text = output / f"{row['clip']}__{target}.txt"
            native = text.with_suffix(".native.json")
            if force or not native.exists():
                command = ["uv", "run", "--script", str(ROOT / "tools/asr-indexecho.py"), "transcribe",
                    "--audio", str(audio), "--text-out", str(text), "--target", target]
                if text.exists():
                    command += ["--force"]
                completed = subprocess.run(command, capture_output=True, text=True, timeout=1800, check=False)
                text.with_suffix(".wrapper.log").write_text(completed.stdout + completed.stderr)
                completed.check_returncode()
            receipt = json.loads(native.read_text())
            if (receipt["audio_sha256"] != row["wav_sha256"] or receipt["target"] != target
                or receipt["weights_revision"] != "08c2fc8e3fc87da1d848c5c75aeba1b761d2624e"
                or receipt["runtime_commit"] != "2892ed3e94b6ccbdacc111fa2896aaae0a5dc8fc"):
                raise SystemExit("Cached receipt belongs to another input or target")
            record = {"clip": row["clip"], "receipt": str(native), "success": True,
                "audio_duration_s": receipt["audio_duration_s"], "wall_s": receipt["wall_s"],
                "source": receipt["text"], "translation": receipt["translation"],
                **score(row["reference"], receipt["text"], lang="zh")}
            if target == "en":
                record["target_reference"] = row["english_reference"]
            records.append(record)
        result = {"clips": records, "source_metrics": aggregate(records)}
        if target == "en":
            hypotheses = [r["translation"] for r in records]
            references = [[r["target_reference"] for r in records]]
            metrics = {"bleu": BLEU(), "chrfpp": CHRF(word_order=2)}
            result["translation_metrics"] = {name: {"score": metric.corpus_score(hypotheses, references).score,
                "signature": str(metric.get_signature())} for name, metric in metrics.items()}
        else:
            result["translation_metrics"] = {"status": "unscored-no-parallel-target-reference"}
        report["targets"][target] = result
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(target, result["source_metrics"], result["translation_metrics"], flush=True)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
