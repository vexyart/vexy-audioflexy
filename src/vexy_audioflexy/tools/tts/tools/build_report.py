#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/build_report.py
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tts_all import save_report_html

def main(issue_file: str, outdir: str):
    root = Path(__file__).parent.parent
    issue_file = Path(issue_file).expanduser()
    outdir = Path(outdir).expanduser()
    report_json = outdir / "report.json"
    
    if not report_json.exists():
        print("report.json not found")
        sys.exit(1)
        
    # Read subjective quality assessment
    lines = issue_file.read_text(encoding="utf-8").splitlines()
    quality_map = {}
    for line in lines:
        line = line.strip()
        # Looking for things like: en__celune-calm-mlx__bf16__mlx__auto.mp3 ++
        match = re.match(r'^([^ ]+\.mp3)\s+([+–-]+)(?:\s.*)?$', line)
        if match:
            filename = match.group(1)
            score = match.group(2)
            quality_map[filename] = score
            
    # Load report.json
    data = json.loads(report_json.read_text(encoding="utf-8"))
    
    # Apply quality assessment and mark ignored models
    from tts_all import IGNORED_ENGINES, MODELS
    
    # Map model name to its supported languages
    model_langs = {m["name"]: m["langs"] for m in MODELS}

    count = 0
    ignored_count = 0
    for result_key, result_dict in data.get("results", {}).items():
        model_name = result_dict.get("model")
        lang = result_dict.get("lang")
        
        # Mark as ignored if in IGNORED_ENGINES or if lang is no longer supported
        if model_name in IGNORED_ENGINES or (model_name in model_langs and lang not in model_langs[model_name]):
            result_dict["ignored"] = True
            ignored_count += 1
            
        out_path = result_dict.get("output_path", "")
        if out_path:
            filename = Path(out_path).name
            if filename in quality_map:
                result_dict["quality"] = quality_map[filename]
                count += 1
            
    print(f"Applied {count} quality assessments. Marked {ignored_count} entries as ignored.")
    
    # Save report.json
    report_json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    
    # Build report.html
    html_path = save_report_html(data, outdir)
    print(f"Report rebuilt: {html_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("issue_file")
    parser.add_argument("outdir")
    main(**vars(parser.parse_args()))
