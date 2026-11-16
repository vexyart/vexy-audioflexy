#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", "fire", "rich", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/vid/tools/vid-audiocpp.py
"""Video output from audio.cpp: talking-head and lip-sync families.

Kept apart from the audio tools (tts/asr/fx) because vexy-audioflexy is an
audio package; this is the one place that writes video. Families:

| family          | inputs                         | output |
|-----------------|--------------------------------|--------|
| soulx_flashhead | speech WAV + portrait image    | MP4    |
| latentsync      | new speech WAV + face video    | MP4    |

Engine, package install and model paths are shared with fx/tools/aud-audiocpp.py.
Requires ffmpeg on PATH.
"""

from __future__ import annotations

import json
import runpy
import shutil
import subprocess
from pathlib import Path

_AUD = runpy.run_path(str(Path(__file__).resolve().parents[2] / "fx" / "tools" / "aud-audiocpp.py"),
                      run_name="aud_audiocpp")

# Families whose engine output is RGB24 video in --out-dir; run() encodes it to MP4.
# Either chunked artifacts (`<prefix><chunk>.json` with meta + payload_hex) or one
# raw file described by `<prefix>.json` ({format, width, height, frames, fps, file}).
VIDEO_ARTIFACT_FAMILIES = {"soulx_flashhead": "soulx_flashhead_video_", "latentsync": "latentsync_video"}


def _encode_video(frames_dir: Path, prefix: str, audio: str, out_path: Path) -> None:
    """Join the engine's RGB24 chunk artifacts into an H.264 MP4, muxing `audio` when given.

    Frames go through a temp file rather than a pipe: with -shortest ffmpeg stops
    reading once the audio ends, which breaks a writer still feeding the pipe.
    """
    single = frames_dir / f"{prefix}.json"
    if single.is_file():
        meta = json.loads(single.read_text())
        raw = frames_dir / meta["file"]
        size = int(meta["width"]) * int(meta["height"]) * int(meta["frames"]) * 3
        if meta["format"] != "rgb24" or not raw.is_file() or raw.stat().st_size != size:
            raise SystemExit(f"{raw.name} does not match the frames described in {single.name}")
        _ffmpeg_rgb24(raw, meta, audio, out_path)
        return
    chunks = sorted(frames_dir.glob(f"{prefix}*.json"), key=lambda f: int(f.stem.rsplit("_", 1)[1]))
    if not chunks:
        raise SystemExit(f"no {prefix}*.json video artifacts in {frames_dir}")
    raw = frames_dir / "frames.rgb24"
    first = None
    with raw.open("wb") as sink:
        for chunk in chunks:
            art = json.loads(chunk.read_text())
            meta = art["meta"]
            first = first or meta
            if meta["format"] != "rgb24" or any(meta[k] != first[k] for k in ("width", "height", "fps")):
                raise SystemExit(f"video chunk {chunk.name} differs in format from the first chunk")
            payload = bytes.fromhex(art["payload_hex"])
            if len(payload) != int(meta["width"]) * int(meta["height"]) * int(meta["frames"]) * 3:
                raise SystemExit(f"video chunk {chunk.name} byte count does not match its dimensions")
            sink.write(payload)
    _ffmpeg_rgb24(raw, first, audio, out_path)
    raw.unlink()


def _ffmpeg_rgb24(raw: Path, meta: dict, audio: str, out_path: Path) -> None:
    """Encode raw RGB24 frames to H.264 MP4, muxing `audio` (cut to the shorter stream)."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
           "-video_size", f"{meta['width']}x{meta['height']}", "-framerate", str(meta["fps"]), "-i", str(raw)]
    if audio:
        cmd += ["-i", str(Path(audio).resolve()), "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True)


def run(
    audio: str,
    package: str = "latentsync_1_6_q8_0_f16",
    output: str = "output.mp4",
    image: str = "",
    video: str = "",
    request_options: str = "",
    session_options: str = "",
    seed: int = 0,
    play: bool = False,
    force: bool = False,
) -> None:
    """Render an MP4 from speech with one audio.cpp video package.

    Args:
        audio: Speech WAV (16 kHz mono works for both families); muxed into the MP4.
        package: audio.cpp package id (`soulx_flashhead_lite_q8_0`, `latentsync_1_6_q8_0_f16`).
        output: Output .mp4 path.
        image: Reference portrait for SoulX FlashHead (`reference_image_file`).
        video: Source face video for LatentSync (`reference_video_file`).
        request_options: Per-request `key=value` settings separated by `;`.
        session_options: Per-session `key=value` settings separated by `;`.
        seed: Fixed seed; 0 leaves the model default.
        play: Open the result instead of reporting the path.
        force: Overwrite an existing output file.
    """
    out_path = Path(output).resolve()
    if out_path.suffix.lower() != ".mp4":
        raise SystemExit("video families write MP4: give --output a .mp4 path")
    if not force and out_path.is_file() and out_path.stat().st_size > 0:
        print(f"Skipped: {output} already exists (use --force to overwrite)")
        return
    binary = _AUD["_require_engine"]()
    fam = _AUD["_resolve"](package)["family"]
    prefix = VIDEO_ARTIFACT_FAMILIES.get(fam)
    if not prefix:
        raise SystemExit(f"{package} ({fam}) is not a video family; use fx/tools/aud-audiocpp.py")
    if not _AUD["_installed"](package):
        _AUD["prep"](package)
    frames_dir = out_path.with_name(out_path.stem + ".frames")
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    cmd = [str(binary), "--task", "gen", "--family", fam, "--model", str(_AUD["_model_path"](package, fam)),
           "--backend", _AUD["BACKEND"], "--out-dir", str(frames_dir), "--metrics",
           "--audio", str(Path(audio).resolve())]
    if image:
        cmd += ["--request-option", f"reference_image_file={Path(image).resolve()}"]
    if video:
        cmd += ["--request-option", f"reference_video_file={Path(video).resolve()}"]
    for flag, blob in (("--request-option", request_options), ("--session-option", session_options)):
        cmd += [x for part in blob.split(";") if part.strip() for x in (flag, part.strip())]
    if seed:
        cmd += ["--seed", str(seed)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_AUD["ENGINE_ROOT"]), check=False)
    if result.returncode != 0 or not list(frames_dir.glob(f"{prefix}*.json")):
        lines = [ln for ln in (result.stdout + "\n" + result.stderr).splitlines()
                 if ln.strip() and not ln.startswith(("ggml_metal", "ggml_backend", "load_backend"))]
        raise SystemExit("audiocpp_cli failed\n" + "\n".join(lines[-12:]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _encode_video(frames_dir, prefix, audio, out_path)
    shutil.rmtree(frames_dir, ignore_errors=True)
    if play:
        subprocess.run(["open", str(out_path)], check=True)
    else:
        print(f"Saved {out_path}")


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
