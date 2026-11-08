# this_file: src/vexy_audioflexy/speech/inference.py
"""Optional Apple Silicon inference; no private machine paths or cloud services."""

from pathlib import Path

from . import qwen
from .hints import vocabulary
from .types import Turn, Word

ASR = qwen.MODEL
WHISPER = "mlx-community/whisper-large-v3-turbo"
DIAR = "mlx-community/Nemotron-3-Diarization-8bit"


def transcribe(
    wav: Path,
    hints: str,
    language: str,
    model: str = ASR,
    alignment_model: str = qwen.ALIGNER,
    reference: tuple = (),
) -> tuple[list[Word], dict]:
    """Primary recognition is Qwen, with actual forced-aligned word timings."""
    qwen.check_model(model)
    return qwen.transcribe(wav, hints, language, model, alignment_model, reference)


def whisper(
    wav: Path, hints: str, language: str, model: str = WHISPER
) -> tuple[list[Word], dict]:
    """Forward the actual hint file to Whisper's contextual initial prompt."""
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(wav),
        path_or_hf_repo=model,
        initial_prompt=(", ".join(vocabulary(hints)) or hints) or None,
        language=language or None,
        word_timestamps=True,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
    )
    words = [
        Word(w["word"].strip(), float(w["start"]), float(w["end"]))
        for segment in result["segments"]
        for w in segment.get("words", [])
        if w["word"].strip()
    ]
    if result["text"].strip() and not words:
        raise ValueError("ASR did not return word timestamps")
    return words, result


def diarize(
    wav: Path, model: str = DIAR, backend: str = "auto", max_speakers: int = 40
) -> tuple[list[Turn], dict]:
    """Retain a continuous arrival-order speaker cache across the entire recording."""
    if backend == "cluster":
        from diarize import diarize as cluster

        result = cluster(wav, max_speakers=max_speakers)
        turns = [Turn(float(s.start), float(s.end), s.speaker) for s in result.segments]
        return turns, {
            "model": "diarize/WeSpeaker-ResNet34-LM",
            "capacity": max_speakers,
            "backend": "cluster",
            "estimated_speakers": result.num_speakers,
            "turns": [
                {"start": t.start, "end": t.end, "speaker": t.speaker} for t in turns
            ],
        }
    if backend not in {"nemotron", "auto"}:
        raise ValueError("diar_backend must be auto, nemotron or cluster")
    if not qwen.mlx_available():
        if backend == "nemotron":
            raise RuntimeError(
                "MLX Nemotron unavailable; choose diar_backend audiocpp or cluster"
            )
        try:
            return diarize(wav, model, "cluster", max_speakers)
        except ImportError as error:
            raise RuntimeError(
                "MLX diarization unavailable; install vexy-audioflexy[cluster], choose native diarization, or supply turns_file"
            ) from error
    from mlx_audio.vad import load

    detector = load(model, strict=True)
    detector.set_streaming_config("offline")
    result = detector.generate(
        str(wav), threshold=0.5, min_duration=0.16, merge_gap=0.16
    )
    turns = [
        Turn(float(s.start), float(s.end), f"speaker_{s.speaker}")
        for s in result.segments
    ]
    if backend == "auto" and len({t.speaker for t in turns}) >= 8:
        try:
            return diarize(wav, model, "cluster", max_speakers)
        except ImportError as error:
            raise RuntimeError(
                "Eight speaker slots filled; install vexy-audioflexy[cluster] or supply verified turns_file"
            ) from error
    return turns, {
        "model": model,
        "turns": [
            {"start": t.start, "end": t.end, "speaker": t.speaker} for t in turns
        ],
        "capacity": 8,
        "anonymous_ids": True,
    }
