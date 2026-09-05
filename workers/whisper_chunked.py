"""Run 2 — Whisper on independent fixed-length chunks with language auto-detect.

Usage: whisper_chunked.py <audio_path> <model_repo> <chunk_seconds>
"""

import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import emit  # noqa: E402

SAMPLE_RATE = 16000


def load_samples(audio_path):
    command = [
        "ffmpeg",
        "-i", audio_path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def main():
    audio_path = sys.argv[1]
    model_repo = sys.argv[2]
    chunk_seconds = float(sys.argv[3])

    started = time.time()
    emit(event="start", msg="decoding audio for chunking")

    samples = load_samples(audio_path)
    duration = len(samples) / SAMPLE_RATE

    chunk_size = int(chunk_seconds * SAMPLE_RATE)
    chunks = [
        samples[start:min(start + chunk_size, len(samples))]
        for start in range(0, len(samples), chunk_size)
    ]

    emit(
        event="progress",
        pct=2.0,
        msg=f"{len(chunks)} chunks of {chunk_seconds:g}s, loading model",
        total_chunks=len(chunks),
    )

    import mlx_whisper

    lines = []

    for i, chunk in enumerate(chunks):
        start_seconds = i * chunk_seconds
        end_seconds = min(start_seconds + chunk_seconds, duration)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name

        sf.write(temp_path, chunk, SAMPLE_RATE)

        try:
            result = mlx_whisper.transcribe(
                temp_path,
                path_or_hf_repo=model_repo,
                # Whisper must detect the language independently per chunk
                language=None,
                condition_on_previous_text=False,
            )

            language = result.get("language", "unknown")
            transcript = result.get("text", "").strip()

            lines.append(
                f"{start_seconds:6.1f} - {end_seconds:6.1f} | {language:5} | {transcript}"
            )

            emit(
                event="chunk",
                index=i,
                start=round(start_seconds, 1),
                end=round(end_seconds, 1),
                language=language,
                text=transcript,
            )
            emit(
                event="progress",
                pct=min(99.0, 100.0 * (i + 1) / len(chunks)),
                msg=f"chunk {i + 1} / {len(chunks)}  ({end_seconds:.0f}s / {duration:.0f}s)",
            )

        except Exception as exc:
            emit(event="chunk_error", index=i, msg=f"{type(exc).__name__}: {exc}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    emit(event="text", text="\n".join(lines))
    emit(
        event="done",
        pct=100.0,
        elapsed=round(time.time() - started, 1),
        msg=f"{len(chunks)} chunks transcribed",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit(event="error", msg=f"{type(exc).__name__}: {exc}")
        sys.exit(1)
