"""Run 1 — one Whisper pass over the complete audio (language forced to English).

Usage: whisper_full.py <audio_path> <model_repo>
"""

import os
import sys
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import emit  # noqa: E402

# Whisper decodes the audio in 30s windows and reports progress once per window,
# so between windows we interpolate from elapsed time to keep the line moving.
WINDOW_SECONDS = 30.0
HEARTBEAT_SECONDS = 0.4


class ProgressBar:
    """Stand-in for tqdm: mlx_whisper's internal bar becomes an event stream."""

    def __init__(self, total=None, **kwargs):
        self.total = float(total or 1)
        self.frames = 0.0
        self.real_pct = 2.0
        self.window_pct = 100.0 * min(WINDOW_SECONDS * 100.0, self.total) / self.total
        self.window_time = 6.0          # refined once a real window completes
        self.marked_at = time.time()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    # -- tqdm surface ------------------------------------------------------

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def update(self, count):
        with self.lock:
            self.frames += count
            now = time.time()
            self.window_time = max(0.5, now - self.marked_at)
            self.marked_at = now
            self.real_pct = 100.0 * self.frames / self.total
            self._emit(self.real_pct)

    def close(self):
        self.stop.set()

    # -- internals ---------------------------------------------------------

    def _emit(self, pct):
        decoded = min(self.total, pct / 100.0 * self.total) / 100.0
        emit(
            event="progress",
            pct=min(99.0, pct),
            msg=f"decoded {decoded:.0f}s / {self.total / 100.0:.0f}s of audio",
        )

    def _heartbeat(self):
        while not self.stop.wait(HEARTBEAT_SECONDS):
            with self.lock:
                if self.real_pct >= 99.0:
                    continue
                fraction = min(0.92, (time.time() - self.marked_at) / self.window_time)
                self._emit(self.real_pct + self.window_pct * fraction)


def main():
    audio_path = sys.argv[1]
    model_repo = sys.argv[2]

    started = time.time()
    emit(event="start", msg=f"loading {model_repo.split('/')[-1]}")

    import mlx_whisper

    # `mlx_whisper.transcribe` is the function, so reach the module by name.
    transcribe_module = sys.modules["mlx_whisper.transcribe"]
    transcribe_module.tqdm = types.SimpleNamespace(tqdm=ProgressBar)

    emit(event="progress", pct=2.0, msg="model ready, decoding full audio")

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_repo,
        language="en",
        condition_on_previous_text=False,
    )

    # Segment timings let run 3 line each chunk up with the matching stretch of
    # this transcript, instead of guessing the alignment from wording alone.
    segments = [
        {
            "start": round(float(segment["start"]), 2),
            "end": round(float(segment["end"]), 2),
            "text": segment["text"].strip(),
        }
        for segment in result.get("segments", [])
    ]

    emit(
        event="text",
        text=result["text"].strip(),
        language=result.get("language", "unknown"),
        segments=segments,
    )
    emit(
        event="done",
        pct=100.0,
        elapsed=round(time.time() - started, 1),
        msg=f"detected language: {result.get('language', 'unknown')}",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # surfaced in the UI rather than swallowed
        emit(event="error", msg=f"{type(exc).__name__}: {exc}")
        sys.exit(1)
