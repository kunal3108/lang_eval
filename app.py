"""Lang-Eval — local interface for the three-run ASR reconciliation pipeline.

Run 1: Whisper over the full audio (forced English).
Run 2: Whisper over independent 5s chunks (auto language detection).
Run 3: an LLM that reconciles the two into a final transcript.

Runs 1 and 2 execute in parallel as separate processes; run 3 starts once both
finish. Progress from all three is streamed to the browser over SSE.

Start with:  ./run.sh      (or: python3.13 app.py)
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
WORKER_DIR = os.path.join(BASE_DIR, "workers")
STATIC_DIR = os.path.join(BASE_DIR, "static")

WHISPER_MODEL = os.environ.get("LANG_EVAL_WHISPER", "mlx-community/whisper-medium-mlx")
LLM_MODEL = os.environ.get("LANG_EVAL_LLM", "mlx-community/Qwen2.5-7B-Instruct-4bit")
CHUNK_SECONDS = float(os.environ.get("LANG_EVAL_CHUNK_SECONDS", "5"))
MAX_TOKENS = int(os.environ.get("LANG_EVAL_MAX_TOKENS", "2000"))
LLM_MODES = ("anchored", "direct", "two_stage", "segment")

ALLOWED_SUFFIXES = {".mp3", ".wav", ".m4a", ".mp4", ".aac", ".flac", ".ogg", ".opus", ".webm"}
SENTINEL = "@@LE@@"
STREAM_LIMIT = 4 * 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Lang-Eval")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# job_id -> {"path": ..., "name": ..., "duration": ...}
JOBS = {}


def probe_duration(path):
    try:
        output = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return round(float(output), 3)
    except Exception:
        return None


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type '{suffix or 'unknown'}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    job_id = uuid.uuid4().hex[:12]
    stored_path = os.path.join(UPLOAD_DIR, job_id + suffix)

    with open(stored_path, "wb") as handle:
        shutil.copyfileobj(file.file, handle)

    duration = probe_duration(stored_path)
    if duration is None:
        os.remove(stored_path)
        raise HTTPException(status_code=400, detail="Could not read that file as audio.")

    JOBS[job_id] = {
        "path": stored_path,
        "name": file.filename,
        "duration": duration,
    }

    return {
        "job_id": job_id,
        "name": file.filename,
        "duration": duration,
        "size": os.path.getsize(stored_path),
        "url": f"/uploads/{os.path.basename(stored_path)}",
        "config": {
            "whisper_model": WHISPER_MODEL,
            "llm_model": LLM_MODEL,
            "chunk_seconds": CHUNK_SECONDS,
        },
    }


@app.post("/api/use-path")
async def use_path(payload: dict):
    """Register an audio file already on this machine, by absolute path."""
    raw_path = (payload or {}).get("path", "").strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail="No path given.")

    path = os.path.expanduser(raw_path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=400, detail=f"No such file: {path}")

    suffix = os.path.splitext(path)[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type '{suffix}'.")

    job_id = uuid.uuid4().hex[:12]
    stored_path = os.path.join(UPLOAD_DIR, job_id + suffix)
    shutil.copyfile(path, stored_path)

    duration = probe_duration(stored_path)
    if duration is None:
        os.remove(stored_path)
        raise HTTPException(status_code=400, detail="Could not read that file as audio.")

    JOBS[job_id] = {"path": stored_path, "name": os.path.basename(path), "duration": duration}

    return {
        "job_id": job_id,
        "name": os.path.basename(path),
        "duration": duration,
        "size": os.path.getsize(stored_path),
        "url": f"/uploads/{os.path.basename(stored_path)}",
        "config": {
            "whisper_model": WHISPER_MODEL,
            "llm_model": LLM_MODEL,
            "chunk_seconds": CHUNK_SECONDS,
        },
    }


async def pump_worker(stage, argv, queue, live_procs):
    """Run one worker process, forwarding its events onto the queue."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=STREAM_LIMIT,
        cwd=BASE_DIR,
    )
    live_procs.append(proc)

    collected = {"text": None, "segments": None, "failed": False}
    stderr_tail = []

    async def read_stdout():
        while True:
            try:
                raw = await proc.stdout.readline()
            except (asyncio.LimitOverrunError, ValueError):
                continue
            if not raw:
                break
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if not line.startswith(SENTINEL):
                continue
            try:
                payload = json.loads(line[len(SENTINEL):])
            except json.JSONDecodeError:
                continue
            if payload.get("event") == "text":
                collected["text"] = payload.get("text", "")
                if payload.get("segments") is not None:
                    collected["segments"] = payload["segments"]
            if payload.get("event") == "error":
                collected["failed"] = True
            payload["stage"] = stage
            await queue.put(payload)

    async def read_stderr():
        while True:
            raw = await proc.stderr.readline()
            if not raw:
                break
            text = raw.decode("utf-8", "replace").strip()
            if text:
                stderr_tail.append(text)
                del stderr_tail[:-12]

    await asyncio.gather(read_stdout(), read_stderr())
    code = await proc.wait()

    if code != 0 and not collected["failed"]:
        collected["failed"] = True
        await queue.put({
            "stage": stage,
            "event": "error",
            "msg": f"worker exited with code {code}: " + " | ".join(stderr_tail[-3:]),
        })

    return collected


async def run_pipeline(job_id, llm_mode):
    job = JOBS[job_id]
    audio_path = job["path"]
    queue = asyncio.Queue()
    live_procs = []

    async def orchestrate():
        full_argv = [
            sys.executable,
            os.path.join(WORKER_DIR, "whisper_full.py"),
            audio_path,
            WHISPER_MODEL,
        ]
        chunked_argv = [
            sys.executable,
            os.path.join(WORKER_DIR, "whisper_chunked.py"),
            audio_path,
            WHISPER_MODEL,
            str(CHUNK_SECONDS),
        ]

        # Runs 1 and 2 in parallel.
        full_result, chunked_result = await asyncio.gather(
            pump_worker("full", full_argv, queue, live_procs),
            pump_worker("chunked", chunked_argv, queue, live_procs),
        )

        if full_result["failed"] or chunked_result["failed"]:
            await queue.put({
                "stage": "llm",
                "event": "error",
                "msg": "skipped — an upstream transcription run failed",
            })
            return

        full_file = os.path.join(UPLOAD_DIR, f"{job_id}.full.txt")
        chunked_file = os.path.join(UPLOAD_DIR, f"{job_id}.chunked.txt")
        segments_file = os.path.join(UPLOAD_DIR, f"{job_id}.segments.json")

        with open(full_file, "w", encoding="utf-8") as handle:
            handle.write(full_result["text"] or "")
        with open(chunked_file, "w", encoding="utf-8") as handle:
            handle.write(chunked_result["text"] or "")
        with open(segments_file, "w", encoding="utf-8") as handle:
            json.dump(full_result["segments"] or [], handle, ensure_ascii=False)

        # Run 3.
        await pump_worker(
            "llm",
            [
                sys.executable,
                os.path.join(WORKER_DIR, "llm_judge.py"),
                full_file,
                chunked_file,
                LLM_MODEL,
                str(MAX_TOKENS),
                llm_mode,
                segments_file,
            ],
            queue,
            live_procs,
        )

    task = asyncio.create_task(orchestrate())

    try:
        yield sse({
            "event": "pipeline_start",
            "duration": job["duration"],
            "whisper_model": WHISPER_MODEL,
            "llm_model": LLM_MODEL,
            "chunk_seconds": CHUNK_SECONDS,
            "llm_mode": llm_mode,
            "max_tokens": MAX_TOKENS,
        })

        while True:
            getter = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {getter, task},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=15,
            )

            if getter in done:
                yield sse(getter.result())
                continue

            getter.cancel()

            if task in done:
                # Drain anything still queued, then finish.
                while not queue.empty():
                    yield sse(queue.get_nowait())
                task.result()  # re-raise orchestration errors, if any
                yield sse({"event": "pipeline_done"})
                return

            yield ": keepalive\n\n"

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        yield sse({"event": "pipeline_error", "msg": f"{type(exc).__name__}: {exc}"})
    finally:
        task.cancel()
        for proc in live_procs:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass


def sse(payload):
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


@app.get("/api/run/{job_id}")
async def run(job_id: str, llm_mode: str = "anchored"):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Unknown job — upload the audio again.")

    if llm_mode not in LLM_MODES:
        raise HTTPException(status_code=400, detail=f"llm_mode must be one of {LLM_MODES}")

    return StreamingResponse(
        run_pipeline(job_id, llm_mode),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("LANG_EVAL_PORT", "8765"))
    print(f"\n  Lang-Eval  →  http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
