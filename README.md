# Lang-Eval

Local interface for the three-run multilingual ASR reconciliation pipeline from
`judge_first.ipynb`. Upload a call recording, play it back, and watch all three
runs report progress live.

```bash
./run.sh
```

Then open **http://127.0.0.1:8765**.

## What it does

| Run | Worker | What it produces |
| --- | --- | --- |
| 1 | `workers/whisper_full.py` | one Whisper pass over the whole file, `language="en"` — good continuity, may translate Hindi into English |
| 2 | `workers/whisper_chunked.py` | Whisper on independent 5s chunks, `language=None` — keeps Hindi/English switches, occasional hallucinations |
| 3 | `workers/llm_judge.py` | an LLM judges each chunk, and the final transcript is assembled from those judgements |

Runs 1 and 2 are separate processes started at the same time, each with its own
progress line and timer. Run 3 starts when both have finished. Everything is
streamed to the browser over SSE, so the three boxes fill as the output arrives
(chunk rows for run 2, token by token for run 3).

## Run 3 prompt modes

Selectable under the player:

- **anchored** (default) — the LLM answers KEEP or FIX for each chunk and writes no
  transcript text at all. The transcript is then assembled in code: KEEP copies the chunk
  verbatim in its own script, FIX substitutes the stretch of run 1 covering the same
  seconds (aligned through run 1's segment timings, the notebook's
  `get_full_text_for_chunk`). Asking the model to write the transcript made it translate
  intelligible Hindi back into the full pass's English, or blend the two readings
  together; judging is the part that needs a model, copying is not. Box 3 shows the
  per-chunk decisions, with a toggle for the joined transcript.
- **direct** — the notebook's prompt: reconcile both transcripts, return only the
  corrected transcript.
- **two-stage** — flag every chunk `CLEAN` / `GIBBERISH`, then `FINAL_TRANSCRIPT`.
  The flag list eats a lot of tokens; on long audio Qwen 7B often hits the token
  cap before it reaches the transcript. The UI says so when that happens.
- **segment** — segment-level reconciliation, transcript only.

## Loading audio

- **Upload audio** / drag and drop, or
- paste an absolute path (the file is copied into `uploads/`).

Accepted: mp3, wav, m4a, mp4, aac, flac, ogg, opus, webm.

## Configuration

Environment variables, all optional:

| Variable | Default |
| --- | --- |
| `LANG_EVAL_PYTHON` | `/opt/homebrew/bin/python3.13` — the interpreter that has `mlx_whisper` and `mlx_lm` |
| `LANG_EVAL_WHISPER` | `mlx-community/whisper-medium-mlx` |
| `LANG_EVAL_LLM` | `mlx-community/Qwen2.5-7B-Instruct-4bit` |
| `LANG_EVAL_CHUNK_SECONDS` | `5` |
| `LANG_EVAL_MAX_TOKENS` | `2000` |
| `LANG_EVAL_PORT` | `8765` |

Example — larger Whisper, longer LLM budget:

```bash
LANG_EVAL_WHISPER=mlx-community/whisper-large-v3-turbo LANG_EVAL_MAX_TOKENS=4000 ./run.sh
```

## Reference timings

2:20 mono mp3, whisper-medium + Qwen2.5-7B-4bit on this Mac:
run 1 ≈ 21s, run 2 ≈ 40s (28 chunks, in parallel with run 1), run 3 ≈ 28s
(anchored, 378 tokens — judgements are far cheaper than a rewritten transcript).

## Requirements

- **Apple Silicon Mac** — the pipeline runs on MLX, which is Metal-only.
- `ffmpeg` and `ffprobe` on PATH.
- The Python packages in `requirements.txt`, installed into the interpreter that
  `run.sh` picks (`LANG_EVAL_PYTHON`, else `/opt/homebrew/bin/python3.13`, else `python3`):

```bash
python3 -m pip install -r requirements.txt
```

Whisper and LLM weights download from Hugging Face on first use and are cached
under `~/.cache/huggingface`. Audio you load is copied into `uploads/`, which is
git-ignored — nothing you transcribe is committed.
