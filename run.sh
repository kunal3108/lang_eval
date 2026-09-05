#!/bin/bash
# Lang-Eval launcher.
#
# MLX only ships for Apple Silicon, and on this Mac mlx_whisper / mlx_lm live in
# the Homebrew 3.13 rather than the default python3 — so prefer that, and fall
# back to whatever python3 is on PATH elsewhere.
set -e
cd "$(dirname "$0")"

if [ -n "$LANG_EVAL_PYTHON" ]; then
  PY="$LANG_EVAL_PYTHON"
elif [ -x /opt/homebrew/bin/python3.13 ]; then
  PY=/opt/homebrew/bin/python3.13
else
  PY=python3
fi

if ! "$PY" -c "import mlx_whisper, mlx_lm" 2>/dev/null; then
  echo "$PY is missing mlx_whisper / mlx_lm." >&2
  echo "Install them there ($PY -m pip install -r requirements.txt)," >&2
  echo "or point LANG_EVAL_PYTHON at the interpreter that has them." >&2
  exit 1
fi

exec "$PY" app.py
