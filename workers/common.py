"""Shared helpers for Lang-Eval workers.

Workers talk to the server over stdout using one JSON object per line,
prefixed with a sentinel so that any stray library output is ignored.
"""

import json
import sys

SENTINEL = "@@LE@@"


def emit(**payload):
    sys.stdout.write(SENTINEL + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()
