"""Gunicorn entry point for the hosted QA web app.

- Reuses the same shared QA runtime as the local CLI entry point so Render and
  local development stay behaviorally aligned.
- Loads provider credentials from environment variables or keyring via the
  existing QA config path resolution.
- Exposes a stable module-level `app` object for WSGI servers such as Gunicorn.
- Does not parse CLI flags or bind sockets directly.
"""

from __future__ import annotations

import atexit
from pathlib import Path

from .runtime import build_qa_browser_runtime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME = build_qa_browser_runtime(_PROJECT_ROOT)
atexit.register(_RUNTIME.close)
app = _RUNTIME.app

__all__ = ["app"]
