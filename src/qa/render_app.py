"""Gunicorn entry point for the hosted QA web app.

- Reuses the same shared QA runtime as the local CLI entry point so Render and
  local development stay behaviorally aligned.
- Supports a temporary bootstrap mode that uploads the production corpus and
  ready cache to Render's persistent disk before the full QA runtime boots.
- Exposes a stable module-level `app` object for WSGI servers such as Gunicorn.
- Does not parse CLI flags or bind sockets directly.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path

from .diagnostics import emit_runtime_diagnostic

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

if os.environ.get("QA_BOOTSTRAP_MODE", "").strip() == "1":
    from .bootstrap_app import create_bootstrap_app

    emit_runtime_diagnostic("Render QA bootstrap mode enabled.")
    app = create_bootstrap_app()
else:
    from .runtime import build_qa_browser_runtime

    _RUNTIME = build_qa_browser_runtime(_PROJECT_ROOT)
    emit_runtime_diagnostic(
        "Render QA runtime ready: "
        f"backend={_RUNTIME.retrieval_backend} "
        f"chunk_count={_RUNTIME.chunk_count}"
    )
    atexit.register(_RUNTIME.close)
    app = _RUNTIME.app

__all__ = ["app"]
