"""Small runtime diagnostics for hosted QA startup analysis.

- Emits flushed startup messages so Render captures them before abrupt process
  exits such as out-of-memory kills.
- Reads Linux RSS from `/proc/self/status` when available without adding any
  non-standard dependencies to the Render image.
- Keeps diagnostics optional and low-noise for local development.
"""

from __future__ import annotations

from pathlib import Path

_PROC_STATUS_PATH = Path("/proc/self/status")
_RSS_PREFIX = "VmRSS:"


def emit_runtime_diagnostic(message: str) -> None:
    """Print one flushed diagnostic line with RSS when available."""

    rss_megabytes = _read_linux_rss_megabytes()
    if rss_megabytes is None:
        print(message, flush=True)
        return
    print(f"{message} rss_mb={rss_megabytes:.1f}", flush=True)


def _read_linux_rss_megabytes() -> float | None:
    """Return the current RSS in MiB from `/proc/self/status`, if available."""

    if not _PROC_STATUS_PATH.exists():
        return None
    try:
        with open(_PROC_STATUS_PATH, encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith(_RSS_PREFIX):
                    continue
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


__all__ = ["emit_runtime_diagnostic"]
