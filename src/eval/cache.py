"""Per-item JSONL cache used by every judge-backed stage.

- Caches one row per (bill, item) key under
  ``output/evals/v1/cache/<method>/stage<N>/<bill_id>.jsonl`` so a killed run
  can restart and only pay for the items it has not judged yet.
- Reads the cache on stage startup to skip items that already have a row,
  and appends new rows atomically as judge calls complete.
- Does not touch the extractor run outputs, does not call the judge, and
  does not know anything about stage-specific schemas; callers choose which
  JSON payload to persist.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CachePaths:
    """Resolved directory layout for one stage's cache.

    Attributes:
        base_dir: Root of the cache tree for this stage and method.
    """

    base_dir: Path

    def file_for(self, bill_id: str) -> Path:
        """Return the JSONL file for one bill id (created by the writer)."""

        return self.base_dir / f"{_sanitise_bill_id(bill_id)}.jsonl"


def cache_paths_for_stage(
    *,
    run_dir: Path,
    stage: int,
    method: str | None = None,
) -> CachePaths:
    """Build the per-stage cache directory.

    Args:
        run_dir: Top-level eval run dir (``output/evals/v1``).
        stage: Stage number (``3``, ``4``, ``6``, ``8``).
        method: Method name when the stage writes per-method caches; omit
            for cross-method caches like Stage 6 pairwise.

    Returns:
        :class:`CachePaths` whose ``base_dir`` exists after the call.
    """

    parts: list[str] = ["cache"]
    if method:
        parts.append(method)
    parts.append(f"stage{stage}")
    base_dir = run_dir.joinpath(*parts)
    base_dir.mkdir(parents=True, exist_ok=True)
    return CachePaths(base_dir=base_dir)


def read_cache_keys(paths: CachePaths, bill_id: str) -> set[str]:
    """Return the set of item keys already cached for one bill.

    Each cache row must carry a top-level ``"key"`` field that uniquely
    identifies the item within the bill (e.g. a quadruplet id for Stage 3,
    or ``f"{bill_id}::{label}"`` for Stage 4). Rows with no key are ignored.

    Args:
        paths: Stage cache paths.
        bill_id: Composite bill identifier.

    Returns:
        Set of string keys already present on disk.
    """

    file_path = paths.file_for(bill_id)
    if not file_path.is_file():
        return set()
    keys: set[str] = set()
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "Cache row %d in %s is malformed; ignoring", line_no, file_path
                )
                continue
            key = row.get("key")
            if key is not None:
                keys.add(str(key))
    return keys


def iter_cache_rows(paths: CachePaths, bill_id: str) -> Iterator[dict[str, Any]]:
    """Yield every cached row for one bill in insertion order."""

    file_path = paths.file_for(bill_id)
    if not file_path.is_file():
        return
    with file_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed cache row in %s", file_path)


def read_all_rows(paths: CachePaths) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(bill_id, row)`` pairs across every JSONL under the cache dir.

    Bill ids are recovered from the filename stem and have their sanitisation
    reversed so callers see the original composite id.
    """

    for file_path in sorted(paths.base_dir.glob("*.jsonl")):
        bill_id = _unsanitise_bill_id(file_path.stem)
        for row in _read_rows_from_file(file_path):
            yield bill_id, row


def wipe_cache(paths: CachePaths) -> None:
    """Remove every JSONL file under the cache dir (called on ``--no-resume``)."""

    if not paths.base_dir.is_dir():
        return
    for file_path in paths.base_dir.glob("*.jsonl"):
        try:
            file_path.unlink()
        except OSError:
            logger.exception("Failed to unlink cache file %s", file_path)


class CacheWriter:
    """Thread-safe appender for one stage's JSONL cache.

    The writer serialises concurrent ``append`` calls so async stages can
    write from inside ``asyncio.gather`` fan-outs without interleaving rows.
    It keeps one file handle open per bill for the lifetime of the writer
    and closes them all in :meth:`close`.
    """

    def __init__(self, paths: CachePaths) -> None:
        """Open a writer rooted at ``paths``.

        Args:
            paths: Stage cache paths built by :func:`cache_paths_for_stage`.
        """

        self._paths = paths
        self._locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def append(self, *, bill_id: str, key: str, payload: dict[str, Any]) -> None:
        """Append one JSON row with a stable key to the bill's JSONL file.

        Args:
            bill_id: Composite bill identifier.
            key: Per-item key used by :func:`read_cache_keys` to skip
                re-judging on resume.
            payload: Additional fields to persist alongside ``key``.
        """

        row = {"key": key, **payload}
        file_path = self._paths.file_for(bill_id)
        lock = self._lock_for(bill_id)
        serialised = json.dumps(row, ensure_ascii=False)
        with lock:
            with file_path.open("a", encoding="utf-8") as handle:
                handle.write(serialised)
                handle.write("\n")

    def close(self) -> None:
        """Release any internal state.

        This implementation opens one handle per append so there is nothing
        to release, but the method is kept so the interface matches the
        more elaborate writer we may adopt later.
        """

        return None

    def _lock_for(self, bill_id: str) -> threading.Lock:
        with self._lock:
            lock = self._locks.get(bill_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[bill_id] = lock
            return lock


def filter_pending(
    items: Iterable[tuple[str, str, Any]],
    paths: CachePaths,
) -> list[tuple[str, str, Any]]:
    """Return only the ``(bill_id, key, payload)`` items not yet cached.

    Designed for the common pattern where a stage builds one work item per
    quadruplet or per (bill, label) pair and wants the orchestrator to
    short-circuit work it has already done.

    Args:
        items: Iterable of ``(bill_id, key, payload)`` tuples. ``payload``
            is opaque here; callers typically pass the original object so
            they can re-use it after filtering.
        paths: Stage cache paths whose existing keys define what to skip.

    Returns:
        List of items whose ``(bill_id, key)`` pair has no cached row.
    """

    seen: dict[str, set[str]] = {}
    pending: list[tuple[str, str, Any]] = []
    for bill_id, key, payload in items:
        keys = seen.get(bill_id)
        if keys is None:
            keys = read_cache_keys(paths, bill_id)
            seen[bill_id] = keys
        if key in keys:
            continue
        pending.append((bill_id, key, payload))
    return pending


def _read_rows_from_file(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON rows from a single cache file, skipping malformed lines."""

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed cache row in %s", path)


def _sanitise_bill_id(bill_id: str) -> str:
    """Collapse filesystem-hostile characters so bill ids round-trip safely."""

    return bill_id.replace("/", "_").replace("\\", "_")


def _unsanitise_bill_id(stem: str) -> str:
    """Inverse of :func:`_sanitise_bill_id`; current impl is a no-op stub.

    Bill ids contain only alphanumerics, spaces and ``__`` separators so the
    sanitiser above is effectively the identity function for our corpus.
    """

    return stem
