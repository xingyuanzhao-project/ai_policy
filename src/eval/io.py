"""Corpus and extractor-run loaders for the eval pipeline.

- Loads per-bill quadruplet JSONs produced by both the orchestrated
  (``src/ner``) and skill-driven (``src/skill_ner``) runners into the common
  :class:`~src.eval.artifacts.Quadruplet` shape.
- Loads the NCSL corpus (bill text from JSONL + bill metadata from CSV) and
  splits the free-text ``topics`` column on both ``;`` and ``,`` so the
  downstream Stage 4 coverage judge sees clean, individual labels.
- Does not make any LLM calls, does not mutate the extractor outputs, and
  does not write any files.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .artifacts import BillRecord, EvidenceSpan, Quadruplet
from .config import CorpusConfig, MethodConfig

logger = logging.getLogger(__name__)

_TOPIC_SPLIT_CHARS = (";", ",")


def load_run_outputs(
    method: MethodConfig,
    *,
    bill_ids: Iterable[str] | None = None,
) -> dict[str, list[Quadruplet]]:
    """Load every per-bill quadruplet JSON for one extractor run.

    The orchestrated (``src/ner``) and skill-driven (``src/skill_ner``) runs
    both serialise the same outer shape: ``{"bill_id", "source_bill_id",
    "year", "state", "quadruplets": [...]}``. This loader normalises each
    row into a :class:`Quadruplet` regardless of minor per-run cosmetic
    differences (e.g. ``text_length`` / ``agent_turns`` present only for the
    skill-driven run).

    Args:
        method: Method config whose ``outputs_dir`` contains per-bill JSONs.
        bill_ids: Optional allow-list of composite ``{year}__{state_id}``
            bill ids. When supplied, only those bills are loaded.

    Returns:
        Map of ``bill_id`` to its list of parsed quadruplets (empty list
        when the bill's JSON had an empty ``quadruplets`` array).
    """

    outputs_dir = method.outputs_dir
    allow: set[str] | None = set(bill_ids) if bill_ids is not None else None
    result: dict[str, list[Quadruplet]] = {}

    for path in sorted(outputs_dir.glob("*.json")):
        bill_id = path.stem
        if allow is not None and bill_id not in allow:
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload: dict[str, Any] = json.load(handle)
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to read run output: %s", path)
            continue

        stored_bill_id = str(payload.get("bill_id") or bill_id)
        if stored_bill_id != bill_id:
            logger.debug(
                "Filename/bill_id mismatch for %s: file=%s payload=%s",
                method.name,
                bill_id,
                stored_bill_id,
            )
        rows = payload.get("quadruplets") or []
        result[stored_bill_id] = [
            _parse_quadruplet(row, bill_id=stored_bill_id, method=method.name, index=i)
            for i, row in enumerate(rows)
        ]

    logger.info(
        "Loaded %s quadruplet files: bills=%d  quadruplets=%d",
        method.name,
        len(result),
        sum(len(v) for v in result.values()),
    )
    return result


def iter_bill_records(
    corpus: CorpusConfig,
    *,
    bill_ids: Iterable[str] | None = None,
) -> Iterator[BillRecord]:
    """Yield :class:`BillRecord` rows joined from NCSL metadata and text.

    The corpus join uses the composite ``{year}__{source_bill_id}`` key
    because that is what the NER runs emit as their output filename stem.
    Rows whose text JSONL entry is missing are skipped with a debug log.

    Args:
        corpus: Corpus file locations resolved by :mod:`config`.
        bill_ids: Optional allow-list of composite bill ids.

    Yields:
        One :class:`BillRecord` per NCSL bill that has both metadata and
        text available.
    """

    allow: set[str] | None = set(bill_ids) if bill_ids is not None else None

    text_index = _load_ncsl_text(corpus.ncsl_text)
    with corpus.ncsl_metadata.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            year_raw = (row.get("year") or "").strip()
            source_bill_id = (row.get("bill_id") or "").strip()
            if not year_raw or not source_bill_id:
                continue
            try:
                year = int(year_raw)
            except ValueError:
                continue
            composite = f"{year}__{source_bill_id}"
            if allow is not None and composite not in allow:
                continue
            text_row = text_index.get(composite)
            if text_row is None:
                logger.debug("No NCSL text row for %s", composite)
                continue
            topics_raw = row.get("topics") or ""
            yield BillRecord(
                bill_id=composite,
                source_bill_id=source_bill_id,
                year=year,
                state=(row.get("state") or "").strip(),
                text=text_row.get("text") or "",
                title=(row.get("title") or "").strip(),
                summary=(row.get("summary") or "").strip(),
                topics_raw=topics_raw,
                topics=split_topics(topics_raw),
            )


def load_bill_records(
    corpus: CorpusConfig,
    *,
    bill_ids: Iterable[str] | None = None,
) -> dict[str, BillRecord]:
    """Materialise :func:`iter_bill_records` into a dict keyed by bill id."""

    return {rec.bill_id: rec for rec in iter_bill_records(corpus, bill_ids=bill_ids)}


def split_topics(raw: str) -> list[str]:
    """Split an NCSL topics string on ``;`` or ``,`` and normalise casing.

    NCSL's ``topics`` column is hand-entered, so it mixes separators: about
    three quarters of rows use ``;`` between labels (e.g.
    ``"Government Use; Workforce"``) while the rest use commas. This splitter
    treats both as label boundaries, then trims and drops duplicates.

    Args:
        raw: The raw cell value.

    Returns:
        Ordered list of cleaned, de-duplicated topic labels.
    """

    if not raw:
        return []
    buffer = raw
    for ch in _TOPIC_SPLIT_CHARS:
        buffer = buffer.replace(ch, "\n")
    seen: set[str] = set()
    cleaned: list[str] = []
    for token in buffer.splitlines():
        label = token.strip().strip('"').strip()
        if not label:
            continue
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(label)
    return cleaned


def available_bill_ids_for_method(method: MethodConfig) -> list[str]:
    """Return sorted output filename stems for the given method's run."""

    return sorted(path.stem for path in method.outputs_dir.glob("*.json"))


def intersect_bill_ids(methods: Iterable[MethodConfig]) -> list[str]:
    """Return bill ids that are present in every method's output directory.

    Pairwise comparison and paired coverage tests require the same bill set
    across methods, so the orchestrator samples from this intersection.
    """

    method_list = list(methods)
    if not method_list:
        return []
    acc: set[str] | None = None
    for method in method_list:
        ids = set(available_bill_ids_for_method(method))
        acc = ids if acc is None else (acc & ids)
    return sorted(acc or set())


def _load_ncsl_text(text_jsonl: Path) -> dict[str, dict[str, Any]]:
    """Return a ``{composite_bill_id: text_row}`` map from the corpus JSONL."""

    index: dict[str, dict[str, Any]] = {}
    with text_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed NCSL text line %d in %s", line_no, text_jsonl
                )
                continue
            year = row.get("year")
            source_bill_id = row.get("bill_id")
            if year is None or not source_bill_id:
                continue
            composite = f"{year}__{source_bill_id}"
            index[composite] = row
    return index


def _parse_quadruplet(
    raw: dict[str, Any], *, bill_id: str, method: str, index: int
) -> Quadruplet:
    """Build a :class:`Quadruplet` from a single per-run JSON row."""

    raw_id = raw.get("refined_id")
    if raw_id is None or raw_id == 0:
        quadruplet_id = f"{bill_id}#{method}#{index}"
    else:
        quadruplet_id = f"{bill_id}#{method}#{raw_id}"
    return Quadruplet(
        bill_id=bill_id,
        method=method,
        quadruplet_id=quadruplet_id,
        entity=str(raw.get("entity") or ""),
        type=str(raw.get("type") or ""),
        attribute=str(raw.get("attribute") or ""),
        value=str(raw.get("value") or ""),
        entity_evidence=_parse_spans(raw.get("entity_evidence")),
        type_evidence=_parse_spans(raw.get("type_evidence")),
        attribute_evidence=_parse_spans(raw.get("attribute_evidence")),
        value_evidence=_parse_spans(raw.get("value_evidence")),
    )


def _parse_spans(raw: Any) -> list[EvidenceSpan]:
    """Normalise one evidence-span list; tolerate missing / malformed rows."""

    if not isinstance(raw, list):
        return []
    spans: list[EvidenceSpan] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start", 0))
            end = int(item.get("end", 0))
        except (TypeError, ValueError):
            continue
        spans.append(
            EvidenceSpan(
                start=start,
                end=end,
                text=str(item.get("text") or ""),
                chunk_id=int(item.get("chunk_id", 0)),
            )
        )
    return spans
