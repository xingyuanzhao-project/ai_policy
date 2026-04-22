"""In-memory store backing the QA ``query_quadruplets`` planner tool.

- Loads the pre-built sidecar JSONL (``<cache_dir>/quadruplets.jsonl``)
  produced by ``scripts/build_quadruplet_sidecar.py`` from the skill-driven
  NER pipeline.
- Exposes filter-based lookup over the flattened records so the planner can
  answer structured questions (``bills that prohibit X``, ``disclosure
  obligations in California``) without touching the chunk embeddings or the
  bill summary index.
- Keeps the store read-only and self-contained: no retriever, no LLM call,
  no dependency on ``BillSummary``.

The sidecar is optional. When the file is missing (e.g. a fresh checkout
without the skill-NER outputs), :func:`load_quadruplet_store` returns an
empty store so the QA service keeps booting and the planner tool simply
reports ``match_count: 0``.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QuadrupletSpan:
    """One evidence span from the bill text."""

    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dict view of the span."""

        return {"start": self.start, "end": self.end, "text": self.text}


@dataclass(frozen=True, slots=True)
class QuadrupletRecord:
    """One flattened (entity, type, attribute, value) extraction for a bill.

    The four planner-facing fields (``regulated_entity``, ``entity_type``,
    ``regulatory_mechanism``, ``provision_text``) map one-to-one onto the
    source skill-NER fields (``entity``, ``type``, ``attribute``, ``value``)
    and are pre-stripped at build time.
    """

    bill_id: str
    state: str
    year: int
    regulated_entity: str
    entity_type: str
    regulatory_mechanism: str
    provision_text: str
    entity_span: QuadrupletSpan | None = None
    provision_span: QuadrupletSpan | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dict with only the planner-facing fields."""

        payload: dict[str, Any] = {
            "bill_id": self.bill_id,
            "state": self.state,
            "year": self.year,
            "regulated_entity": self.regulated_entity,
            "entity_type": self.entity_type,
            "regulatory_mechanism": self.regulatory_mechanism,
            "provision_text": self.provision_text,
        }
        if self.entity_span is not None:
            payload["entity_span"] = self.entity_span.to_dict()
        if self.provision_span is not None:
            payload["provision_span"] = self.provision_span.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class QuadrupletVocabulary:
    """Top-K category counts exposed as hints for the planner prompt."""

    entity_types: tuple[tuple[str, int], ...]
    regulatory_mechanisms: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Return a JSON-ready dict view of the vocabulary counts."""

        return {
            "entity_types": [
                {"name": name, "count": count} for name, count in self.entity_types
            ],
            "regulatory_mechanisms": [
                {"name": name, "count": count}
                for name, count in self.regulatory_mechanisms
            ],
        }


class QuadrupletStore:
    """Read-only, filterable view over the flattened quadruplet sidecar."""

    def __init__(self, records: Sequence[QuadrupletRecord]) -> None:
        self._records: tuple[QuadrupletRecord, ...] = tuple(records)

    @classmethod
    def empty(cls) -> "QuadrupletStore":
        """Return a store with zero records (used when the sidecar is absent)."""

        return cls(())

    @classmethod
    def from_jsonl(cls, path: Path) -> "QuadrupletStore":
        """Load one JSON object per line from ``path`` into a store."""

        records: list[QuadrupletRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as error:
                    logger.warning(
                        "QuadrupletStore.from_jsonl: skipping line %d of %s: %s",
                        line_number,
                        path,
                        error,
                    )
                    continue
                record = _record_from_payload(payload)
                if record is not None:
                    records.append(record)
        return cls(records)

    def __len__(self) -> int:
        """Return the total record count stored in memory."""

        return len(self._records)

    @property
    def total_quadruplets(self) -> int:
        """Return the total record count stored in memory."""

        return len(self._records)

    def search(
        self,
        *,
        regulated_entity: str | None = None,
        entity_type: Sequence[str] | None = None,
        regulatory_mechanism: str | None = None,
        provision_contains: str | None = None,
        state: Sequence[str] | None = None,
        year: Sequence[int] | None = None,
        bill_id: Sequence[str] | None = None,
        limit: int = 30,
    ) -> list[QuadrupletRecord]:
        """Return the first ``limit`` records that satisfy every active filter.

        Filter semantics:

        - String needles (``regulated_entity``, ``regulatory_mechanism``,
          ``provision_contains``) match as case-insensitive substrings.
        - Categorical sets (``entity_type``, ``state``, ``bill_id``) match
          case-insensitively so ``"AI Application"``, ``"ai application"``,
          and ``"AI application"`` all resolve to the same records. Callers
          that need USPS <-> full-state-name aliasing should resolve state
          inputs against :meth:`state_vocabulary` with
          :func:`src.qa.filter_normalizers.normalize_states` before calling
          ``search``.
        - ``year`` matches by integer equality.
        - All categorical sets use OR within the list and AND across fields.

        Iteration order is the insertion order of the sidecar, which is the
        sorted filesystem order of the per-bill skill-NER outputs.
        """

        if limit <= 0:
            return []

        entity_needle = _lower_substring(regulated_entity)
        mechanism_needle = _lower_substring(regulatory_mechanism)
        provision_needle = _lower_substring(provision_contains)
        entity_type_set = _lower_str_set(entity_type)
        state_set = _lower_str_set(state)
        bill_id_set = _lower_str_set(bill_id)
        year_set = _clean_int_set(year)

        matches: list[QuadrupletRecord] = []
        for record in self._records:
            if entity_needle and entity_needle not in record.regulated_entity.lower():
                continue
            if (
                mechanism_needle
                and mechanism_needle not in record.regulatory_mechanism.lower()
            ):
                continue
            if provision_needle and provision_needle not in record.provision_text.lower():
                continue
            if entity_type_set and record.entity_type.lower() not in entity_type_set:
                continue
            if state_set and record.state.lower() not in state_set:
                continue
            if year_set and int(record.year) not in year_set:
                continue
            if bill_id_set and record.bill_id.lower() not in bill_id_set:
                continue
            matches.append(record)
            if len(matches) >= limit:
                break
        return matches

    def state_vocabulary(self) -> tuple[str, ...]:
        """Return every distinct state value present in the sidecar.

        Callers use this as the ``canonical_states`` argument to
        :func:`src.qa.filter_normalizers.normalize_states` so a planner
        filter like ``state="TX"`` still resolves against a store that
        indexes ``state="Texas"``.
        """

        return tuple(sorted({record.state for record in self._records if record.state}))

    def entity_type_vocabulary(self) -> tuple[str, ...]:
        """Return every distinct ``entity_type`` value present in the sidecar.

        Used by the planner tool handler to case-fold incoming
        ``entity_type`` arguments against the full category set (not just
        the top-K slice returned by :meth:`vocabulary`).
        """

        return tuple(
            sorted(
                {record.entity_type for record in self._records if record.entity_type}
            )
        )

    def vocabulary(
        self,
        *,
        top_entity_types: int = 15,
        top_mechanisms: int = 15,
    ) -> QuadrupletVocabulary:
        """Return the most common entity_type and regulatory_mechanism values."""

        type_counter: Counter[str] = Counter()
        mechanism_counter: Counter[str] = Counter()
        for record in self._records:
            type_counter[record.entity_type] += 1
            mechanism_counter[record.regulatory_mechanism] += 1
        return QuadrupletVocabulary(
            entity_types=tuple(type_counter.most_common(top_entity_types)),
            regulatory_mechanisms=tuple(
                mechanism_counter.most_common(top_mechanisms)
            ),
        )


def load_quadruplet_store(path: Path) -> QuadrupletStore:
    """Load the sidecar at ``path`` or return an empty store when absent.

    Missing or unreadable files are treated as a non-fatal "feature off"
    signal so the QA service still boots without the sidecar. Line-level
    JSON parse errors are logged at WARNING level with line numbers and the
    offending line is skipped.
    """

    if not path.is_file():
        logger.info(
            "QuadrupletStore: sidecar not found at %s; query_quadruplets tool "
            "will return empty results",
            path,
        )
        return QuadrupletStore.empty()
    try:
        store = QuadrupletStore.from_jsonl(path)
    except OSError as error:
        logger.warning(
            "QuadrupletStore: failed to read sidecar %s (%s); continuing empty",
            path,
            error,
        )
        return QuadrupletStore.empty()
    logger.info(
        "QuadrupletStore: loaded %d quadruplets from %s",
        len(store),
        path,
    )
    return store


def _record_from_payload(payload: Any) -> QuadrupletRecord | None:
    """Convert a sidecar JSONL row into a :class:`QuadrupletRecord`.

    Rows missing any of the five mandatory string fields (``bill_id``,
    ``regulated_entity``, ``entity_type``, ``regulatory_mechanism``,
    ``provision_text``) are skipped; ``state`` and ``year`` default to empty
    string and ``0`` respectively to tolerate legacy rows.
    """

    if not isinstance(payload, dict):
        return None
    bill_id = _clean_str(payload.get("bill_id"))
    regulated_entity = _clean_str(payload.get("regulated_entity"))
    entity_type = _clean_str(payload.get("entity_type"))
    regulatory_mechanism = _clean_str(payload.get("regulatory_mechanism"))
    provision_text = _clean_str(payload.get("provision_text"))
    if not (
        bill_id
        and regulated_entity
        and entity_type
        and regulatory_mechanism
        and provision_text
    ):
        return None

    state = _clean_str(payload.get("state"))
    try:
        year = int(payload.get("year", 0))
    except (TypeError, ValueError):
        year = 0

    entity_span = _span_from_payload(payload.get("entity_span"))
    provision_span = _span_from_payload(payload.get("provision_span"))

    return QuadrupletRecord(
        bill_id=bill_id,
        state=state,
        year=year,
        regulated_entity=regulated_entity,
        entity_type=entity_type,
        regulatory_mechanism=regulatory_mechanism,
        provision_text=provision_text,
        entity_span=entity_span,
        provision_span=provision_span,
    )


def _span_from_payload(payload: Any) -> QuadrupletSpan | None:
    """Convert a JSON span fragment into a :class:`QuadrupletSpan`."""

    if not isinstance(payload, dict):
        return None
    try:
        start = int(payload.get("start"))
        end = int(payload.get("end"))
    except (TypeError, ValueError):
        return None
    text = payload.get("text")
    if not isinstance(text, str):
        text = ""
    return QuadrupletSpan(start=start, end=end, text=text)


def _clean_str(value: Any) -> str:
    """Return ``value`` coerced to a stripped string, or ``""`` if not a str."""

    if not isinstance(value, str):
        return ""
    return value.strip()


def _clean_str_set(values: Iterable[Any] | None) -> set[str]:
    """Return an exact-match set of non-empty stripped strings."""

    if values is None:
        return set()
    cleaned: set[str] = set()
    for item in values:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                cleaned.add(stripped)
    return cleaned


def _lower_str_set(values: Iterable[Any] | None) -> set[str]:
    """Return a case-insensitive set of non-empty stripped strings.

    Used inside :meth:`QuadrupletStore.search` so the categorical filters
    (``entity_type``, ``state``, ``bill_id``) compare against lowercased
    record fields. Callers still get to normalize state inputs via the
    USPS alias map before calling ``search``; that normalization produces
    the canonical form, and this helper only handles the case-fold.
    """

    if values is None:
        return set()
    cleaned: set[str] = set()
    for item in values:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                cleaned.add(stripped.lower())
    return cleaned


def _clean_int_set(values: Iterable[Any] | None) -> set[int]:
    """Return an exact-match set of integers coerced from ``values``."""

    if values is None:
        return set()
    cleaned: set[int] = set()
    for item in values:
        try:
            cleaned.add(int(item))
        except (TypeError, ValueError):
            continue
    return cleaned


def _lower_substring(value: Any) -> str:
    """Return a lowercase, trimmed substring needle or ``""`` for no filter."""

    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    if not stripped:
        return ""
    return stripped.lower()


__all__ = [
    "QuadrupletRecord",
    "QuadrupletSpan",
    "QuadrupletStore",
    "QuadrupletVocabulary",
    "load_quadruplet_store",
]
