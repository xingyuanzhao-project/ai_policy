"""Robust normalization for QA filter inputs.

- Resolves USPS two-letter codes, full state names, case variations, and minor
  spelling drift into whatever canonical form is actually stored in the QA
  corpus (``bill_index`` states, ``STATUS_BUCKETS``, topic tags).
- Lets the planner call ``list_bills(filters={"state": "TX"})`` or
  ``query_quadruplets(state="tx")`` and still hit a corpus that stores
  ``"Texas"``.
- Does not call any LLM, touch the retrievers, or mutate the corpus; this is a
  pure lookup table plus case-insensitive + fuzzy matching against vocabularies
  the caller provides.

The canonical form is always determined by the iterable the caller passes
(``canonical_states``, ``canonical_values``, ``canonical_topics``). The
normalizer returns the exact string that appears in that iterable so the
caller can feed it straight into downstream exact-equality filters.
"""

from __future__ import annotations

import difflib
from typing import Iterable

US_STATE_ABBREV_TO_NAME: dict[str, str] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
    "PR": "Puerto Rico",
    "GU": "Guam",
    "VI": "U.S. Virgin Islands",
    "AS": "American Samoa",
    "MP": "Northern Mariana Islands",
}

US_STATE_NAME_TO_ABBREV: dict[str, str] = {
    name.lower(): abbrev for abbrev, name in US_STATE_ABBREV_TO_NAME.items()
}

_DEFAULT_TOPIC_FUZZY_CUTOFF = 0.85


def _build_lower_index(values: Iterable[str]) -> dict[str, str]:
    """Return ``{lowercase -> original}`` so callers can fold case cheaply.

    Duplicates keep the first-seen casing so the resolved form is stable
    across calls.
    """

    index: dict[str, str] = {}
    for value in values:
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key not in index:
            index[key] = stripped
    return index


def normalize_state(
    value: object,
    canonical_states: Iterable[str],
) -> str | None:
    """Return the canonical form of ``value`` as it appears in ``canonical_states``.

    Resolution order:
    1. Case-insensitive exact match against ``canonical_states``.
    2. If ``value`` parses as a USPS two-letter code, look up the full name and
       retry the case-insensitive match.
    3. If ``value`` parses as a full state name, look up the USPS code and
       retry the case-insensitive match.

    Returns ``None`` when no alias could be resolved. When ``canonical_states``
    is empty (no vocabulary available) the stripped input is returned as-is so
    callers can still apply the filter in tests / cold-boot paths.
    """

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None

    index = _build_lower_index(canonical_states)
    if not index:
        return stripped

    key = stripped.lower()
    if key in index:
        return index[key]

    upper_key = stripped.upper()
    if upper_key in US_STATE_ABBREV_TO_NAME:
        full_name_lower = US_STATE_ABBREV_TO_NAME[upper_key].lower()
        if full_name_lower in index:
            return index[full_name_lower]

    abbrev = US_STATE_NAME_TO_ABBREV.get(key)
    if abbrev is not None and abbrev.lower() in index:
        return index[abbrev.lower()]

    return None


def normalize_states(
    values: Iterable[object],
    canonical_states: Iterable[str],
) -> list[str]:
    """Resolve each entry in ``values`` to a canonical state, dropping misses.

    Preserves first-seen order and removes duplicates so the returned list is
    safe to feed into exact-equality filters downstream.
    """

    canonical_list = [
        str(entry).strip()
        for entry in canonical_states
        if isinstance(entry, str) and str(entry).strip()
    ]
    result: list[str] = []
    for value in values:
        resolved = normalize_state(value, canonical_list)
        if resolved and resolved not in result:
            result.append(resolved)
    return result


def normalize_categorical(
    value: object,
    canonical_values: Iterable[str],
) -> str | None:
    """Case-insensitive exact match of ``value`` against ``canonical_values``.

    Returns the canonical form as it appears in ``canonical_values`` or
    ``None`` when no match is found. An empty ``canonical_values`` iterable
    returns the stripped input unchanged.
    """

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    index = _build_lower_index(canonical_values)
    if not index:
        return stripped
    return index.get(stripped.lower())


def normalize_categoricals(
    values: Iterable[object],
    canonical_values: Iterable[str],
) -> list[str]:
    """Resolve each entry case-insensitively, dropping anything unresolved."""

    canonical_list = [
        str(entry).strip()
        for entry in canonical_values
        if isinstance(entry, str) and str(entry).strip()
    ]
    result: list[str] = []
    for value in values:
        resolved = normalize_categorical(value, canonical_list)
        if resolved and resolved not in result:
            result.append(resolved)
    return result


def normalize_topic(
    value: object,
    canonical_topics: Iterable[str],
    *,
    fuzzy_cutoff: float = _DEFAULT_TOPIC_FUZZY_CUTOFF,
) -> str | None:
    """Resolve ``value`` to a canonical topic tag with fuzzy fallback.

    Resolution order:
    1. Case-insensitive exact match.
    2. ``difflib.get_close_matches`` with ``fuzzy_cutoff`` ratio against the
       lowercase canonical keys.

    Returns the original (first-seen) casing from ``canonical_topics``. Raise
    ``fuzzy_cutoff`` to reject near-misses; lower it to be more permissive.
    """

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    index = _build_lower_index(canonical_topics)
    if not index:
        return stripped
    key = stripped.lower()
    if key in index:
        return index[key]
    matches = difflib.get_close_matches(
        key, list(index.keys()), n=1, cutoff=fuzzy_cutoff
    )
    if matches:
        return index[matches[0]]
    return None


def normalize_topics(
    values: Iterable[object],
    canonical_topics: Iterable[str],
    *,
    fuzzy_cutoff: float = _DEFAULT_TOPIC_FUZZY_CUTOFF,
) -> list[str]:
    """Resolve each entry with case-fold + fuzzy match, dropping unresolved."""

    canonical_list = [
        str(entry).strip()
        for entry in canonical_topics
        if isinstance(entry, str) and str(entry).strip()
    ]
    result: list[str] = []
    for value in values:
        resolved = normalize_topic(value, canonical_list, fuzzy_cutoff=fuzzy_cutoff)
        if resolved and resolved not in result:
            result.append(resolved)
    return result


__all__ = [
    "US_STATE_ABBREV_TO_NAME",
    "US_STATE_NAME_TO_ABBREV",
    "normalize_categorical",
    "normalize_categoricals",
    "normalize_state",
    "normalize_states",
    "normalize_topic",
    "normalize_topics",
]
