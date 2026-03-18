"""Canonical constants and stable-id helpers for the NER pipeline.

- Defines the canonical field order shared by all field-wise outputs.
- Defines the canonical refinement relation label set.
- Defines deterministic id derivation used across reruns and resumes.
"""

from __future__ import annotations

import hashlib

CANONICAL_FIELD_ORDER: tuple[str, str, str, str] = (
    "entity",
    "type",
    "attribute",
    "value",
)

CANONICAL_RELATION_LABELS: tuple[str, str, str, str, str] = (
    "support",
    "overlap",
    "conflict",
    "duplicate",
    "refinement",
)

_STABLE_ID_DIGEST_BYTES = 8


def stable_int_id(namespace: str, *parts: object) -> int:
    """Return a deterministic integer id for the supplied namespace and parts.

    The NER pipeline needs ids that survive reruns and stage resumes. Hashing
    the namespace plus stable input parts keeps ids deterministic without
    relying on mutable counters that drift across reruns.

    Args:
        namespace (str): Logical namespace that separates unrelated id
            families.
        *parts (object): Stable values used to derive the deterministic
            identifier.

    Returns:
        int: Deterministic integer identifier derived from the namespace and
            parts.
    """

    payload = "||".join(str(part) for part in (namespace, *parts))
    digest = hashlib.blake2b(
        payload.encode("utf-8"),
        digest_size=_STABLE_ID_DIGEST_BYTES,
    ).hexdigest()
    return int(digest, 16)

