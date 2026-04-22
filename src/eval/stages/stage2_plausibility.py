"""Stage 2: rule-based fabrication pre-filter.

- Narrow scope: catches unambiguous fabrications so Stage 3 (the LLM
  judge) does not have to spend tokens on obviously-broken extractions.
  Semantic grounding (does the span actually support the quadruplet?) is
  deferred to Stage 3's RefChecker-style NLI, not approximated here with
  literal string matching.
- For every quadruplet in both methods, verifies that:
    1. the four required fields (``entity`` / ``type`` / ``attribute`` /
       ``value``) are all non-empty,
    2. at least one evidence span is present and has a non-empty
       ``[start, end)`` offset range,
    3. every non-empty span's stored ``text`` field actually appears in
       the bill text; offset correctness is *not* required because the
       two methods use different chunking schemes and the skill-driven
       extractor reports chunk-relative offsets. See
       :func:`_span_matches_bill_text` for the matcher.
- Earlier versions also enforced ``entity_surface_form_missing`` and
  ``value_surface_form_missing`` checks (requiring the literal entity /
  value text to appear inside its span). That was dropped because it
  double-counts Stage 3's work and over-filters extractors that
  paraphrase values by design (e.g. the orchestrated Claude run), which
  would corrupt every downstream comparison.
- Writes ``results/stage2_plausibility.json`` containing the pass / fail
  decision per quadruplet, the pass-set used by downstream stages, and a
  per-method summary.
- Pure rule-based: no judge calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..artifacts import EvidenceSpan, Quadruplet, StageResult
from ._common import StageContext, write_result_json

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = ("entity", "type", "attribute", "value")


@dataclass(slots=True)
class _PerQuadrupletDecision:
    """Internal row recording why a single quadruplet passed or failed."""

    quadruplet_id: str
    bill_id: str
    method: str
    passed: bool
    reasons: list[str]


def run(ctx: StageContext) -> StageResult:
    """Apply the plausibility rules to every quadruplet and emit results.

    Args:
        ctx: Pre-built :class:`StageContext`; bill records provide bill
            text used to verify literal-substring checks.

    Returns:
        :class:`StageResult` with pass / fail counts per method and per
        failure reason.
    """

    per_method: dict[str, list[_PerQuadrupletDecision]] = {}
    summary: dict[str, dict[str, Any]] = {}
    pass_set: dict[str, dict[str, list[str]]] = {}

    for method_name, bills in ctx.method_outputs.items():
        decisions: list[_PerQuadrupletDecision] = []
        for bill_id, quadruplets in bills.items():
            bill_text = _bill_text(ctx, bill_id)
            for quadruplet in quadruplets:
                decision = _evaluate(quadruplet, bill_text=bill_text)
                decisions.append(decision)
        per_method[method_name] = decisions
        summary[method_name] = _summarise(decisions)
        pass_set[method_name] = _extract_pass_set(decisions)

    payload: dict[str, Any] = {
        "stage": 2,
        "summary": summary,
        "passed": pass_set,
        "decisions": {
            method: [
                {
                    "quadruplet_id": d.quadruplet_id,
                    "bill_id": d.bill_id,
                    "passed": d.passed,
                    "reasons": d.reasons,
                }
                for d in decisions
            ]
            for method, decisions in per_method.items()
        },
    }
    result_file = write_result_json(
        ctx, filename="stage2_plausibility.json", payload=payload
    )

    overall_pass = sum(
        entry["pass_count"] for entry in summary.values()
    )
    overall_total = sum(entry["total"] for entry in summary.values())
    logger.info(
        "Stage 2 done: total=%d pass=%d fail=%d  per-method=%s",
        overall_total,
        overall_pass,
        overall_total - overall_pass,
        summary,
    )
    return StageResult(
        stage=2,
        status="completed",
        summary=(
            f"Plausibility: {overall_pass}/{overall_total} quadruplets passed "
            f"across {len(summary)} methods."
        ),
        metrics={"per_method": summary},
        artifacts={"result_file": str(result_file)},
    )


def _evaluate(
    quadruplet: Quadruplet, *, bill_text: str
) -> _PerQuadrupletDecision:
    """Apply the fabrication-only rules to a single quadruplet.

    Checks the four required fields are non-empty, that at least one
    evidence span is present and non-empty, and that every non-empty
    span resolves to real bill text (via the relaxed matcher). Any
    semantic claim about whether the span actually grounds the
    quadruplet is deferred to Stage 3.
    """

    reasons: list[str] = []
    for field in _REQUIRED_FIELDS:
        value = getattr(quadruplet, field)
        if not value:
            reasons.append(f"empty_field:{field}")
    all_spans = quadruplet.all_spans()
    if not all_spans:
        reasons.append("no_evidence_span")
    else:
        if not _any_span_non_empty(all_spans):
            reasons.append("all_evidence_spans_empty")
        if not bill_text:
            reasons.append("bill_text_missing")
        else:
            for span in all_spans:
                if not _span_matches_bill_text(span, bill_text):
                    reasons.append(
                        f"span_not_literal:{span.start}-{span.end}"
                    )
                    break

    return _PerQuadrupletDecision(
        quadruplet_id=quadruplet.quadruplet_id,
        bill_id=quadruplet.bill_id,
        method=quadruplet.method,
        passed=not reasons,
        reasons=reasons,
    )


def _any_span_non_empty(spans: list[EvidenceSpan]) -> bool:
    """Return whether at least one evidence span has an end > start."""

    return any(span.end > span.start for span in spans)


_SPAN_FALLBACK_STRIP_CHARS = ".,;:!?\"'"


def _span_matches_bill_text(span: EvidenceSpan, bill_text: str) -> bool:
    """Return whether ``span`` resolves to text that actually exists in the bill.

    The plausibility question is narrow: does the span's stored ``text``
    field correspond to real bill content, or was it fabricated? Offset
    correctness is deliberately *not* required -- the orchestrated and
    skill-driven extractors use different chunking schemes, and the
    skill-driven run reports offsets relative to its chunk stream rather
    than the full NCSL ``text`` field, producing drifts of tens to
    hundreds of characters. Downstream stages (3, 4, 6, 8) consume the
    span's ``text`` field and the bill text, never the raw offsets, so
    offset drift is harmless.

    The match runs in three tiers, cheapest first:

    1. Offset sanity for empty-text spans: when ``span.text`` is empty we
       can only check that the offsets form a valid, non-empty slice
       against the bill.
    2. Exact slice equality at the reported offsets, i.e.
       ``bill_text[span.start:span.end] == span.text``. This is the fast
       path for extractors whose offsets are faithful.
    3. Substring fallback: the stripped span text must appear anywhere
       in the bill text. This catches both whitespace / trailing-
       punctuation stripping and large offset drift from chunk-relative
       numbering, while still rejecting genuinely fabricated span text.

    Spans whose stripped text is empty after normalisation fail the
    check -- there is no useful content to verify.
    """

    if not span.text:
        if span.start < 0 or span.end > len(bill_text) or span.start >= span.end:
            return False
        return True
    if (
        0 <= span.start < span.end <= len(bill_text)
        and bill_text[span.start : span.end] == span.text
    ):
        return True
    needle = span.text.strip().strip(_SPAN_FALLBACK_STRIP_CHARS)
    if not needle:
        return False
    return needle in bill_text


def _bill_text(ctx: StageContext, bill_id: str) -> str:
    """Look up the NCSL bill text for a composite bill id."""

    record = ctx.bill_records.get(bill_id)
    return record.text if record else ""


def _summarise(decisions: list[_PerQuadrupletDecision]) -> dict[str, Any]:
    """Aggregate per-quadruplet decisions to a small method summary."""

    total = len(decisions)
    passed = sum(1 for d in decisions if d.passed)
    reason_counts: dict[str, int] = {}
    for decision in decisions:
        for reason in decision.reasons:
            key = reason.split(":", 1)[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1
    return {
        "total": total,
        "pass_count": passed,
        "fail_count": total - passed,
        "pass_rate": (passed / total) if total else 0.0,
        "fail_reasons": reason_counts,
    }


def _extract_pass_set(
    decisions: list[_PerQuadrupletDecision],
) -> dict[str, list[str]]:
    """Collect the quadruplet ids that passed, grouped by bill id."""

    bills: dict[str, list[str]] = {}
    for decision in decisions:
        if not decision.passed:
            continue
        bills.setdefault(decision.bill_id, []).append(decision.quadruplet_id)
    return bills
