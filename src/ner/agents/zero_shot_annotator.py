"""Zero-shot Annotator for chunk-level candidate extraction.

- Owns the first NER stage: `ContextChunk -> CandidateQuadruplet[]`.
- Owns conversion of model evidence payloads into canonical `SpanRef` objects.
- Does not group candidates, finalize outputs, or write to storage.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import AgentResult, BaseAgent
from .shared import AgentExecutionConfig, PromptExecutor, StructuredOutputParser, render_prompt
from ..runtime.llm_client import EmptyCompletionError, RefusalError
from ..schemas.artifacts import CandidateQuadruplet, ContextChunk, SpanRef
from ..schemas.constants import stable_int_id
from ..schemas.validation import (
    SchemaValidationError,
    validate_candidate_quadruplet,
    validate_context_chunk,
)

logger = logging.getLogger(__name__)

_FALLBACK_PROMPT_TEMPLATE = """role: you are a strict JSON annotator for AI policy legislation.

task:
Return exactly one compact valid JSON object with the top-level key "candidates"
for the given chunk. If the chunk does not support a reliable candidate, return
{{"candidates": []}}.

requirements:
- Return at most 1 candidate.
- Keep every field present in every candidate.
- Keep at most 1 evidence span per field.
- Keep each evidence text span under 120 characters. Extract only the key clause.
- Every evidence start/end must be integer character offsets relative to the
  chunk text and must be between 0 and {chunk_length}.
- If exact offsets are uncertain, return an empty evidence array for that field.
- Never use a bill id or chunk id as an offset.
- Return compact valid JSON on one line only. Do not pretty-print, pad with
  spaces, or add commentary.

Bill id: {bill_id}
Chunk text:
{text}"""


class ZeroShotAnnotator(BaseAgent[ContextChunk, list[CandidateQuadruplet]]):
    """Extract candidate quadruplets from a single schema-valid context chunk.

    This agent is the first stage of the NER pipeline. It reads one
    ``ContextChunk`` at a time and produces zero-shot candidate quadruplets plus
    field-linked evidence spans.
    """

    def __init__(
        self,
        prompt_template: str,
        output_schema: dict[str, Any],
        execution_config: AgentExecutionConfig,
        prompt_executor: PromptExecutor,
    ) -> None:
        """Initialize the zero-shot annotator.

        Args:
            prompt_template (str): Prompt template used to query the backing
                LLM.
            output_schema (dict[str, Any]): JSON schema that constrains the
                structured response.
            execution_config (AgentExecutionConfig): Generation settings for
                this agent.
            prompt_executor (PromptExecutor): Shared executor used to send
                prompts to the LLM.
        """

        self._prompt_template = prompt_template
        self._output_schema = output_schema
        self._execution_config = execution_config
        self._prompt_executor = prompt_executor

    @property
    def name(self) -> str:
        """Return the stable config name for this agent.

        Returns:
            str: Agent config key and logger name.
        """

        return "zero_shot_annotator"

    @property
    def input_schema_name(self) -> str:
        """Return the explicit input schema name.

        Returns:
            str: Name of the chunk-level input contract.
        """

        return "ContextChunk"

    @property
    def output_schema_name(self) -> str:
        """Return the explicit output schema name.

        Returns:
            str: Name of the zero-shot candidate output contract.
        """

        return "list[CandidateQuadruplet]"

    def run(self, input_data: ContextChunk) -> AgentResult[list[CandidateQuadruplet]]:
        """Execute chunk-level candidate extraction against the shared local LLM.

        Args:
            input_data (ContextChunk): Schema-valid context chunk to annotate.

        Returns:
            AgentResult[list[CandidateQuadruplet]]: Agent result containing the
                raw model response plus parsed candidate quadruplets.

        Raises:
            SchemaValidationError: If the input chunk is invalid or the model
                response does not satisfy the declared candidate schema.
        """

        if not isinstance(input_data, ContextChunk):
            raise SchemaValidationError(
                "ZeroShotAnnotator input must be a schema-valid ContextChunk"
            )
        validate_context_chunk(input_data)

        try:
            raw_response = self._execute_primary_prompt(input_data)
            parsed_candidates = self._parse_candidates(input_data, raw_response)
        except RefusalError as exc:
            logger.warning(
                "ZeroShotAnnotator upstream refusal on primary prompt  "
                "bill_id=%s  chunk_id=%d  provider=%s  prompt_tokens=%d  "
                "action=retry_with_fallback_prompt",
                input_data.bill_id,
                input_data.chunk_id,
                exc.provider,
                exc.prompt_tokens,
            )
            parsed_candidates, raw_response = self._annotate_via_fallback_only(
                input_data,
                primary_detail=f"refusal provider={exc.provider}",
            )
        except (SchemaValidationError, EmptyCompletionError) as primary_exc:
            parsed_candidates, raw_response = self._annotate_via_fallback_only(
                input_data,
                primary_detail=str(primary_exc),
            )

        return AgentResult(
            input_schema_name=self.input_schema_name,
            output_schema_name=self.output_schema_name,
            raw_response=raw_response,
            parsed_response=parsed_candidates,
        )

    def _annotate_via_fallback_only(
        self,
        input_data: ContextChunk,
        *,
        primary_detail: str,
    ) -> tuple[list[CandidateQuadruplet], str]:
        """Try the fallback prompt once; raise ``SchemaValidationError`` on failure.

        The primary extraction path has already failed (schema error, empty
        content, or upstream refusal).  This helper runs the stricter
        fallback prompt once.  If the fallback also fails for any model-side
        reason (refusal, empty, schema error), it re-raises as
        ``SchemaValidationError`` so the orchestrator's existing per-chunk
        failure handler can drop this chunk without aborting the bill.

        Args:
            input_data: Chunk the primary prompt could not annotate.
            primary_detail: Human-readable description of the primary failure
                for inclusion in the combined error message.

        Returns:
            Tuple ``(parsed_candidates, raw_response)`` from the fallback.

        Raises:
            SchemaValidationError: If the fallback also fails.
        """

        try:
            fallback_raw_response = self._execute_fallback_prompt(input_data)
            parsed_candidates = self._parse_candidates(
                input_data, fallback_raw_response
            )
            return parsed_candidates, fallback_raw_response
        except RefusalError as fallback_exc:
            logger.warning(
                "ZeroShotAnnotator upstream refusal on fallback prompt  "
                "bill_id=%s  chunk_id=%d  provider=%s  "
                "action=drop_chunk",
                input_data.bill_id,
                input_data.chunk_id,
                fallback_exc.provider,
            )
            raise SchemaValidationError(
                "ZeroShotAnnotator failed primary "
                f"({primary_detail}) and fallback refused "
                f"(provider={fallback_exc.provider})"
            ) from fallback_exc
        except (SchemaValidationError, EmptyCompletionError) as fallback_exc:
            raise SchemaValidationError(
                "ZeroShotAnnotator failed primary structured extraction "
                f"({primary_detail}) and fallback structured extraction "
                f"({fallback_exc})"
            ) from fallback_exc

    def _execute_primary_prompt(self, input_data: ContextChunk) -> str:
        """Execute the primary zero-shot extraction prompt.

        Args:
            input_data (ContextChunk): Schema-valid context chunk to annotate.

        Returns:
            str: Raw structured response text returned by the LLM.
        """

        prompt = render_prompt(
            self._prompt_template,
            bill_id=input_data.bill_id,
            chunk_id=input_data.chunk_id,
            text=input_data.text,
        )
        return self._prompt_executor.execute(
            prompt=prompt,
            output_schema=self._output_schema,
            execution_config=self._execution_config,
        )

    def _execute_fallback_prompt(self, input_data: ContextChunk) -> str:
        """Execute a stricter fallback prompt after malformed primary output.

        The fallback prompt narrows the output surface to one candidate and
        emphasizes valid relative offsets so that a single malformed primary
        response does not abort the whole bill run immediately.

        Args:
            input_data (ContextChunk): Schema-valid context chunk to annotate.

        Returns:
            str: Raw structured response text returned by the LLM.
        """

        fallback_execution_config = AgentExecutionConfig(
            temperature=self._execution_config.temperature,
            max_tokens=min(self._execution_config.max_tokens, 768),
        )
        prompt = render_prompt(
            _FALLBACK_PROMPT_TEMPLATE,
            bill_id=input_data.bill_id,
            chunk_length=len(input_data.text),
            text=input_data.text,
        )
        return self._prompt_executor.execute(
            prompt=prompt,
            output_schema=self._output_schema,
            execution_config=fallback_execution_config,
        )

    def _parse_candidates(
        self,
        chunk: ContextChunk,
        raw_response: str,
    ) -> list[CandidateQuadruplet]:
        """Parse and validate raw model output into canonical candidates.

        Args:
            chunk (ContextChunk): Source chunk that produced the raw response.
            raw_response (str): Raw response text returned by the LLM.

        Returns:
            list[CandidateQuadruplet]: Parsed validated candidate quadruplets.

        Raises:
            SchemaValidationError: If the raw response does not satisfy the
                declared candidate schema.
        """

        parsed_payload = StructuredOutputParser.parse_object(raw_response)
        raw_candidates = parsed_payload.get("candidates")
        if not isinstance(raw_candidates, list):
            raise SchemaValidationError(
                "ZeroShotAnnotator output must contain a candidates list"
            )

        parsed_candidates = [
            self._build_candidate(chunk, raw_candidate, index)
            for index, raw_candidate in enumerate(raw_candidates)
        ]
        for candidate in parsed_candidates:
            validate_candidate_quadruplet(candidate)
        return parsed_candidates

    def _build_candidate(
        self,
        chunk: ContextChunk,
        raw_candidate: dict[str, Any],
        candidate_index: int,
    ) -> CandidateQuadruplet:
        """Convert one raw candidate payload into a typed candidate object.

        Args:
            chunk (ContextChunk): Source context chunk that produced the
                candidate.
            raw_candidate (dict[str, Any]): JSON object decoded from the model
                response.
            candidate_index (int): Stable position of the candidate inside the
                chunk response.

        Returns:
            CandidateQuadruplet: Parsed candidate quadruplet with deterministic
                ids and evidence.

        Raises:
            SchemaValidationError: If the raw payload is not a JSON object.
        """

        if not isinstance(raw_candidate, dict):
            raise SchemaValidationError("Each candidate payload must be a JSON object")

        candidate_id = stable_int_id(
            "candidate",
            chunk.bill_id,
            chunk.chunk_id,
            candidate_index,
        )
        return CandidateQuadruplet(
            candidate_id=candidate_id,
            entity=_normalize_optional_text(raw_candidate.get("entity")),
            type=_normalize_optional_text(raw_candidate.get("type")),
            attribute=_normalize_optional_text(raw_candidate.get("attribute")),
            value=_normalize_optional_text(raw_candidate.get("value")),
            entity_evidence=self._build_spans(
                chunk=chunk,
                candidate_id=candidate_id,
                field_name="entity",
                raw_spans=raw_candidate.get("entity_evidence", []),
            ),
            type_evidence=self._build_spans(
                chunk=chunk,
                candidate_id=candidate_id,
                field_name="type",
                raw_spans=raw_candidate.get("type_evidence", []),
            ),
            attribute_evidence=self._build_spans(
                chunk=chunk,
                candidate_id=candidate_id,
                field_name="attribute",
                raw_spans=raw_candidate.get("attribute_evidence", []),
            ),
            value_evidence=self._build_spans(
                chunk=chunk,
                candidate_id=candidate_id,
                field_name="value",
                raw_spans=raw_candidate.get("value_evidence", []),
            ),
        )

    def _build_spans(
        self,
        chunk: ContextChunk,
        candidate_id: int,
        field_name: str,
        raw_spans: Any,
    ) -> list[SpanRef]:
        """Convert one field's raw evidence payload into typed span references.

        Args:
            chunk (ContextChunk): Source chunk from which evidence offsets are
                measured.
            candidate_id (int): Deterministic candidate id used when deriving
                span ids.
            field_name (str): Canonical field name that owns the evidence list.
            raw_spans (Any): Raw JSON value returned for the field evidence.

        Returns:
            list[SpanRef]: Parsed list of ``SpanRef`` objects aligned to source
                offsets. Structurally valid spans with malformed offsets are
                deterministically repaired when possible; otherwise they are
                dropped.

        Raises:
            SchemaValidationError: If the raw evidence payload is not a valid
                list of evidence objects with bounded offsets.
        """

        if not isinstance(raw_spans, list):
            raise SchemaValidationError(f"{field_name}_evidence must be a list")

        parsed_spans: list[SpanRef] = []
        for span_index, raw_span in enumerate(raw_spans):
            parsed_span = self._build_span(
                chunk=chunk,
                candidate_id=candidate_id,
                field_name=field_name,
                span_index=span_index,
                raw_span=raw_span,
            )
            if parsed_span is not None:
                parsed_spans.append(parsed_span)
        return parsed_spans

    def _build_span(
        self,
        chunk: ContextChunk,
        candidate_id: int,
        field_name: str,
        span_index: int,
        raw_span: Any,
    ) -> SpanRef | None:
        """Build one evidence span, repairing malformed offsets when possible."""

        if not isinstance(raw_span, dict):
            raise SchemaValidationError(f"{field_name}_evidence entries must be objects")
        if "start" not in raw_span or "end" not in raw_span or "text" not in raw_span:
            raise SchemaValidationError(
                f"{field_name}_evidence entries must include start, end, and text"
            )

        relative_start = raw_span["start"]
        relative_end = raw_span["end"]
        span_text = raw_span["text"]
        if not isinstance(relative_start, int):
            raise SchemaValidationError(f"{field_name}_evidence.start must be an integer")
        if not isinstance(relative_end, int):
            raise SchemaValidationError(f"{field_name}_evidence.end must be an integer")
        if not isinstance(span_text, str):
            raise SchemaValidationError(f"{field_name}_evidence.text must be a string")

        resolved_span = self._resolve_relative_span(
            chunk_text=chunk.text,
            relative_start=relative_start,
            relative_end=relative_end,
            span_text=span_text,
        )
        if resolved_span is None:
            return None

        repaired_start, repaired_end, repaired_text = resolved_span
        source_start = chunk.start_offset + repaired_start
        source_end = chunk.start_offset + repaired_end
        return SpanRef(
            span_id=stable_int_id(
                "span",
                candidate_id,
                field_name,
                span_index,
                source_start,
                source_end,
                repaired_text,
            ),
            start=source_start,
            end=source_end,
            text=repaired_text,
            chunk_id=chunk.chunk_id,
        )

    def _resolve_relative_span(
        self,
        chunk_text: str,
        relative_start: int,
        relative_end: int,
        span_text: str,
    ) -> tuple[int, int, str] | None:
        """Resolve one span against the chunk text using offsets and exact text."""

        if (
            0 <= relative_start <= relative_end <= len(chunk_text)
            and chunk_text[relative_start:relative_end] == span_text
        ):
            return relative_start, relative_end, chunk_text[relative_start:relative_end]

        matches = self._find_exact_text_matches(chunk_text, span_text)
        if not matches:
            return None

        anchor_start = relative_start if relative_start >= 0 else 0
        anchor_end = relative_end if relative_end >= relative_start else anchor_start + len(span_text)
        best_start, best_end = min(
            matches,
            key=lambda match: (
                abs(match[0] - anchor_start),
                abs(match[1] - anchor_end),
                match[0],
            ),
        )
        return best_start, best_end, chunk_text[best_start:best_end]

    def _find_exact_text_matches(
        self,
        chunk_text: str,
        span_text: str,
    ) -> list[tuple[int, int]]:
        """Return all exact occurrences of span text inside the chunk."""

        if not span_text:
            return []

        matches: list[tuple[int, int]] = []
        search_start = 0
        while True:
            match_start = chunk_text.find(span_text, search_start)
            if match_start == -1:
                return matches
            matches.append((match_start, match_start + len(span_text)))
            search_start = match_start + 1


def _normalize_optional_text(value: Any) -> str | None:
    """Normalize optional text fields into the canonical nullable form.

    Args:
        value (Any): Raw field value decoded from the model response.

    Returns:
        str | None: Trimmed string value, or ``None`` when the raw value is
            null or blank.

    Raises:
        SchemaValidationError: If the value is neither a string nor ``None``.
    """

    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaValidationError("Field values must be strings or null")
    normalized = value.strip()
    return normalized or None

