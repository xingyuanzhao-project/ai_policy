"""Zero-shot Annotator for chunk-level candidate extraction.

- Owns the first NER stage: `ContextChunk -> CandidateQuadruplet[]`.
- Owns conversion of model evidence payloads into canonical `SpanRef` objects.
- Does not group candidates, finalize outputs, or write to storage.
"""

from __future__ import annotations

from typing import Any

from .base import AgentResult, BaseAgent
from .shared import AgentExecutionConfig, PromptExecutor, StructuredOutputParser, render_prompt
from ..schemas.artifacts import CandidateQuadruplet, ContextChunk, SpanRef
from ..schemas.constants import stable_int_id
from ..schemas.validation import (
    SchemaValidationError,
    validate_candidate_quadruplet,
    validate_context_chunk,
)

_FALLBACK_PROMPT_TEMPLATE = """role: you are a strict JSON annotator for AI policy legislation.

task:
Return exactly one compact valid JSON object with the top-level key "candidates"
for the given chunk. If the chunk does not support a reliable candidate, return
{{"candidates": []}}.

requirements:
- Return at most 1 candidate.
- Keep every field present in every candidate.
- Keep at most 1 evidence span per field.
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

        raw_response = self._execute_primary_prompt(input_data)
        try:
            parsed_candidates = self._parse_candidates(input_data, raw_response)
        except SchemaValidationError as primary_exc:
            fallback_raw_response = self._execute_fallback_prompt(input_data)
            try:
                parsed_candidates = self._parse_candidates(input_data, fallback_raw_response)
            except SchemaValidationError as fallback_exc:
                raise SchemaValidationError(
                    "ZeroShotAnnotator failed primary structured extraction "
                    f"({primary_exc}) and fallback structured extraction "
                    f"({fallback_exc})"
                ) from fallback_exc
            raw_response = fallback_raw_response

        return AgentResult(
            input_schema_name=self.input_schema_name,
            output_schema_name=self.output_schema_name,
            raw_response=raw_response,
            parsed_response=parsed_candidates,
        )

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
            max_tokens=min(self._execution_config.max_tokens, 512),
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
                offsets.

        Raises:
            SchemaValidationError: If the raw evidence payload is not a valid
                list of evidence objects with bounded offsets.
        """

        if not isinstance(raw_spans, list):
            raise SchemaValidationError(f"{field_name}_evidence must be a list")

        parsed_spans: list[SpanRef] = []
        for span_index, raw_span in enumerate(raw_spans):
            if not isinstance(raw_span, dict):
                raise SchemaValidationError(f"{field_name}_evidence entries must be objects")
            if "start" not in raw_span or "end" not in raw_span or "text" not in raw_span:
                raise SchemaValidationError(
                    f"{field_name}_evidence entries must include start, end, and text"
                )

            relative_start = raw_span["start"]
            relative_end = raw_span["end"]
            span_text = raw_span["text"]
            if not isinstance(relative_start, int) or relative_start < 0:
                raise SchemaValidationError(f"{field_name}_evidence.start must be >= 0")
            if not isinstance(relative_end, int) or relative_end < relative_start:
                raise SchemaValidationError(
                    f"{field_name}_evidence.end must be >= {field_name}_evidence.start"
                )
            if relative_end > len(chunk.text):
                raise SchemaValidationError(
                    f"{field_name}_evidence.end exceeds the chunk length"
                )
            if not isinstance(span_text, str):
                raise SchemaValidationError(f"{field_name}_evidence.text must be a string")

            source_start = chunk.start_offset + relative_start
            source_end = chunk.start_offset + relative_end
            parsed_spans.append(
                SpanRef(
                    span_id=stable_int_id(
                        "span",
                        candidate_id,
                        field_name,
                        span_index,
                        source_start,
                        source_end,
                        span_text,
                    ),
                    start=source_start,
                    end=source_end,
                    text=span_text,
                    chunk_id=chunk.chunk_id,
                )
            )
        return parsed_spans


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

