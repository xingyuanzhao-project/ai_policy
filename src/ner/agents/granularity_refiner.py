"""Granularity Refiner for grouped-candidate refinement.

- Owns the third NER stage: grouped-set refinement into final outputs.
- Owns conversion of model refinement payloads into `RefinedQuadruplet` and
  optional `RefinementArtifact`.
- Does not rerun annotation, regroup candidates, or persist artifacts itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .base import AgentResult, BaseAgent
from .shared import (
    AgentExecutionConfig,
    PromptExecutor,
    StructuredOutputParser,
    render_prompt,
    serialize_for_prompt,
)
from ..runtime.llm_client import EmptyCompletionError, RefusalError
from ..schemas.artifacts import (
    CandidateQuadruplet,
    GroupedCandidateSet,
    RefinedQuadruplet,
    RefinementArtifact,
    SpanRef,
)
from ..schemas.constants import stable_int_id
from ..schemas.validation import (
    SchemaValidationError,
    validate_candidate_quadruplet,
    validate_grouped_candidate_set,
    validate_refined_quadruplet,
    validate_refinement_artifact,
    validate_span_ref,
)

logger = logging.getLogger(__name__)

_FALLBACK_REFINEMENT_PROMPT_TEMPLATE = """role: you are a strict JSON refiner for one grouped candidate set.

task:
Return exactly one compact valid JSON object with keys "refined_quadruplets"
and "refinement_artifact". If unsure, keep the single best-supported candidate
as the only refined quadruplet and set refinement_artifact to null.

requirements:
- Return exactly one refined_quadruplet.
- source_candidate_ids must come from the grouped candidate set.
- Preserve at most 1 evidence span per field.
- Set refinement_artifact to null.
- Return compact valid JSON on one line only. Do not pretty-print, pad with
  spaces, or add commentary.

Grouped candidate set:
{grouped_candidate_set_json}

Referenced candidates:
{candidate_pool_json}"""


@dataclass(slots=True)
class RefinementRequest:
    """Typed input contract for refining one grouped candidate set.

    Attributes:
        grouped_candidate_set (GroupedCandidateSet): Group-level structure
            selected for refinement.
        candidate_pool_by_id (dict[int, CandidateQuadruplet]): Candidate lookup
            table keyed by ``candidate_id``.
    """

    grouped_candidate_set: GroupedCandidateSet
    candidate_pool_by_id: dict[int, CandidateQuadruplet]


class GranularityRefiner(
    BaseAgent[RefinementRequest, tuple[list[RefinedQuadruplet], RefinementArtifact | None]]
):
    """Refine one grouped candidate set into final structured outputs.

    This agent is the third NER stage. It reads a grouped candidate set plus the
    referenced candidate pool and emits final refined quadruplets together with
    an optional refinement-side artifact.
    """

    def __init__(
        self,
        prompt_template: str,
        output_schema: dict[str, Any],
        execution_config: AgentExecutionConfig,
        prompt_executor: PromptExecutor,
    ) -> None:
        """Initialize the granularity refiner.

        Args:
            prompt_template (str): Prompt template used to query the backing
                LLM.
            output_schema (dict[str, Any]): JSON schema constraining refinement
                output.
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

        return "granularity_refiner"

    @property
    def input_schema_name(self) -> str:
        """Return the explicit input schema name.

        Returns:
            str: Name of the grouped-set refinement request contract.
        """

        return "RefinementRequest"

    @property
    def output_schema_name(self) -> str:
        """Return the explicit output schema name.

        Returns:
            str: Name of the refinement output contract.
        """

        return "tuple[list[RefinedQuadruplet], RefinementArtifact | None]"

    def run(
        self,
        input_data: RefinementRequest,
    ) -> AgentResult[tuple[list[RefinedQuadruplet], RefinementArtifact | None]]:
        """Execute refinement for one grouped candidate set against the shared LLM.

        Args:
            input_data (RefinementRequest): Refinement request containing one
                grouped set plus its referenced candidate pool.

        Returns:
            AgentResult[tuple[list[RefinedQuadruplet], RefinementArtifact | None]]:
                Agent result containing the raw model response plus parsed
                refined quadruplets and an optional refinement artifact.

        Raises:
            SchemaValidationError: If the refinement request itself is
                malformed (bad input contract).  Model-side failures
                (invalid JSON, empty content, upstream refusal) are
                recovered internally via the fallback-prompt and
                deterministic-passthrough tiers; they do not escape this
                method.
        """

        if not isinstance(input_data, RefinementRequest):
            raise SchemaValidationError(
                "GranularityRefiner input must be a RefinementRequest"
            )

        grouped_set = input_data.grouped_candidate_set
        candidate_pool_by_id = input_data.candidate_pool_by_id
        validate_grouped_candidate_set(grouped_set)
        if not isinstance(candidate_pool_by_id, dict):
            raise SchemaValidationError("candidate_pool_by_id must be a dict")

        referenced_candidates: list[CandidateQuadruplet] = []
        for candidate_id in grouped_set.candidate_ids:
            if candidate_id not in candidate_pool_by_id:
                raise SchemaValidationError(
                    f"GranularityRefiner missing candidate_id {candidate_id} in pool"
                )
            candidate = candidate_pool_by_id[candidate_id]
            validate_candidate_quadruplet(candidate)
            referenced_candidates.append(candidate)

        try:
            raw_response = self._execute_primary_prompt(grouped_set, referenced_candidates)
            refined_outputs, refinement_artifact = self._parse_refinement_response(
                grouped_set,
                raw_response,
            )
        except RefusalError as exc:
            # Upstream model refused the primary prompt.  Refusal is
            # deterministic at temperature 0 for the same prompt + schema +
            # content, so the fallback prompt (same content, different
            # wording) is unlikely to unstick it and not worth the extra
            # call.  Route directly to deterministic passthrough.
            self._log_refusal(
                grouped_set,
                referenced_candidates,
                exc,
                stage="primary_prompt",
            )
            refined_outputs = [
                self._build_deterministic_fallback_refined_output(
                    grouped_set,
                    referenced_candidates,
                )
            ]
            refinement_artifact = None
            raw_response = ""
        except (SchemaValidationError, EmptyCompletionError):
            # Parse failed or model returned empty content without an
            # explicit refusal flag.  Try the stricter fallback prompt
            # once, then fall through to deterministic passthrough.
            fallback_raw_response = ""
            try:
                fallback_raw_response = self._execute_fallback_prompt(
                    grouped_set,
                    referenced_candidates,
                )
                refined_outputs, refinement_artifact = self._parse_refinement_response(
                    grouped_set,
                    fallback_raw_response,
                )
                raw_response = fallback_raw_response
            except RefusalError as exc:
                self._log_refusal(
                    grouped_set,
                    referenced_candidates,
                    exc,
                    stage="fallback_prompt",
                )
                refined_outputs = [
                    self._build_deterministic_fallback_refined_output(
                        grouped_set,
                        referenced_candidates,
                    )
                ]
                refinement_artifact = None
                raw_response = fallback_raw_response
            except (SchemaValidationError, EmptyCompletionError):
                logger.warning(
                    "GranularityRefiner deterministic passthrough after both "
                    "prompt attempts failed  group_id=%d  candidate_ids=%s",
                    grouped_set.group_id,
                    grouped_set.candidate_ids,
                )
                refined_outputs = [
                    self._build_deterministic_fallback_refined_output(
                        grouped_set,
                        referenced_candidates,
                    )
                ]
                refinement_artifact = None
                raw_response = fallback_raw_response

        return AgentResult(
            input_schema_name=self.input_schema_name,
            output_schema_name=self.output_schema_name,
            raw_response=raw_response,
            parsed_response=(refined_outputs, refinement_artifact),
        )

    def _log_refusal(
        self,
        grouped_set: GroupedCandidateSet,
        referenced_candidates: list[CandidateQuadruplet],
        exc: RefusalError,
        *,
        stage: str,
    ) -> None:
        """Log an upstream refusal with full stage and group context.

        Args:
            grouped_set: Grouped candidate set being refined.
            referenced_candidates: Candidate pool referenced by the group.
            exc: The raised ``RefusalError`` carrying provider and token counts.
            stage: Which prompt attempt refused (``"primary_prompt"`` or
                ``"fallback_prompt"``), so operators can tell which attempt
                each refusal log line belongs to.
        """

        entity_preview = (
            referenced_candidates[0].entity if referenced_candidates else None
        )
        logger.warning(
            "GranularityRefiner upstream refusal  stage=%s  group_id=%d  "
            "candidate_ids=%s  provider=%s  prompt_tokens=%d  "
            "completion_tokens=%d  entity_preview=%r  "
            "action=deterministic_passthrough",
            stage,
            grouped_set.group_id,
            grouped_set.candidate_ids,
            exc.provider,
            exc.prompt_tokens,
            exc.completion_tokens,
            entity_preview,
        )

    def _execute_primary_prompt(
        self,
        grouped_set: GroupedCandidateSet,
        referenced_candidates: list[CandidateQuadruplet],
    ) -> str:
        """Execute the primary refinement prompt.

        Args:
            grouped_set (GroupedCandidateSet): Group currently being refined.
            referenced_candidates (list[CandidateQuadruplet]): Candidate pool
                rows referenced by the group.

        Returns:
            str: Raw structured response text returned by the LLM.
        """

        prompt = render_prompt(
            self._prompt_template,
            grouped_candidate_set_json=serialize_for_prompt(grouped_set),
            candidate_pool_json=serialize_for_prompt(referenced_candidates),
        )
        return self._prompt_executor.execute(
            prompt=prompt,
            output_schema=self._output_schema,
            execution_config=self._execution_config,
        )

    def _execute_fallback_prompt(
        self,
        grouped_set: GroupedCandidateSet,
        referenced_candidates: list[CandidateQuadruplet],
    ) -> str:
        """Execute a stricter fallback refinement prompt after malformed output.

        Args:
            grouped_set (GroupedCandidateSet): Group currently being refined.
            referenced_candidates (list[CandidateQuadruplet]): Candidate pool
                rows referenced by the group.

        Returns:
            str: Raw structured response text returned by the LLM.
        """

        prompt = render_prompt(
            _FALLBACK_REFINEMENT_PROMPT_TEMPLATE,
            grouped_candidate_set_json=serialize_for_prompt(grouped_set),
            candidate_pool_json=serialize_for_prompt(referenced_candidates),
        )
        return self._prompt_executor.execute(
            prompt=prompt,
            output_schema=self._output_schema,
            execution_config=self._execution_config,
        )

    def _parse_refinement_response(
        self,
        grouped_set: GroupedCandidateSet,
        raw_response: str,
    ) -> tuple[list[RefinedQuadruplet], RefinementArtifact | None]:
        """Parse and validate raw refinement output.

        Args:
            grouped_set (GroupedCandidateSet): Group currently being refined.
            raw_response (str): Raw structured response text returned by the LLM.

        Returns:
            tuple[list[RefinedQuadruplet], RefinementArtifact | None]:
                Validated refined outputs and optional artifact.

        Raises:
            SchemaValidationError: If the raw response violates the declared
                refinement contracts.
        """

        parsed_payload = StructuredOutputParser.parse_object(raw_response)

        raw_refined_outputs = parsed_payload.get("refined_quadruplets")
        if not isinstance(raw_refined_outputs, list):
            raise SchemaValidationError(
                "GranularityRefiner output must contain refined_quadruplets"
            )

        refined_outputs = [
            self._build_refined_output(grouped_set, raw_quadruplet, output_index)
            for output_index, raw_quadruplet in enumerate(raw_refined_outputs)
        ]
        for refined_output in refined_outputs:
            if not set(refined_output.source_candidate_ids).issubset(grouped_set.candidate_ids):
                raise SchemaValidationError(
                    "Refined output source_candidate_ids must come from the grouped set"
                )
            validate_refined_quadruplet(refined_output)

        raw_refinement_artifact = parsed_payload.get("refinement_artifact")
        refinement_artifact = None
        if raw_refinement_artifact is not None:
            try:
                refinement_artifact = self._build_refinement_artifact(
                    grouped_set,
                    raw_refinement_artifact,
                )
                validate_refinement_artifact(refinement_artifact)
            except SchemaValidationError:
                refinement_artifact = None

        return refined_outputs, refinement_artifact

    def _build_deterministic_fallback_refined_output(
        self,
        grouped_set: GroupedCandidateSet,
        referenced_candidates: list[CandidateQuadruplet],
    ) -> RefinedQuadruplet:
        """Build a deterministic refined output when both LLM attempts fail.

        The fallback selects the candidate with the strongest grouped-set score
        row and lifts its canonical fields and evidence directly into the final
        output so the pipeline can complete without fabricating new content.

        Args:
            grouped_set (GroupedCandidateSet): Group currently being refined.
            referenced_candidates (list[CandidateQuadruplet]): Candidate pool
                rows referenced by the group, aligned to ``candidate_ids`` order.

        Returns:
            RefinedQuadruplet: Deterministic refined output derived from the best
                candidate in the group.
        """

        best_index = max(
            range(len(referenced_candidates)),
            key=lambda index: (
                sum(
                    score
                    for score in grouped_set.field_score_matrix[index]
                    if score is not None
                ),
                -index,
            ),
        )
        best_candidate = referenced_candidates[best_index]
        refined_output = RefinedQuadruplet(
            refined_id=stable_int_id(
                "refined",
                grouped_set.group_id,
                0,
                best_candidate.candidate_id,
            ),
            source_group_id=grouped_set.group_id,
            source_candidate_ids=[best_candidate.candidate_id],
            entity=best_candidate.entity,
            type=best_candidate.type,
            attribute=best_candidate.attribute,
            value=best_candidate.value,
            entity_evidence=list(best_candidate.entity_evidence),
            type_evidence=list(best_candidate.type_evidence),
            attribute_evidence=list(best_candidate.attribute_evidence),
            value_evidence=list(best_candidate.value_evidence),
        )
        validate_refined_quadruplet(refined_output)
        return refined_output

    def _build_refined_output(
        self,
        grouped_set: GroupedCandidateSet,
        raw_quadruplet: dict[str, Any],
        output_index: int,
    ) -> RefinedQuadruplet:
        """Convert one raw refined quadruplet payload into a typed object.

        Args:
            grouped_set (GroupedCandidateSet): Group currently being refined.
            raw_quadruplet (dict[str, Any]): JSON object decoded from the model
                response.
            output_index (int): Stable position of the refined output inside the
                response payload.

        Returns:
            RefinedQuadruplet: Parsed refined quadruplet with deterministic ids.

        Raises:
            SchemaValidationError: If the raw payload is malformed.
        """

        if not isinstance(raw_quadruplet, dict):
            raise SchemaValidationError(
                "Each refined quadruplet payload must be a JSON object"
            )
        source_candidate_ids = raw_quadruplet.get("source_candidate_ids")
        if not isinstance(source_candidate_ids, list) or not source_candidate_ids:
            raise SchemaValidationError(
                "Each refined quadruplet must include non-empty source_candidate_ids"
            )

        return RefinedQuadruplet(
            refined_id=stable_int_id(
                "refined",
                grouped_set.group_id,
                output_index,
                *sorted(source_candidate_ids),
            ),
            source_group_id=grouped_set.group_id,
            source_candidate_ids=[int(candidate_id) for candidate_id in source_candidate_ids],
            entity=_normalize_optional_text(raw_quadruplet.get("entity")),
            type=_normalize_optional_text(raw_quadruplet.get("type")),
            attribute=_normalize_optional_text(raw_quadruplet.get("attribute")),
            value=_normalize_optional_text(raw_quadruplet.get("value")),
            entity_evidence=self._build_spans(
                grouped_set.group_id,
                "entity",
                raw_quadruplet.get("entity_evidence", []),
            ),
            type_evidence=self._build_spans(
                grouped_set.group_id,
                "type",
                raw_quadruplet.get("type_evidence", []),
            ),
            attribute_evidence=self._build_spans(
                grouped_set.group_id,
                "attribute",
                raw_quadruplet.get("attribute_evidence", []),
            ),
            value_evidence=self._build_spans(
                grouped_set.group_id,
                "value",
                raw_quadruplet.get("value_evidence", []),
            ),
        )

    def _build_refinement_artifact(
        self,
        grouped_set: GroupedCandidateSet,
        raw_artifact: dict[str, Any],
    ) -> RefinementArtifact:
        """Convert one raw refinement artifact payload into a typed object.

        Args:
            grouped_set (GroupedCandidateSet): Group currently being refined.
            raw_artifact (dict[str, Any]): JSON object decoded from the model
                response.

        Returns:
            RefinementArtifact: Parsed refinement artifact aligned to the
                grouped candidate ids.

        Raises:
            SchemaValidationError: If the payload is malformed or the candidate
                id order diverges from the grouped set.
        """

        if not isinstance(raw_artifact, dict):
            raise SchemaValidationError("refinement_artifact must be an object or null")
        candidate_ids = raw_artifact.get("candidate_ids")
        if candidate_ids != grouped_set.candidate_ids:
            raise SchemaValidationError(
                "refinement_artifact candidate_ids must exactly match the grouped set order"
            )
        return RefinementArtifact(
            group_id=grouped_set.group_id,
            candidate_ids=[int(candidate_id) for candidate_id in candidate_ids],
            entity_relations=raw_artifact.get("entity_relations", []),
            type_relations=raw_artifact.get("type_relations", []),
            attribute_relations=raw_artifact.get("attribute_relations", []),
            value_relations=raw_artifact.get("value_relations", []),
        )

    def _build_spans(
        self,
        group_id: int,
        field_name: str,
        raw_spans: Any,
    ) -> list[SpanRef]:
        """Convert one field's raw evidence payload into typed span references.

        Args:
            group_id (int): Group id owning the refined output.
            field_name (str): Canonical field name that owns the evidence list.
            raw_spans (Any): Raw JSON value returned for the field evidence.

        Returns:
            list[SpanRef]: Parsed list of ``SpanRef`` objects aligned to the
                returned source offsets.

        Raises:
            SchemaValidationError: If the raw evidence payload is not a valid
                list of evidence objects.
        """

        if not isinstance(raw_spans, list):
            raise SchemaValidationError(f"{field_name}_evidence must be a list")

        parsed_spans: list[SpanRef] = []
        for span_index, raw_span in enumerate(raw_spans):
            if not isinstance(raw_span, dict):
                raise SchemaValidationError(f"{field_name}_evidence entries must be objects")
            required_keys = {"start", "end", "text", "chunk_id"}
            if not required_keys.issubset(raw_span):
                raise SchemaValidationError(
                    f"{field_name}_evidence entries must include start, end, text, and chunk_id"
                )
            span = SpanRef(
                span_id=stable_int_id(
                    "refined_span",
                    group_id,
                    field_name,
                    span_index,
                    raw_span["start"],
                    raw_span["end"],
                    raw_span["text"],
                    raw_span["chunk_id"],
                ),
                start=raw_span["start"],
                end=raw_span["end"],
                text=raw_span["text"],
                chunk_id=raw_span["chunk_id"],
            )
            validate_span_ref(span)
            parsed_spans.append(span)
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

