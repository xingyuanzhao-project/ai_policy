"""Eval Assembler for bill-level candidate grouping and scoring.

- Owns the second NER stage: bill-level candidate grouping and field scoring.
- Owns conversion of model grouping payloads into canonical `GroupedCandidateSet`.
- Does not rerun annotation, refine outputs, or write to storage.
"""

from __future__ import annotations

import logging
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
from ..schemas.artifacts import CandidateQuadruplet, GroupedCandidateSet
from ..schemas.constants import CANONICAL_FIELD_ORDER, stable_int_id
from ..schemas.validation import (
    SchemaValidationError,
    validate_candidate_quadruplet,
    validate_grouped_candidate_set,
)

logger = logging.getLogger(__name__)

_MAX_CANDIDATES_FOR_LLM_GROUPING = 80


class EvalAssembler(BaseAgent[list[CandidateQuadruplet], list[GroupedCandidateSet]]):
    """Group related bill-level candidates and score them by canonical fields.

    This agent is the second NER stage. It reads one bill-level candidate pool
    and emits grouped candidate sets whose matrix rows and columns remain aligned
    to canonical identifiers and field order.
    """

    def __init__(
        self,
        prompt_template: str,
        output_schema: dict[str, Any],
        execution_config: AgentExecutionConfig,
        prompt_executor: PromptExecutor,
    ) -> None:
        """Initialize the eval assembler.

        Args:
            prompt_template (str): Prompt template used to query the backing
                LLM.
            output_schema (dict[str, Any]): JSON schema constraining grouped-set
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

        return "eval_assembler"

    @property
    def input_schema_name(self) -> str:
        """Return the explicit input schema name.

        Returns:
            str: Name of the bill-level candidate-pool contract.
        """

        return "list[CandidateQuadruplet]"

    @property
    def output_schema_name(self) -> str:
        """Return the explicit output schema name.

        Returns:
            str: Name of the grouped-candidate-set output contract.
        """

        return "list[GroupedCandidateSet]"

    def run(
        self,
        input_data: list[CandidateQuadruplet],
    ) -> AgentResult[list[GroupedCandidateSet]]:
        """Execute bill-level grouping and field scoring against the shared LLM.

        Args:
            input_data (list[CandidateQuadruplet]): Bill-level candidate pool
                to group and score.

        Returns:
            AgentResult[list[GroupedCandidateSet]]: Agent result containing the
                raw model response plus parsed grouped candidate sets.

        Raises:
            SchemaValidationError: If the input pool is malformed or the model
                response violates the declared grouped-set schema.
        """

        if not isinstance(input_data, list):
            raise SchemaValidationError(
                "EvalAssembler input must be a list of CandidateQuadruplet"
            )
        for candidate in input_data:
            if not isinstance(candidate, CandidateQuadruplet):
                raise SchemaValidationError(
                    "EvalAssembler input must contain only CandidateQuadruplet"
                )
            validate_candidate_quadruplet(candidate)

        if not input_data:
            return AgentResult(
                input_schema_name=self.input_schema_name,
                output_schema_name=self.output_schema_name,
                raw_response='{"groups": []}',
                parsed_response=[],
            )

        if len(input_data) > _MAX_CANDIDATES_FOR_LLM_GROUPING:
            return AgentResult(
                input_schema_name=self.input_schema_name,
                output_schema_name=self.output_schema_name,
                raw_response=(
                    '{"fallback":"deterministic_exact_match_grouping",'
                    f'"candidate_count":{len(input_data)}'
                    '}'
                ),
                parsed_response=self._build_fallback_groups(input_data),
            )

        prompt = render_prompt(
            self._prompt_template,
            candidates_json=serialize_for_prompt(input_data),
        )
        raw_response = ""
        try:
            raw_response = self._prompt_executor.execute(
                prompt=prompt,
                output_schema=self._output_schema,
                execution_config=self._execution_config,
            )
            parsed_groups = self._parse_groups(input_data, raw_response)
        except RefusalError as exc:
            logger.warning(
                "EvalAssembler upstream refusal  candidate_count=%d  "
                "provider=%s  prompt_tokens=%d  action=deterministic_fallback_groups",
                len(input_data),
                exc.provider,
                exc.prompt_tokens,
            )
            parsed_groups = self._build_fallback_groups(input_data)
        except (SchemaValidationError, EmptyCompletionError):
            parsed_groups = self._build_fallback_groups(input_data)

        return AgentResult(
            input_schema_name=self.input_schema_name,
            output_schema_name=self.output_schema_name,
            raw_response=raw_response,
            parsed_response=parsed_groups,
        )

    def _parse_groups(
        self,
        input_data: list[CandidateQuadruplet],
        raw_response: str,
    ) -> list[GroupedCandidateSet]:
        """Parse and validate raw model output into grouped candidate sets.

        Args:
            input_data (list[CandidateQuadruplet]): Bill-level candidate pool
                supplied to the assembler.
            raw_response (str): Raw structured response text returned by the LLM.

        Returns:
            list[GroupedCandidateSet]: Parsed validated grouped candidate sets.

        Raises:
            SchemaValidationError: If the raw response does not satisfy the
                declared grouped-set schema.
        """

        parsed_payload = StructuredOutputParser.parse_object(raw_response)
        raw_groups = parsed_payload.get("groups")
        if not isinstance(raw_groups, list):
            raise SchemaValidationError("EvalAssembler output must contain a groups list")

        candidate_id_set = {candidate.candidate_id for candidate in input_data}
        parsed_groups = [
            self._build_group(raw_group, candidate_id_set)
            for raw_group in raw_groups
        ]
        for grouped_set in parsed_groups:
            validate_grouped_candidate_set(grouped_set)
        return parsed_groups

    def _build_fallback_groups(
        self,
        input_data: list[CandidateQuadruplet],
    ) -> list[GroupedCandidateSet]:
        """Build deterministic fallback groups when LLM JSON is malformed.

        The fallback keeps the pipeline moving without inventing new entities. It
        groups exact canonical quadruplet matches together and assigns
        deterministic per-field confidence rows from field presence and evidence.

        Args:
            input_data (list[CandidateQuadruplet]): Bill-level candidate pool to
                group deterministically.

        Returns:
            list[GroupedCandidateSet]: Deterministic grouped candidate sets that
                preserve every candidate exactly once.
        """

        grouped_candidates: dict[tuple[str | None, str | None, str | None, str | None], list[CandidateQuadruplet]] = {}
        for candidate in input_data:
            signature = (
                _normalize_group_key(candidate.entity),
                _normalize_group_key(candidate.type),
                _normalize_group_key(candidate.attribute),
                _normalize_group_key(candidate.value),
            )
            grouped_candidates.setdefault(signature, []).append(candidate)

        fallback_groups: list[GroupedCandidateSet] = []
        for candidates_in_group in grouped_candidates.values():
            grouped_set = GroupedCandidateSet(
                group_id=stable_int_id(
                    "group",
                    *sorted(candidate.candidate_id for candidate in candidates_in_group),
                ),
                candidate_ids=[candidate.candidate_id for candidate in candidates_in_group],
                field_score_matrix=[
                    self._fallback_score_row(candidate)
                    for candidate in candidates_in_group
                ],
                field_order=CANONICAL_FIELD_ORDER,
            )
            validate_grouped_candidate_set(grouped_set)
            fallback_groups.append(grouped_set)
        return fallback_groups

    def _fallback_score_row(self, candidate: CandidateQuadruplet) -> list[float]:
        """Build a deterministic fallback score row for one candidate.

        Args:
            candidate (CandidateQuadruplet): Candidate whose field presence and
                evidence should be converted into fallback scores.

        Returns:
            list[float]: Four scores aligned to the canonical field order.
        """

        return [
            _fallback_field_score(candidate.entity, candidate.entity_evidence),
            _fallback_field_score(candidate.type, candidate.type_evidence),
            _fallback_field_score(candidate.attribute, candidate.attribute_evidence),
            _fallback_field_score(candidate.value, candidate.value_evidence),
        ]

    def _build_group(
        self,
        raw_group: dict[str, Any],
        candidate_id_set: set[int],
    ) -> GroupedCandidateSet:
        """Convert one raw group payload into a typed grouped candidate set.

        Args:
            raw_group (dict[str, Any]): JSON object decoded from the model
                response.
            candidate_id_set (set[int]): Valid candidate ids available in the
                bill-level pool.

        Returns:
            GroupedCandidateSet: Parsed grouped candidate set with deterministic
                group id.

        Raises:
            SchemaValidationError: If the raw payload is malformed or references
                a candidate id outside the current bill-level pool.
        """

        if not isinstance(raw_group, dict):
            raise SchemaValidationError("Each grouped set payload must be a JSON object")

        candidate_ids = raw_group.get("candidate_ids")
        field_score_matrix = raw_group.get("field_score_matrix")
        if not isinstance(candidate_ids, list):
            raise SchemaValidationError("Grouped set candidate_ids must be a list")
        if not isinstance(field_score_matrix, list):
            raise SchemaValidationError("Grouped set field_score_matrix must be a list")

        for candidate_id in candidate_ids:
            if candidate_id not in candidate_id_set:
                raise SchemaValidationError(
                    f"Grouped set references unknown candidate_id {candidate_id}"
                )

        return GroupedCandidateSet(
            group_id=stable_int_id("group", *sorted(candidate_ids)),
            candidate_ids=[int(candidate_id) for candidate_id in candidate_ids],
            field_score_matrix=field_score_matrix,
            field_order=CANONICAL_FIELD_ORDER,
        )


def _normalize_group_key(value: str | None) -> str | None:
    """Normalize canonical field values for deterministic exact-match grouping.

    Args:
        value (str | None): Canonical candidate field value.

    Returns:
        str | None: Lowercased trimmed value, or ``None`` when absent.
    """

    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None


def _fallback_field_score(value: str | None, evidence: list[Any]) -> float:
    """Assign a deterministic fallback score from field and evidence presence.

    Args:
        value (str | None): Canonical candidate field value.
        evidence (list[Any]): Evidence spans supporting the field.

    Returns:
        float: Deterministic fallback confidence score.
    """

    if value is None:
        return 0.0
    if evidence:
        return 0.9
    return 0.6

