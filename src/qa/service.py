"""Service layer for planner-driven question answering over bill retrieval.

- Always runs a two-step flow: one cheap ``FilterExtractor`` pass to pre-parse
  the question followed by one ``PlannerAgent`` loop that decides which tools
  to call and synthesizes the final answer.
- Exposes the same :class:`AnswerResult` payload the UI and API routes
  already consume, with the extractor's filters surfaced via
  ``applied_filters`` and a constant ``routing_path = "agent"`` marker.
- Validates per-request answer-model selection against the configured allowlist
  and rejects selections that resolve to a local (non-tool-capable) target.
- Does not own filesystem indexing, tool execution, or LLM inference itself;
  those are delegated to :class:`PlannerAgent` and the retriever backends.
"""

from __future__ import annotations

from .artifacts import STATUS_BUCKETS, AnswerResult, validate_answer_result
from .filter_extractor import FilterExtractor
from .lexical_retriever import LexicalRetriever
from .local_answer_support import AnswerModelOption, LocalAnswerSupport
from .planner_agent import PlannerAgent
from .retriever import Retriever, _coerce_int_values, _coerce_str_values


class QAService:
    """Run the planner-driven question-answering flow over the local bill index."""

    def __init__(
        self,
        *,
        retriever: Retriever | None,
        planner_agent: PlannerAgent,
        filter_extractor: FilterExtractor,
        retrieval_top_k: int,
        default_answer_model: str,
        available_answer_models: tuple[str, ...],
        lexical_retriever: LexicalRetriever | None = None,
        answer_model_options: tuple[AnswerModelOption, ...] | None = None,
        local_answer_support: LocalAnswerSupport | None = None,
        capture_trace: bool = False,
    ) -> None:
        """Initialize the planner-backed QA service."""

        if retrieval_top_k <= 0:
            raise ValueError("QA retrieval_top_k must be > 0")
        if not default_answer_model.strip():
            raise ValueError("QA default_answer_model must be a non-empty string")
        if not available_answer_models:
            raise ValueError("QA available_answer_models must not be empty")
        if default_answer_model not in available_answer_models:
            raise ValueError(
                "QA default_answer_model must be included in available_answer_models"
            )
        if retriever is None and lexical_retriever is None:
            raise ValueError(
                "QAService requires either a vector retriever or a lexical retriever"
            )
        if retriever is not None and lexical_retriever is not None:
            raise ValueError("QAService accepts only one retrieval backend at a time")
        if answer_model_options is None:
            answer_model_options = tuple(
                AnswerModelOption(option_id=model_name, label=model_name)
                for model_name in available_answer_models
            )
        option_ids = tuple(option.option_id for option in answer_model_options)
        if option_ids != available_answer_models:
            raise ValueError(
                "QA answer_model_options must align exactly with available_answer_models"
            )
        self._retriever = retriever
        self._lexical_retriever = lexical_retriever
        self._planner_agent = planner_agent
        self._filter_extractor = filter_extractor
        self._retrieval_top_k = retrieval_top_k
        self._default_answer_model = default_answer_model
        self._available_answer_models = available_answer_models
        self._answer_model_options = answer_model_options
        self._answer_model_labels = {
            option.option_id: option.label for option in self._answer_model_options
        }
        self._local_answer_support = local_answer_support
        self._capture_trace = bool(capture_trace)

    @property
    def default_answer_model(self) -> str:
        """Return the default answer model used by the QA service."""

        return self._default_answer_model

    @property
    def available_answer_models(self) -> tuple[str, ...]:
        """Return the answer models users may select at query time."""

        return self._available_answer_models

    @property
    def answer_model_options(self) -> tuple[AnswerModelOption, ...]:
        """Return answer-model options with stable ids and user-facing labels."""

        return self._answer_model_options

    @property
    def available_filter_values(self) -> dict[str, list]:
        """Return unique facet values for the year/state/status/topics dropdowns."""

        return self._compute_available_filter_values()

    @property
    def capture_trace_default(self) -> bool:
        """Return the service-wide default for trace capture.

        The UI uses this to pre-check or un-check the per-request
        "Show planner trace" toggle when the page first loads.
        """

        return self._capture_trace

    def answer_question(
        self,
        question: str,
        answer_model: str | None = None,
        filters: dict | None = None,
        capture_trace: bool | None = None,
    ) -> AnswerResult:
        """Answer one user question via the extractor -> planner pipeline.

        The extractor always runs first as a cheap pre-filter pass that gives
        the planner a topical rewrite (``semantic_query``) and any metadata
        filters it could infer. Programmatic callers may still pass explicit
        ``filters`` to seed the planner directly and bypass the extractor.

        ``capture_trace`` is a per-request override of the service's default
        trace-capture setting (``self._capture_trace``). Pass ``True`` to
        record the planner's internal messages and tool calls for this one
        answer and ``False`` to skip capture even when the service default
        is on. ``None`` (the default) falls back to the service-wide setting.
        """

        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must be a non-empty string")
        selected_answer_model = self._resolve_answer_model(answer_model)
        self._reject_local_answer_model(selected_answer_model)

        available_filter_values = self.available_filter_values
        if filters is None:
            extracted = self._filter_extractor.extract(
                normalized_question, available_filter_values
            )
            semantic_query = extracted.semantic_query or normalized_question
            normalized_filters = self._normalize_filters(extracted.filters) or {}
        else:
            semantic_query = normalized_question
            normalized_filters = self._normalize_filters(filters) or {}

        effective_capture_trace = (
            self._capture_trace if capture_trace is None else bool(capture_trace)
        )

        planner_answer = self._planner_agent.answer(
            normalized_question,
            semantic_query=semantic_query,
            initial_filters=normalized_filters,
            available_filter_values=available_filter_values,
            planner_model=selected_answer_model,
            citation_cap=self._retrieval_top_k * 2,
            capture_trace=effective_capture_trace,
            worker_model=selected_answer_model,
        )

        applied_filters: dict = {
            **normalized_filters,
            "routing_path": planner_answer.routing_path,
        }

        result = AnswerResult(
            question=normalized_question,
            answer=planner_answer.answer_text,
            answer_model=self._display_answer_model(selected_answer_model),
            citations=list(planner_answer.citations),
            applied_filters=applied_filters,
            trace=list(planner_answer.trace),
        )
        validate_answer_result(result)
        return result

    def _resolve_answer_model(self, answer_model: str | None) -> str:
        """Resolve and validate the requested answer model."""

        if answer_model is None:
            return self._default_answer_model
        normalized_answer_model = answer_model.strip()
        if not normalized_answer_model:
            return self._default_answer_model
        if normalized_answer_model not in self._available_answer_models:
            raise ValueError(
                f"Unsupported answer_model '{normalized_answer_model}'. "
                f"Choose one of: {', '.join(self._available_answer_models)}"
            )
        return normalized_answer_model

    def _reject_local_answer_model(self, answer_model: str) -> None:
        """Reject local answer-model selections that cannot drive the planner loop."""

        if self._local_answer_support is None:
            return
        local_target = self._local_answer_support.resolve_answer_model(answer_model)
        if local_target is not None:
            raise ValueError(
                "Local answer models are not supported on the agentic path. "
                "Select a remote model from the dropdown."
            )

    def _display_answer_model(self, answer_model: str) -> str:
        """Return the user-facing label for one selected answer model id."""

        return self._answer_model_labels.get(answer_model, answer_model)

    @staticmethod
    def _normalize_filters(filters: dict | None) -> dict | None:
        """Drop empty filter fields so downstream retrievers can short-circuit.

        Accepts scalar or list input for ``year``, ``state``, ``status_bucket``.
        A single value is stored as a scalar so ``applied_filters`` stays
        concise for the UI; two or more values are preserved as a list so the
        retriever masks with OR-within-field semantics.
        """

        if not filters:
            return None
        cleaned: dict = {}

        years = _coerce_int_values(filters.get("year"))
        if years:
            cleaned["year"] = years[0] if len(years) == 1 else years

        for key in ("state", "status_bucket"):
            values = _coerce_str_values(filters.get(key))
            if values:
                cleaned[key] = values[0] if len(values) == 1 else values

        topics = _coerce_str_values(filters.get("topics"))
        if topics:
            cleaned["topics"] = topics

        return cleaned or None

    def _compute_available_filter_values(self) -> dict[str, list]:
        """Derive unique filter values from whichever retriever is active."""

        years: set[int] = set()
        states: set[str] = set()
        topics: set[str] = set()

        if self._retriever is not None:
            metadata = self._retriever.chunk_metadata
            years.update(int(year) for year in metadata.years.tolist() if int(year) > 0)
            states.update(str(state) for state in metadata.states.tolist() if str(state).strip())
            for topic_set in metadata.topic_sets:
                topics.update(topic_set)
        elif self._lexical_retriever is not None:
            for chunk in self._lexical_retriever.chunks:
                if int(chunk.year) > 0:
                    years.add(int(chunk.year))
                if str(chunk.state).strip():
                    states.add(str(chunk.state))
                topics.update(chunk.topics_list)

        return {
            "year": sorted(years, reverse=True),
            "state": sorted(states),
            "status_bucket": list(STATUS_BUCKETS),
            "topics": sorted(topics),
        }


__all__ = ["QAService"]
