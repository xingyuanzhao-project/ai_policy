"""Service layer for question answering over bill retrieval results.

- Orchestrates query embedding, nearest-neighbor retrieval, and answer
  synthesis through the configured provider client.
- Supports either a persisted vector index or an in-memory lexical retriever
  when the full app is running without hosted embeddings.
- Validates per-request answer-model selection against the configured allowlist
  and delegates optional local-answer routing to a dedicated helper.
- Produces typed answer payloads for both the browser UI and API routes.
- Does not own filesystem indexing or Flask route definitions.
"""

from __future__ import annotations

from .artifacts import AnswerResult, validate_answer_result
from .lexical_retriever import LexicalRetriever
from .local_answer_support import AnswerModelOption, LocalAnswerSupport
from .provider_client import OpenAICompatibleClient
from .retriever import Retriever


class QAService:
    """Run the full question-answering flow over the local bill index."""

    def __init__(
        self,
        retriever: Retriever | None,
        provider_client: OpenAICompatibleClient,
        retrieval_top_k: int,
        default_answer_model: str,
        available_answer_models: tuple[str, ...],
        lexical_retriever: LexicalRetriever | None = None,
        answer_model_options: tuple[AnswerModelOption, ...] | None = None,
        local_answer_support: LocalAnswerSupport | None = None,
    ) -> None:
        """Initialize the QA service."""

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
            raise ValueError("QAService requires either a vector retriever or a lexical retriever")
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
        self._provider_client = provider_client
        self._retrieval_top_k = retrieval_top_k
        self._default_answer_model = default_answer_model
        self._available_answer_models = available_answer_models
        self._answer_model_options = answer_model_options
        self._answer_model_labels = {
            option.option_id: option.label for option in self._answer_model_options
        }
        self._local_answer_support = local_answer_support

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

    def answer_question(
        self,
        question: str,
        answer_model: str | None = None,
    ) -> AnswerResult:
        """Answer one user question using bill retrieval plus provider synthesis."""

        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must be a non-empty string")
        selected_answer_model = self._resolve_answer_model(answer_model)

        if self._lexical_retriever is not None:
            retrieved_chunks = self._lexical_retriever.retrieve_question(
                normalized_question,
                top_k=self._retrieval_top_k,
            )
        else:
            assert self._retriever is not None
            query_embedding = self._provider_client.embed_query(normalized_question)
            retrieved_chunks = self._retriever.retrieve(
                query_embedding=query_embedding,
                top_k=self._retrieval_top_k,
            )
        if not retrieved_chunks:
            result = AnswerResult(
                question=normalized_question,
                answer="I could not find relevant bill text to answer that question.",
                answer_model=self._display_answer_model(selected_answer_model),
                citations=[],
            )
            validate_answer_result(result)
            return result

        answer_client = self._provider_client
        answer_model_name = selected_answer_model
        if self._local_answer_support is not None:
            local_answer_target = self._local_answer_support.resolve_answer_model(
                selected_answer_model
            )
            if local_answer_target is not None:
                answer_client = local_answer_target.client
                answer_model_name = local_answer_target.raw_model_name

        answer_text = answer_client.generate_answer(
            question=normalized_question,
            retrieved_chunks=retrieved_chunks,
            answer_model=answer_model_name,
        )
        result = AnswerResult(
            question=normalized_question,
            answer=answer_text,
            answer_model=self._display_answer_model(selected_answer_model),
            citations=retrieved_chunks,
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

    def _display_answer_model(self, answer_model: str) -> str:
        """Return the user-facing label for one selected answer model id."""

        return self._answer_model_labels.get(answer_model, answer_model)


__all__ = ["QAService"]
