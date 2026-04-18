"""OpenAI-compatible client wrapper for QA embeddings and answer generation.

- Owns one OpenAI-compatible client surface that can talk to OpenRouter or
  another compatible provider endpoint by configuration change.
- Keeps retrieval embeddings fixed while allowing answer generation to override
  the model per request.
- Normalizes embeddings so cosine retrieval can use a simple dot product.
- Does not load corpora, persist cache files, or serve HTTP routes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from openai import OpenAI

from .artifacts import RetrievedChunk

_DEFAULT_ANSWER_MAX_TOKENS = 1024


class OpenAICompatibleClient:
    """Wrap an OpenAI-compatible provider endpoint for the local QA app."""

    def __init__(
        self,
        api_key: str,
        api_base_url: str,
        embedding_model: str,
        answer_model: str,
    ) -> None:
        """Initialize the shared OpenAI-compatible client wrapper."""

        self._client = OpenAI(api_key=api_key, base_url=api_base_url)
        self._api_base_url = api_base_url
        self._embedding_model = embedding_model
        self._answer_model = answer_model

    @property
    def embedding_model(self) -> str:
        """Return the configured embedding model name."""

        return self._embedding_model

    @property
    def answer_model(self) -> str:
        """Return the configured default answer model name."""

        return self._answer_model

    @property
    def api_base_url(self) -> str:
        """Return the configured provider base URL."""

        return self._api_base_url

    @property
    def openai_client(self) -> OpenAI:
        """Return the underlying OpenAI-compatible client for tool-use calls."""

        return self._client

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Embed document chunks for retrieval."""

        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._embedding_model,
            input=list(texts),
        )
        embeddings = getattr(response, "data", [])
        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Embedding response length did not match input length"
            )
        return [self._normalize_embedding(embedding.embedding) for embedding in embeddings]

    def embed_query(self, question: str) -> np.ndarray:
        """Embed one question for nearest-neighbor retrieval."""

        response = self._client.embeddings.create(
            model=self._embedding_model,
            input=[question],
        )
        embeddings = getattr(response, "data", [])
        if len(embeddings) != 1:
            raise RuntimeError(
                "Query embedding response must contain exactly one vector"
            )
        return self._normalize_embedding(embeddings[0].embedding)

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
        answer_model: str | None = None,
    ) -> str:
        """Generate an answer grounded in retrieved bill chunks.

        Note: QAService no longer calls this method on the hot path; the
        orchestrator-workers :class:`PlannerAgent` now owns answer synthesis.
        It is kept here for the live-OpenRouter test harness and as a
        back-compatible utility for ad-hoc callers.
        """

        prompt = self._build_answer_prompt(question, retrieved_chunks)
        response = self._client.chat.completions.create(
            model=self._answer_model if answer_model is None else answer_model,
            max_tokens=_DEFAULT_ANSWER_MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions about state AI bills using only the "
                        "retrieved bill text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content
        if not answer:
            raise RuntimeError("Answer generation returned empty text")
        return answer.strip()

    def close(self) -> None:
        """Release any HTTP resources held by the underlying client."""

        close_method = getattr(self._client, "close", None)
        if callable(close_method):
            close_method()

    def _build_answer_prompt(
        self,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> str:
        """Render the answer-generation prompt from retrieved evidence."""

        context_blocks: list[str] = []
        for chunk in retrieved_chunks:
            context_blocks.append(
                "\n".join(
                    [
                        f"[{chunk.rank}] bill_id={chunk.bill_id}",
                        f"state={chunk.state}",
                        f"title={chunk.title or 'N/A'}",
                        f"status={chunk.status or 'N/A'}",
                        f"offsets={chunk.start_offset}:{chunk.end_offset}",
                        f"score={chunk.score:.4f}",
                        "text:",
                        chunk.text,
                    ]
                )
            )
        context_text = "\n\n".join(context_blocks)
        return "\n".join(
            [
                "You answer questions about United States state AI bills.",
                "Use only the retrieved bill context below.",
                "If the context does not support the answer, say that the answer is not supported by the retrieved bill text.",
                "Cite supporting chunks inline with bracketed citation numbers such as [1] and [2].",
                "",
                f"Question: {question.strip()}",
                "",
                "Retrieved context:",
                context_text,
            ]
        )

    def _normalize_embedding(self, values: Sequence[float]) -> np.ndarray:
        """Return a normalized float32 embedding vector."""

        vector = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise RuntimeError("Embedding response contained a zero-norm vector")
        return vector / norm


ProviderClient = OpenAICompatibleClient

__all__ = ["OpenAICompatibleClient", "ProviderClient"]
