"""Offline route tests for the local Flask QA app."""

from __future__ import annotations

import unittest

from src.qa.artifacts import AnswerResult, RetrievedChunk
from src.qa.local_answer_support import AnswerModelOption
from src.qa.web_app import create_app


class FakeQAService:
    """Simple fake service used to test the Flask routes."""

    def __init__(self) -> None:
        self.default_answer_model = "gemini-2.5-flash"
        self.available_answer_models = (
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "local::hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        )
        self.answer_model_options = (
            AnswerModelOption(
                option_id="gemini-2.5-flash",
                label="gemini-2.5-flash",
            ),
            AnswerModelOption(
                option_id="gemini-2.5-pro",
                label="gemini-2.5-pro",
            ),
            AnswerModelOption(
                option_id="local::hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                label="Local / hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            ),
        )
        self._answer_model_labels = {
            answer_model_option.option_id: answer_model_option.label
            for answer_model_option in self.answer_model_options
        }
        self.last_answer_model: str | None = None

    def answer_question(
        self,
        question: str,
        answer_model: str | None = None,
    ) -> AnswerResult:
        self.last_answer_model = (
            self.default_answer_model if not answer_model else answer_model
        )
        return AnswerResult(
            question=question,
            answer="The bill requires impact disclosures. [1]",
            answer_model=self._answer_model_labels[self.last_answer_model],
            citations=[
                RetrievedChunk(
                    rank=1,
                    score=0.95,
                    chunk_id=10,
                    bill_id="BILL-001",
                    text="Impact disclosures are required for covered systems.",
                    start_offset=0,
                    end_offset=52,
                    state="CA",
                    title="AI Accountability Act",
                    status="Introduced",
                )
            ],
        )


class FlaskRouteTests(unittest.TestCase):
    """Verify the minimal local browser and JSON routes."""

    def setUp(self) -> None:
        self._qa_service = FakeQAService()
        self._client = create_app(self._qa_service).test_client()

    def test_api_route_returns_typed_answer_payload(self) -> None:
        """Verify the JSON route returns the serialized AnswerResult payload."""

        response = self._client.post(
            "/api/ask",
            json={"question": "What disclosure does the bill require?"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        assert payload is not None
        self.assertEqual(payload["question"], "What disclosure does the bill require?")
        self.assertEqual(payload["citations"][0]["bill_id"], "BILL-001")
        self.assertEqual(payload["answer_model"], "gemini-2.5-flash")

    def test_form_route_renders_answer_and_citations(self) -> None:
        """Verify the browser route renders answer text and citation metadata."""

        response = self._client.post(
            "/",
            data={"question": "What disclosure does the bill require?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"The bill requires impact disclosures.", response.data)
        self.assertIn(b"BILL-001", response.data)
        self.assertIn(b"gemini-2.5-flash", response.data)

    def test_form_route_renders_model_dropdown(self) -> None:
        """Verify the browser page exposes the configured answer-model options."""

        response = self._client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<select id=\"answer_model\"", response.data)
        self.assertIn(b"gemini-2.5-pro", response.data)
        self.assertIn(
            b"Local / hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            response.data,
        )

    def test_api_route_accepts_selected_answer_model(self) -> None:
        """Verify the API route forwards the requested answer model."""

        response = self._client.post(
            "/api/ask",
            json={
                "question": "What disclosure does the bill require?",
                "answer_model": "gemini-2.5-pro",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        assert payload is not None
        self.assertEqual(
            payload["answer_model"],
            "gemini-2.5-pro",
        )
        self.assertEqual(
            self._qa_service.last_answer_model,
            "gemini-2.5-pro",
        )

    def test_api_route_rejects_empty_questions(self) -> None:
        """Verify the JSON route returns a clear validation error."""

        response = self._client.post("/api/ask", json={"question": ""})
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        assert payload is not None
        self.assertIn("question is required", payload["error"])


if __name__ == "__main__":
    unittest.main()
