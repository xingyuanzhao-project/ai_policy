"""Offline route tests for the local Flask QA app."""

from __future__ import annotations

import unittest

from src.qa.artifacts import STATUS_BUCKETS, AnswerResult, RetrievedChunk
from src.qa.local_answer_support import AnswerModelOption
from src.qa.web_app import create_app

_DEFAULT_ANSWER_MODEL = "google/gemini-2.5-flash"
_ALTERNATE_ANSWER_MODEL = "anthropic/claude-haiku-4.5"


class FakeQAService:
    """Simple fake service used to test the Flask routes."""

    def __init__(self) -> None:
        self.default_answer_model = _DEFAULT_ANSWER_MODEL
        self.available_answer_models = (
            _DEFAULT_ANSWER_MODEL,
            _ALTERNATE_ANSWER_MODEL,
            "local::hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        )
        self.answer_model_options = (
            AnswerModelOption(
                option_id=_DEFAULT_ANSWER_MODEL,
                label=_DEFAULT_ANSWER_MODEL,
            ),
            AnswerModelOption(
                option_id=_ALTERNATE_ANSWER_MODEL,
                label=_ALTERNATE_ANSWER_MODEL,
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
        self.last_filters: dict | None = None
        self.last_question: str | None = None
        self.last_capture_trace: bool | None = None
        self.inferred_filters: dict = {}
        self.trace_payload: list[dict] = []

    @property
    def available_filter_values(self) -> dict[str, list]:
        return {
            "year": [2025, 2024, 2023],
            "state": ["CA", "NY", "TX"],
            "status_bucket": list(STATUS_BUCKETS),
            "topics": ["Government Use", "Private Sector Use"],
        }

    def answer_question(
        self,
        question: str,
        answer_model: str | None = None,
        filters: dict | None = None,
        capture_trace: bool | None = None,
    ) -> AnswerResult:
        self.last_answer_model = (
            self.default_answer_model if not answer_model else answer_model
        )
        self.last_filters = filters
        self.last_question = question
        self.last_capture_trace = capture_trace
        applied_filters = filters if filters is not None else dict(self.inferred_filters)
        effective_capture = bool(capture_trace) if capture_trace is not None else False
        trace = list(self.trace_payload) if effective_capture else []
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
            applied_filters=applied_filters,
            trace=trace,
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
        self.assertEqual(payload["answer_model"], _DEFAULT_ANSWER_MODEL)

    def test_form_route_renders_answer_and_citations(self) -> None:
        """Verify the browser route renders answer text and citation metadata."""

        response = self._client.post(
            "/",
            data={"question": "What disclosure does the bill require?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"The bill requires impact disclosures.", response.data)
        self.assertIn(b"BILL-001", response.data)
        self.assertIn(_DEFAULT_ANSWER_MODEL.encode("utf-8"), response.data)

    def test_form_route_renders_model_dropdown(self) -> None:
        """Verify the browser page exposes the configured answer-model options."""

        response = self._client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<select id=\"answer_model\"", response.data)
        self.assertIn(_ALTERNATE_ANSWER_MODEL.encode("utf-8"), response.data)
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
                "answer_model": _ALTERNATE_ANSWER_MODEL,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        assert payload is not None
        self.assertEqual(
            payload["answer_model"],
            _ALTERNATE_ANSWER_MODEL,
        )
        self.assertEqual(
            self._qa_service.last_answer_model,
            _ALTERNATE_ANSWER_MODEL,
        )

    def test_api_route_rejects_empty_questions(self) -> None:
        """Verify the JSON route returns a clear validation error."""

        response = self._client.post("/api/ask", json={"question": ""})
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        assert payload is not None
        self.assertIn("question is required", payload["error"])

    def test_form_page_does_not_render_manual_filter_dropdowns(self) -> None:
        """Verify the filter dropdowns have been removed from the browser UI."""

        response = self._client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b'name="state"', response.data)
        self.assertNotIn(b'name="year"', response.data)
        self.assertNotIn(b'name="status_bucket"', response.data)
        self.assertNotIn(b'name="topics"', response.data)

    def test_form_route_renders_inferred_filters_line(self) -> None:
        """Verify inferred filters populated by the service appear in the result panel."""

        self._qa_service.inferred_filters = {
            "state": "CA",
            "year": 2024,
            "status_bucket": "Enacted",
            "topics": ["Facial Recognition"],
        }

        response = self._client.post(
            "/",
            data={"question": "What disclosure does the bill require?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Filters inferred from question:", response.data)
        self.assertIn(b"state=CA", response.data)
        self.assertIn(b"year=2024", response.data)
        self.assertIn(b"status=Enacted", response.data)
        self.assertIn(b"Facial Recognition", response.data)

    def test_form_route_renders_multi_value_filters_with_or_syntax(self) -> None:
        """Verify list-valued fields render as ``field in [a, b]`` for OR logic."""

        self._qa_service.inferred_filters = {
            "state": ["CA", "TX"],
            "year": [2024, 2025],
            "status_bucket": ["Enacted", "Pending"],
            "topics": ["Facial Recognition", "Private Sector Use"],
        }

        response = self._client.post(
            "/",
            data={"question": "AI bills passed or pending in CA or TX?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"state in [CA, TX]", response.data)
        self.assertIn(b"year in [2024, 2025]", response.data)
        self.assertIn(b"status in [Enacted, Pending]", response.data)
        self.assertIn(
            b"topics in [Facial Recognition, Private Sector Use]", response.data
        )

    def test_api_route_returns_applied_filters(self) -> None:
        """Verify the JSON payload carries the applied_filters field."""

        self._qa_service.inferred_filters = {"state": "NY"}

        response = self._client.post(
            "/api/ask",
            json={"question": "What disclosure does the bill require?"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        assert payload is not None
        self.assertEqual(payload.get("applied_filters"), {"state": "NY"})

    def test_form_always_renders_show_trace_checkbox(self) -> None:
        """Verify the 'Show planner trace' checkbox is always present on GET."""

        response = self._client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'name="show_trace"', response.data)
        self.assertIn(b"Show planner trace", response.data)

    def test_show_trace_checkbox_defaults_to_unchecked(self) -> None:
        """Verify the checkbox is unchecked when the app was built with show_trace=False."""

        response = self._client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b'name="show_trace" value="1" checked', response.data)

    def test_show_trace_checkbox_defaults_to_checked_when_configured(self) -> None:
        """Verify show_trace=True on create_app pre-checks the checkbox."""

        client = create_app(FakeQAService(), show_trace=True).test_client()

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'name="show_trace" value="1" checked', response.data)

    def test_form_submit_with_show_trace_passes_capture_trace_true(self) -> None:
        """Verify checking the box forwards capture_trace=True to the service."""

        self._qa_service.trace_payload = [{"turn": 0, "role": "seed", "messages": []}]

        response = self._client.post(
            "/",
            data={
                "question": "What disclosure does the bill require?",
                "show_trace": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._qa_service.last_capture_trace, True)
        self.assertIn(b"Planner trace", response.data)

    def test_form_submit_without_show_trace_passes_capture_trace_false(self) -> None:
        """Verify omitting the checkbox forwards capture_trace=False to the service."""

        response = self._client.post(
            "/",
            data={"question": "What disclosure does the bill require?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._qa_service.last_capture_trace, False)
        self.assertNotIn(b"Planner trace", response.data)

    def test_api_route_forwards_capture_trace_flag(self) -> None:
        """Verify capture_trace in JSON body reaches the service."""

        self._qa_service.trace_payload = [{"turn": 0, "role": "seed", "messages": []}]

        response = self._client.post(
            "/api/ask",
            json={
                "question": "What disclosure does the bill require?",
                "capture_trace": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        assert payload is not None
        self.assertEqual(self._qa_service.last_capture_trace, True)
        self.assertEqual(
            payload.get("trace"),
            [{"turn": 0, "role": "seed", "messages": []}],
        )


if __name__ == "__main__":
    unittest.main()
