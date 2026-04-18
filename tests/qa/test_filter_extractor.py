"""Offline tests for the self-query filter extractor."""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from typing import Any

from src.qa.filter_extractor import FilterExtractor


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    function: _FakeFunction


@dataclass
class _FakeMessage:
    tool_calls: list[_FakeToolCall]
    content: str | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]


class _FakeCompletions:
    """Capture ``chat.completions.create`` arguments and yield a preset response."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None
        self.response: _FakeResponse | None = None
        self.exception: Exception | None = None
        self.call_count = 0

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.call_count += 1
        self.last_kwargs = kwargs
        if self.exception is not None:
            raise self.exception
        assert self.response is not None
        return self.response


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeOpenAIClient:
    """Minimal OpenAI-compatible stub exposing ``chat.completions.create``."""

    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.chat = _FakeChat(self.completions)


_AVAILABLE = {
    "year": [2025, 2024, 2023],
    "state": ["CA", "NY", "TX"],
    "status_bucket": ["Enacted", "Failed", "Vetoed", "Pending", "Other"],
    "topics": ["Government Use", "Private Sector Use", "Facial Recognition"],
}


def _response_with_tool_call(arguments: dict[str, Any]) -> _FakeResponse:
    """Build a fake response shaped like an OpenAI tool-call completion."""

    return _FakeResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    tool_calls=[
                        _FakeToolCall(
                            function=_FakeFunction(
                                name="search_corpus",
                                arguments=json.dumps(arguments),
                            )
                        )
                    ]
                )
            )
        ]
    )


class FilterExtractorTests(unittest.TestCase):
    """Cover the extract() happy and fallback paths."""

    def test_valid_tool_call_is_parsed(self) -> None:
        client = _FakeOpenAIClient()
        client.completions.response = _response_with_tool_call(
            {
                "semantic_query": "facial recognition disclosure",
                "state": "CA",
                "year": 2024,
                "status_bucket": "Enacted",
                "topics": ["Facial Recognition"],
            }
        )
        extractor = FilterExtractor(client=client, model="test/model")

        extracted = extractor.extract(
            "In California, show me 2024 enacted facial recognition disclosure bills",
            _AVAILABLE,
        )

        self.assertEqual(extracted.semantic_query, "facial recognition disclosure")
        self.assertEqual(
            extracted.filters,
            {
                "year": 2024,
                "state": "CA",
                "status_bucket": "Enacted",
                "topics": ["Facial Recognition"],
            },
        )
        kwargs = client.completions.last_kwargs
        assert kwargs is not None
        self.assertEqual(kwargs["model"], "test/model")
        self.assertEqual(
            kwargs["tool_choice"],
            {"type": "function", "function": {"name": "search_corpus"}},
        )
        self.assertEqual(len(kwargs["tools"]), 1)

    def test_unknown_enum_values_are_dropped(self) -> None:
        client = _FakeOpenAIClient()
        client.completions.response = _response_with_tool_call(
            {
                "semantic_query": "disclosure requirements",
                "state": "ZZ",
                "status_bucket": "MadeUp",
                "topics": ["Facial Recognition", "Imaginary Topic"],
            }
        )
        extractor = FilterExtractor(client=client, model="test/model")

        extracted = extractor.extract("any question", _AVAILABLE)

        self.assertEqual(extracted.semantic_query, "disclosure requirements")
        self.assertNotIn("state", extracted.filters)
        self.assertNotIn("status_bucket", extracted.filters)
        self.assertEqual(extracted.filters.get("topics"), ["Facial Recognition"])

    def test_year_coerced_from_string(self) -> None:
        client = _FakeOpenAIClient()
        client.completions.response = _response_with_tool_call(
            {"semantic_query": "anything", "year": "2023"}
        )
        extractor = FilterExtractor(client=client, model="test/model")

        extracted = extractor.extract("anything about 2023 bills", _AVAILABLE)

        self.assertEqual(extracted.filters.get("year"), 2023)

    def test_model_without_tool_call_falls_back(self) -> None:
        client = _FakeOpenAIClient()
        client.completions.response = _FakeResponse(
            choices=[_FakeChoice(message=_FakeMessage(tool_calls=[]))]
        )
        extractor = FilterExtractor(client=client, model="test/model")

        extracted = extractor.extract("what bills protect privacy?", _AVAILABLE)

        self.assertEqual(extracted.semantic_query, "what bills protect privacy?")
        self.assertEqual(extracted.filters, {})

    def test_malformed_json_arguments_fall_back(self) -> None:
        client = _FakeOpenAIClient()
        client.completions.response = _FakeResponse(
            choices=[
                _FakeChoice(
                    message=_FakeMessage(
                        tool_calls=[
                            _FakeToolCall(
                                function=_FakeFunction(
                                    name="search_corpus",
                                    arguments="{not valid json",
                                )
                            )
                        ]
                    )
                )
            ]
        )
        extractor = FilterExtractor(client=client, model="test/model")

        extracted = extractor.extract("broken response test", _AVAILABLE)

        self.assertEqual(extracted.semantic_query, "broken response test")
        self.assertEqual(extracted.filters, {})

    def test_api_exception_falls_back(self) -> None:
        client = _FakeOpenAIClient()
        client.completions.exception = RuntimeError("network down")
        extractor = FilterExtractor(client=client, model="test/model")

        extracted = extractor.extract("anything", _AVAILABLE)

        self.assertEqual(extracted.semantic_query, "anything")
        self.assertEqual(extracted.filters, {})

    def test_empty_question_short_circuits(self) -> None:
        client = _FakeOpenAIClient()
        extractor = FilterExtractor(client=client, model="test/model")

        extracted = extractor.extract("   ", _AVAILABLE)

        self.assertEqual(client.completions.call_count, 0)
        self.assertEqual(extracted.filters, {})

    def test_schema_includes_dynamic_state_enum(self) -> None:
        client = _FakeOpenAIClient()
        client.completions.response = _response_with_tool_call(
            {"semantic_query": "x"}
        )
        extractor = FilterExtractor(client=client, model="test/model")

        extractor.extract("any question", _AVAILABLE)

        kwargs = client.completions.last_kwargs
        assert kwargs is not None
        schema = kwargs["tools"][0]["function"]["parameters"]["properties"]
        self.assertEqual(schema["state"]["type"], "array")
        self.assertEqual(schema["state"]["items"]["enum"], ["CA", "NY", "TX"])
        self.assertEqual(schema["status_bucket"]["type"], "array")
        self.assertEqual(
            schema["status_bucket"]["items"]["enum"],
            ["Enacted", "Failed", "Vetoed", "Pending", "Other"],
        )
        self.assertEqual(schema["year"]["type"], "array")
        self.assertEqual(schema["year"]["items"]["type"], "integer")
        self.assertEqual(
            schema["topics"]["items"]["enum"],
            ["Government Use", "Private Sector Use", "Facial Recognition"],
        )

    def test_multi_value_state_is_preserved_as_list(self) -> None:
        """Verify ``state=["CA","TX"]`` survives validation as a list for OR logic."""

        client = _FakeOpenAIClient()
        client.completions.response = _response_with_tool_call(
            {
                "semantic_query": "disclosure rules",
                "state": ["CA", "TX"],
                "status_bucket": ["Enacted", "Pending"],
                "year": [2024, 2025],
            }
        )
        extractor = FilterExtractor(client=client, model="test/model")

        extracted = extractor.extract(
            "AI bills in California or Texas, enacted or pending, 2024 or 2025",
            _AVAILABLE,
        )

        self.assertEqual(extracted.filters["state"], ["CA", "TX"])
        self.assertEqual(
            extracted.filters["status_bucket"], ["Enacted", "Pending"]
        )
        self.assertEqual(extracted.filters["year"], [2024, 2025])

    def test_single_element_list_is_collapsed_to_scalar(self) -> None:
        """Verify ``state=["CA"]`` collapses to scalar ``state="CA"``."""

        client = _FakeOpenAIClient()
        client.completions.response = _response_with_tool_call(
            {
                "semantic_query": "whatever",
                "state": ["CA"],
                "status_bucket": ["Enacted"],
                "year": [2024],
            }
        )
        extractor = FilterExtractor(client=client, model="test/model")

        extracted = extractor.extract("California enacted bills in 2024", _AVAILABLE)

        self.assertEqual(extracted.filters["state"], "CA")
        self.assertEqual(extracted.filters["status_bucket"], "Enacted")
        self.assertEqual(extracted.filters["year"], 2024)

    def test_unknown_values_are_dropped_from_list(self) -> None:
        """Verify invalid entries inside a list are filtered out."""

        client = _FakeOpenAIClient()
        client.completions.response = _response_with_tool_call(
            {
                "semantic_query": "x",
                "state": ["CA", "ZZ", "TX"],
                "status_bucket": ["Enacted", "MadeUp"],
            }
        )
        extractor = FilterExtractor(client=client, model="test/model")

        extracted = extractor.extract("question", _AVAILABLE)

        self.assertEqual(extracted.filters["state"], ["CA", "TX"])
        self.assertEqual(extracted.filters["status_bucket"], "Enacted")

    def test_scalar_inputs_remain_backward_compatible(self) -> None:
        """Verify scalar inputs continue to validate and collapse as before."""

        client = _FakeOpenAIClient()
        client.completions.response = _response_with_tool_call(
            {
                "semantic_query": "x",
                "state": "CA",
                "year": 2024,
                "status_bucket": "Enacted",
            }
        )
        extractor = FilterExtractor(client=client, model="test/model")

        extracted = extractor.extract("California 2024 enacted", _AVAILABLE)

        self.assertEqual(extracted.filters["state"], "CA")
        self.assertEqual(extracted.filters["year"], 2024)
        self.assertEqual(extracted.filters["status_bucket"], "Enacted")


if __name__ == "__main__":
    unittest.main()
