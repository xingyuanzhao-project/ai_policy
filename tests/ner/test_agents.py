"""Agent and LLM client validation tests for the NER pipeline.

- Verifies schema-level agent guarantees without contacting the model.
- Verifies structured-output provider branching in ``LLMClient.generate``.
- Verifies live online agent behavior against the configured LLM endpoint.
- Does not implement runtime orchestration or storage behavior.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from src.ner.agents import (
    AgentExecutionConfig,
    EvalAssembler,
    GranularityRefiner,
    PromptExecutor,
    RefinementRequest,
    ZeroShotAnnotator,
)
from src.ner.runtime.llm_client import (
    STRUCTURED_OUTPUT_GUIDED_JSON,
    STRUCTURED_OUTPUT_JSON_SCHEMA,
    LLMClient,
    LLMConfig,
)
from src.ner.schemas.artifacts import (
    CandidateQuadruplet,
    ContextChunk,
    GroupedCandidateSet,
    RefinedQuadruplet,
    RefinementArtifact,
    SpanRef,
)
from src.ner.schemas.validation import (
    validate_candidate_quadruplet,
    validate_grouped_candidate_set,
    validate_refinement_artifact,
    validate_refined_quadruplet,
)
from src.ner.storage.config_store import ConfigStore

LIVE_TESTS_ENABLED = os.environ.get("NER_RUN_LIVE_TESTS") == "1"


class AgentContractTests(unittest.TestCase):
    """Schema-level agent guarantees that do not require a live model."""

    def test_candidate_validation_allows_partially_missing_fields(self) -> None:
        """Verify candidate validation permits nullable canonical fields."""

        candidate = CandidateQuadruplet(
            candidate_id=101,
            entity="artificial intelligence system",
            type=None,
            attribute="definition",
            value=None,
        )
        validate_candidate_quadruplet(candidate)

    def test_zero_shot_annotator_retries_once_with_fallback_prompt(self) -> None:
        """Verify malformed primary output retries once with a stricter prompt."""

        prompt_executor = Mock()
        prompt_executor.execute.side_effect = [
            '{"candidates": [{"entity": "broken"',
            (
                '{"candidates":[{"entity":"artificial intelligence system",'
                '"type":"technology","attribute":"definition",'
                '"value":"machine-based system","entity_evidence":[{"start":0,'
                '"end":30,"text":"artificial intelligence system"}],'
                '"type_evidence":[],"attribute_evidence":[],"value_evidence":[]}]}'
            ),
        ]
        annotator = ZeroShotAnnotator(
            prompt_template="{bill_id}\n{chunk_id}\n{text}",
            output_schema={"type": "object"},
            execution_config=AgentExecutionConfig(temperature=0.0, max_tokens=768),
            prompt_executor=prompt_executor,
        )
        chunk = ContextChunk(
            chunk_id=11,
            bill_id="TEST-BILL",
            text=(
                "For purposes of this Act, an artificial intelligence system means "
                "a machine-based system."
            ),
            start_offset=0,
            end_offset=91,
        )

        result = annotator.run(chunk)

        self.assertEqual(len(result.parsed_response), 1)
        self.assertEqual(result.parsed_response[0].entity, "artificial intelligence system")
        self.assertEqual(prompt_executor.execute.call_count, 2)

    def test_zero_shot_annotator_repairs_out_of_bounds_span_using_exact_text(self) -> None:
        """Verify exact span text repairs malformed relative offsets."""

        prompt_executor = Mock()
        prompt_executor.execute.return_value = (
            '{"candidates":[{"entity":"artificial intelligence system",'
            '"type":"technology","attribute":"definition",'
            '"value":"machine-based system","entity_evidence":[{"start":999,'
            '"end":1200,"text":"artificial intelligence system"}],'
            '"type_evidence":[],"attribute_evidence":[],"value_evidence":[]}]}'
        )
        annotator = ZeroShotAnnotator(
            prompt_template="{bill_id}\n{chunk_id}\n{text}",
            output_schema={
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity_evidence": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "start": {"type": "integer", "maximum": 3000},
                                            "end": {"type": "integer", "maximum": 3000},
                                            "text": {"type": "string"},
                                        },
                                    },
                                }
                            },
                        },
                    }
                },
            },
            execution_config=AgentExecutionConfig(temperature=0.0, max_tokens=768),
            prompt_executor=prompt_executor,
        )
        chunk = ContextChunk(
            chunk_id=11,
            bill_id="TEST-BILL",
            text=(
                "For purposes of this Act, an artificial intelligence system means "
                "a machine-based system."
            ),
            start_offset=100,
            end_offset=191,
        )

        result = annotator.run(chunk)

        expected_relative_start = chunk.text.index("artificial intelligence system")
        expected_relative_end = expected_relative_start + len("artificial intelligence system")
        evidence = result.parsed_response[0].entity_evidence
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].start, chunk.start_offset + expected_relative_start)
        self.assertEqual(evidence[0].end, chunk.start_offset + expected_relative_end)
        self.assertEqual(evidence[0].text, "artificial intelligence system")
        self.assertEqual(prompt_executor.execute.call_count, 1)

    def test_zero_shot_annotator_drops_unresolvable_malformed_span(self) -> None:
        """Verify malformed evidence is dropped when it cannot be aligned."""

        prompt_executor = Mock()
        prompt_executor.execute.return_value = (
            '{"candidates":[{"entity":"artificial intelligence system",'
            '"type":"technology","attribute":"definition",'
            '"value":"machine-based system","entity_evidence":[{"start":999,'
            '"end":1200,"text":"not present in the chunk"}],'
            '"type_evidence":[],"attribute_evidence":[],"value_evidence":[]}]}'
        )
        annotator = ZeroShotAnnotator(
            prompt_template="{bill_id}\n{chunk_id}\n{text}",
            output_schema={"type": "object"},
            execution_config=AgentExecutionConfig(temperature=0.0, max_tokens=768),
            prompt_executor=prompt_executor,
        )
        chunk = ContextChunk(
            chunk_id=11,
            bill_id="TEST-BILL",
            text=(
                "For purposes of this Act, an artificial intelligence system means "
                "a machine-based system."
            ),
            start_offset=0,
            end_offset=91,
        )

        result = annotator.run(chunk)

        self.assertEqual(len(result.parsed_response), 1)
        self.assertEqual(result.parsed_response[0].entity_evidence, [])
        self.assertEqual(prompt_executor.execute.call_count, 1)

    def test_grouped_set_validation_enforces_matrix_shape_and_field_order(self) -> None:
        """Verify grouped-set validation enforces matrix and field-order rules."""

        grouped_set = GroupedCandidateSet(
            group_id=202,
            candidate_ids=[101, 102],
            field_score_matrix=[[0.9, 0.8, 0.7, 0.6], [0.6, 0.7, 0.8, 0.9]],
        )
        validate_grouped_candidate_set(grouped_set)

    def test_eval_assembler_uses_deterministic_fallback_on_malformed_json(self) -> None:
        """Verify malformed assembler JSON falls back to deterministic grouping."""

        prompt_executor = Mock()
        prompt_executor.execute.return_value = '{"groups": ['
        assembler = EvalAssembler(
            prompt_template="{candidates_json}",
            output_schema={"type": "object"},
            execution_config=AgentExecutionConfig(temperature=0.0, max_tokens=256),
            prompt_executor=prompt_executor,
        )
        candidates = [
            CandidateQuadruplet(
                candidate_id=101,
                entity="artificial intelligence system",
                type="technology",
                attribute="definition",
                value="machine-based system",
                entity_evidence=[SpanRef(1, 0, 30, "artificial intelligence system", 11)],
            ),
            CandidateQuadruplet(
                candidate_id=102,
                entity="artificial intelligence system",
                type="technology",
                attribute="definition",
                value="machine-based system",
                entity_evidence=[SpanRef(2, 0, 30, "artificial intelligence system", 12)],
            ),
            CandidateQuadruplet(
                candidate_id=103,
                entity="high-risk system",
                type="technology",
                attribute="restriction",
                value="requires disclosure",
            ),
        ]

        result = assembler.run(candidates)

        self.assertEqual(len(result.parsed_response), 2)
        self.assertEqual(result.parsed_response[0].candidate_ids, [101, 102])
        self.assertEqual(result.parsed_response[1].candidate_ids, [103])
        self.assertEqual(
            result.parsed_response[0].field_score_matrix[0],
            [0.9, 0.6, 0.6, 0.6],
        )

    def test_eval_assembler_skips_llm_for_oversized_candidate_pools(self) -> None:
        """Verify oversized candidate pools bypass the LLM and group deterministically."""

        prompt_executor = Mock()
        assembler = EvalAssembler(
            prompt_template="{candidates_json}",
            output_schema={"type": "object"},
            execution_config=AgentExecutionConfig(temperature=0.0, max_tokens=256),
            prompt_executor=prompt_executor,
        )
        candidates = [
            CandidateQuadruplet(
                candidate_id=1000 + index,
                entity=f"entity-{index % 2}",
                type="technology",
                attribute="definition",
                value=f"value-{index % 2}",
            )
            for index in range(81)
        ]

        result = assembler.run(candidates)

        self.assertFalse(prompt_executor.execute.called)
        self.assertEqual(len(result.parsed_response), 2)

    def test_refined_outputs_and_relation_labels_validate(self) -> None:
        """Verify refined outputs and refinement artifacts validate cleanly."""

        refined_output = RefinedQuadruplet(
            refined_id=303,
            source_group_id=202,
            source_candidate_ids=[101],
            entity="artificial intelligence system",
            type="technology",
            entity_evidence=[SpanRef(1, 0, 30, "artificial intelligence system", 11)],
        )
        refinement_artifact = RefinementArtifact(
            group_id=202,
            candidate_ids=[101],
            entity_relations=[[None]],
            type_relations=[[None]],
            attribute_relations=[[None]],
            value_relations=[[None]],
        )
        validate_refined_quadruplet(refined_output)
        validate_refinement_artifact(refinement_artifact)

    def test_malformed_optional_refinement_artifact_is_dropped(self) -> None:
        """Verify malformed optional artifacts are dropped instead of failing refinement."""

        prompt_executor = Mock()
        prompt_executor.execute.return_value = """
        {
          "refined_quadruplets": [
            {
              "source_candidate_ids": [101],
              "entity": "artificial intelligence system",
              "type": "technology",
              "attribute": "definition",
              "value": "machine-based system",
              "entity_evidence": [{"start": 0, "end": 30, "text": "artificial intelligence system", "chunk_id": 11}],
              "type_evidence": [],
              "attribute_evidence": [],
              "value_evidence": []
            }
          ],
          "refinement_artifact": {
            "candidate_ids": [101],
            "entity_relations": [],
            "type_relations": [],
            "attribute_relations": [],
            "value_relations": []
          }
        }
        """
        refiner = GranularityRefiner(
            prompt_template="{grouped_candidate_set_json}\n{candidate_pool_json}",
            output_schema={"type": "object"},
            execution_config=AgentExecutionConfig(temperature=0.0, max_tokens=128),
            prompt_executor=prompt_executor,
        )
        grouped_set = GroupedCandidateSet(
            group_id=202,
            candidate_ids=[101],
            field_score_matrix=[[0.9, 0.9, 0.9, 0.9]],
        )
        candidates = {
            101: CandidateQuadruplet(
                candidate_id=101,
                entity="artificial intelligence system",
                type="technology",
                attribute="definition",
                value="machine-based system",
            )
        }
        refined_outputs, refinement_artifact = refiner.run(
            RefinementRequest(
                grouped_candidate_set=grouped_set,
                candidate_pool_by_id=candidates,
            )
        ).parsed_response

        self.assertEqual(len(refined_outputs), 1)
        self.assertIsNone(refinement_artifact)

    def test_granularity_refiner_uses_deterministic_fallback_after_two_bad_jsons(self) -> None:
        """Verify repeated malformed refinement JSON falls back to best candidate."""

        prompt_executor = Mock()
        prompt_executor.execute.side_effect = [
            '{"refined_quadruplets": [',
            '{"refined_quadruplets": [',
        ]
        refiner = GranularityRefiner(
            prompt_template="{grouped_candidate_set_json}\n{candidate_pool_json}",
            output_schema={"type": "object"},
            execution_config=AgentExecutionConfig(temperature=0.0, max_tokens=256),
            prompt_executor=prompt_executor,
        )
        grouped_set = GroupedCandidateSet(
            group_id=202,
            candidate_ids=[101, 102],
            field_score_matrix=[[0.9, 0.9, 0.9, 0.8], [0.2, 0.2, 0.2, 0.2]],
        )
        candidates = {
            101: CandidateQuadruplet(
                candidate_id=101,
                entity="artificial intelligence system",
                type="technology",
                attribute="definition",
                value="machine-based system",
                entity_evidence=[SpanRef(1, 0, 30, "artificial intelligence system", 11)],
            ),
            102: CandidateQuadruplet(
                candidate_id=102,
                entity="AI system",
                type="technology",
                attribute="definition",
                value="system",
            ),
        }

        refined_outputs, refinement_artifact = refiner.run(
            RefinementRequest(
                grouped_candidate_set=grouped_set,
                candidate_pool_by_id=candidates,
            )
        ).parsed_response

        self.assertEqual(len(refined_outputs), 1)
        self.assertEqual(refined_outputs[0].source_candidate_ids, [101])
        self.assertIsNone(refinement_artifact)


class LLMClientStructuredOutputTests(unittest.TestCase):
    """Verify LLMClient.generate branches structured output by provider mode."""

    _TEST_SCHEMA: dict = {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
    }

    def _build_client(self, mode: str, skip_listing: bool = False) -> LLMClient:
        return LLMClient(
            LLMConfig(
                base_url="http://fake:8000/v1",
                api_key="test-key",
                model_name="test-model",
                temperature=0.0,
                max_tokens=64,
                structured_output_mode=mode,
                skip_model_listing=skip_listing,
            )
        )

    def _mock_completions(self, client: LLMClient) -> MagicMock:
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"status":"ok"}'
        mock_openai.chat.completions.create.return_value = mock_response
        client._client = mock_openai
        return mock_openai

    def test_guided_json_mode_passes_extra_body(self) -> None:
        """Verify guided_json mode sends schema via extra_body."""

        client = self._build_client(STRUCTURED_OUTPUT_GUIDED_JSON)
        mock_openai = self._mock_completions(client)

        client.generate(prompt="test", output_schema=self._TEST_SCHEMA)

        call_kwargs = mock_openai.chat.completions.create.call_args
        self.assertEqual(
            call_kwargs.kwargs["extra_body"],
            {"guided_json": self._TEST_SCHEMA},
        )
        self.assertNotIn("response_format", call_kwargs.kwargs)

    def test_json_schema_mode_passes_response_format(self) -> None:
        """Verify json_schema mode sends schema via response_format."""

        client = self._build_client(STRUCTURED_OUTPUT_JSON_SCHEMA)
        mock_openai = self._mock_completions(client)

        client.generate(prompt="test", output_schema=self._TEST_SCHEMA)

        call_kwargs = mock_openai.chat.completions.create.call_args
        expected_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "ner_output",
                "schema": self._TEST_SCHEMA,
                "strict": True,
            },
        }
        self.assertEqual(call_kwargs.kwargs["response_format"], expected_format)
        self.assertNotIn("extra_body", call_kwargs.kwargs)

    def test_invalid_structured_output_mode_raises_at_init(self) -> None:
        """Verify invalid structured_output_mode raises ValueError."""

        with self.assertRaises(ValueError):
            LLMConfig(
                base_url="http://fake:8000/v1",
                api_key="test-key",
                model_name="test-model",
                temperature=0.0,
                max_tokens=64,
                structured_output_mode="invalid_mode",
            )

    def test_verify_runtime_skips_model_listing_when_configured(self) -> None:
        """Verify skip_model_listing=True bypasses the /v1/models check."""

        client = self._build_client(STRUCTURED_OUTPUT_GUIDED_JSON, skip_listing=True)
        mock_openai = self._mock_completions(client)

        result = client.verify_runtime()

        mock_openai.models.list.assert_not_called()
        self.assertEqual(result.resolved_model_name, "test-model")

    def test_verify_runtime_checks_model_listing_by_default(self) -> None:
        """Verify skip_model_listing=False calls /v1/models and validates."""

        client = self._build_client(STRUCTURED_OUTPUT_GUIDED_JSON, skip_listing=False)
        mock_openai = self._mock_completions(client)
        mock_model = MagicMock()
        mock_model.id = "test-model"
        mock_openai.models.list.return_value.data = [mock_model]

        result = client.verify_runtime()

        mock_openai.models.list.assert_called_once()
        self.assertEqual(result.resolved_model_name, "test-model")


@unittest.skipUnless(LIVE_TESTS_ENABLED, "Set NER_RUN_LIVE_TESTS=1 to run live agent tests")
class LiveAgentTests(unittest.TestCase):
    """Real agent execution tests against the configured LLM endpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build live agent fixtures against the configured LLM endpoint."""

        project_root = Path(__file__).resolve().parents[3]
        settings_dir = project_root / "settings"

        config_store = ConfigStore()
        config_store.load(
            base_config_path=settings_dir / "config.yml",
            ner_config_path=settings_dir / "ner_config.yml",
            prompt_config_path=settings_dir / "ner_prompts.json",
        )
        from src.ner.runtime.bootstrap import _resolve_api_key

        api_key = _resolve_api_key(config_store.llm_config)
        llm_client = LLMClient(
            LLMConfig(
                base_url=config_store.llm_config["base_url"],
                api_key=api_key,
                model_name=config_store.llm_config["model_name"],
                temperature=float(config_store.llm_config["temperature"]),
                max_tokens=int(config_store.llm_config["max_tokens"]),
                max_retries=int(
                    config_store.runtime_config.get(
                        "max_retries",
                        config_store.llm_config.get("max_retries", 2),
                    )
                ),
                request_timeout_seconds=float(
                    config_store.runtime_config.get("request_timeout_seconds", 60)
                ),
                structured_output_mode=str(
                    config_store.llm_config.get(
                        "structured_output_mode", STRUCTURED_OUTPUT_GUIDED_JSON
                    )
                ),
                skip_model_listing=bool(
                    config_store.llm_config.get("skip_model_listing", False)
                ),
            )
        )
        llm_client.connect()
        llm_client.verify_runtime()

        prompt_executor = PromptExecutor(llm_client)
        cls._llm_client = llm_client
        cls._annotator = ZeroShotAnnotator(
            prompt_template=config_store.prompt_config("zero_shot_annotator")["prompt_template"],
            output_schema=config_store.prompt_config("zero_shot_annotator")["output_schema"],
            execution_config=AgentExecutionConfig(**config_store.agent_config("zero_shot_annotator")),
            prompt_executor=prompt_executor,
        )
        cls._assembler = EvalAssembler(
            prompt_template=config_store.prompt_config("eval_assembler")["prompt_template"],
            output_schema=config_store.prompt_config("eval_assembler")["output_schema"],
            execution_config=AgentExecutionConfig(**config_store.agent_config("eval_assembler")),
            prompt_executor=prompt_executor,
        )
        cls._refiner = GranularityRefiner(
            prompt_template=config_store.prompt_config("granularity_refiner")["prompt_template"],
            output_schema=config_store.prompt_config("granularity_refiner")["output_schema"],
            execution_config=AgentExecutionConfig(**config_store.agent_config("granularity_refiner")),
            prompt_executor=prompt_executor,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Close the shared LLM client after the test class finishes."""

        cls._llm_client.close()

    def test_zero_shot_annotator_runs_with_live_llm(self) -> None:
        """Verify the zero-shot annotator returns valid candidates online."""

        chunk = ContextChunk(
            chunk_id=11,
            bill_id="LIVE-BILL",
            text=(
                "For purposes of this Act, an artificial intelligence system means "
                "a machine-based system that can make predictions, recommendations, "
                "or decisions."
            ),
            start_offset=0,
            end_offset=147,
        )

        result = self._annotator.run(chunk)

        self.assertGreaterEqual(len(result.parsed_response), 1)
        for candidate in result.parsed_response:
            validate_candidate_quadruplet(candidate)

    def test_eval_assembler_runs_with_live_llm(self) -> None:
        """Verify the eval assembler returns valid grouped sets online."""

        candidates = [
            CandidateQuadruplet(
                candidate_id=101,
                entity="artificial intelligence system",
                type="technology",
                attribute="definition",
                value="machine-based system",
                entity_evidence=[SpanRef(1, 0, 30, "artificial intelligence system", 11)],
            ),
            CandidateQuadruplet(
                candidate_id=102,
                entity="AI system",
                type="technology",
                attribute="definition",
                value="makes predictions and recommendations",
                entity_evidence=[SpanRef(2, 31, 40, "AI system", 11)],
            ),
        ]

        result = self._assembler.run(candidates)

        self.assertGreaterEqual(len(result.parsed_response), 1)
        for grouped_set in result.parsed_response:
            validate_grouped_candidate_set(grouped_set)

    def test_granularity_refiner_runs_with_live_llm(self) -> None:
        """Verify the granularity refiner returns valid final outputs online."""

        candidates = {
            101: CandidateQuadruplet(
                candidate_id=101,
                entity="artificial intelligence system",
                type="technology",
                attribute="definition",
                value="machine-based system",
                entity_evidence=[SpanRef(1, 0, 30, "artificial intelligence system", 11)],
                value_evidence=[SpanRef(3, 41, 61, "machine-based system", 11)],
            ),
            102: CandidateQuadruplet(
                candidate_id=102,
                entity="AI system",
                type="technology",
                attribute="definition",
                value="makes predictions and recommendations",
                entity_evidence=[SpanRef(2, 31, 40, "AI system", 11)],
                value_evidence=[SpanRef(4, 62, 102, "makes predictions and recommendations", 11)],
            ),
        }
        grouped_set = GroupedCandidateSet(
            group_id=202,
            candidate_ids=[101, 102],
            field_score_matrix=[[0.9, 0.9, 0.9, 0.8], [0.8, 0.9, 0.9, 0.8]],
        )

        result = self._refiner.run(
            RefinementRequest(
                grouped_candidate_set=grouped_set,
                candidate_pool_by_id=candidates,
            )
        )
        refined_outputs, refinement_artifact = result.parsed_response

        self.assertGreaterEqual(len(refined_outputs), 1)
        for refined_output in refined_outputs:
            validate_refined_quadruplet(refined_output)
        if refinement_artifact is not None:
            validate_refinement_artifact(refinement_artifact)


if __name__ == "__main__":
    unittest.main()

