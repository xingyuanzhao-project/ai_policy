"""Reproduce the granularity_refiner empty-content failure on one bill group.

Loads the grouped candidate set and the referenced candidate from the
run artifacts on disk, constructs a live ``GranularityRefiner``, and prints
the refiner's return value (or the raised exception).

Skipped unless ``NER_RUN_LIVE_TESTS=1`` is set because it contacts the live
LLM endpoint.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from src.ner.agents import (
    AgentExecutionConfig,
    GranularityRefiner,
    PromptExecutor,
    RefinementRequest,
)
from src.ner.agents.shared import render_prompt, serialize_for_prompt
from src.ner.runtime.llm_client import (
    STRUCTURED_OUTPUT_GUIDED_JSON,
    LLMClient,
    LLMConfig,
)
from src.ner.schemas.artifacts import CandidateQuadruplet, GroupedCandidateSet
from src.ner.storage.config_store import ConfigStore

LIVE_TESTS_ENABLED = os.environ.get("NER_RUN_LIVE_TESTS") == "1"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_DIR = PROJECT_ROOT / "settings"
RUN_DIR = PROJECT_ROOT / "data" / "ner_runs" / "runs" / "run_sonnet_full_20260416"
BILL_ID = "2024__RI S   2540"
TARGET_GROUP_ID = 7945457209652008633
TARGET_CANDIDATE_ID = 3473583289579683607


def _load_target_group() -> GroupedCandidateSet:
    """Read the grouped set file and return the failing group."""

    grouped_path = RUN_DIR / "grouped" / f"{BILL_ID}.json"
    grouped_payload = json.loads(grouped_path.read_text(encoding="utf-8"))
    for group_payload in grouped_payload:
        if int(group_payload["group_id"]) == TARGET_GROUP_ID:
            return GroupedCandidateSet.from_dict(group_payload)
    raise RuntimeError(f"group {TARGET_GROUP_ID} not found in {grouped_path}")


def _load_target_candidate() -> CandidateQuadruplet:
    """Scan candidate chunk files and return the failing candidate."""

    candidates_dir = RUN_DIR / "candidates" / BILL_ID
    for chunk_file in candidates_dir.glob("*.json"):
        if chunk_file.name.endswith(".raw.json"):
            continue
        payload = json.loads(chunk_file.read_text(encoding="utf-8"))
        for candidate_payload in payload:
            if int(candidate_payload["candidate_id"]) == TARGET_CANDIDATE_ID:
                return CandidateQuadruplet.from_dict(candidate_payload)
    raise RuntimeError(
        f"candidate {TARGET_CANDIDATE_ID} not found under {candidates_dir}"
    )


@unittest.skipUnless(
    LIVE_TESTS_ENABLED,
    "Set NER_RUN_LIVE_TESTS=1 to run this live reproducer",
)
class RefinerEmptyContentReproducer(unittest.TestCase):
    """Call the refiner with the exact input that crashed production."""

    def test_refiner_on_failing_group_prints_return(self) -> None:
        """Feed the failing group to the refiner and print the return."""

        config_store = ConfigStore()
        config_store.load(
            base_config_path=SETTINGS_DIR / "config.yml",
            ner_config_path=SETTINGS_DIR / "ner_config.yml",
            prompt_config_path=SETTINGS_DIR / "ner_prompts.json",
        )

        from src.ner.runtime.bootstrap import _resolve_api_key

        llm_client = LLMClient(
            LLMConfig(
                base_url=config_store.llm_config["base_url"],
                api_key=_resolve_api_key(config_store.llm_config),
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

        prompt_executor = PromptExecutor(llm_client)
        refiner = GranularityRefiner(
            prompt_template=config_store.prompt_config("granularity_refiner")[
                "prompt_template"
            ],
            output_schema=config_store.prompt_config("granularity_refiner")[
                "output_schema"
            ],
            execution_config=AgentExecutionConfig(
                **config_store.agent_config("granularity_refiner")
            ),
            prompt_executor=prompt_executor,
        )

        grouped_set = _load_target_group()
        candidate = _load_target_candidate()
        request = RefinementRequest(
            grouped_candidate_set=grouped_set,
            candidate_pool_by_id={candidate.candidate_id: candidate},
        )

        print("=" * 80)
        print("INPUT: GroupedCandidateSet")
        print("=" * 80)
        print(json.dumps(grouped_set.to_dict(), indent=2, ensure_ascii=False))
        print("=" * 80)
        print("INPUT: CandidateQuadruplet")
        print("=" * 80)
        print(json.dumps(candidate.to_dict(), indent=2, ensure_ascii=False))
        print("=" * 80)
        print("INPUT: rendered prompt sent to LLM")
        print("=" * 80)
        rendered_prompt = render_prompt(
            config_store.prompt_config("granularity_refiner")["prompt_template"],
            grouped_candidate_set_json=serialize_for_prompt(grouped_set),
            candidate_pool_json=serialize_for_prompt([candidate]),
        )
        print(rendered_prompt)
        print("=" * 80)
        print(f"rendered prompt length: {len(rendered_prompt)} chars")
        print("=" * 80)
        print("REFINER RETURN / EXCEPTION")
        print("=" * 80)

        try:
            result = refiner.run(request)
        except Exception as exc:
            print(f"EXCEPTION: {type(exc).__name__}: {exc}")
            raise
        finally:
            llm_client.close()

        print("raw_response:")
        print(result.raw_response)
        print()
        print("parsed_response:")
        refined_outputs, refinement_artifact = result.parsed_response
        for refined in refined_outputs:
            print(json.dumps(refined.to_dict(), indent=2, ensure_ascii=False))
        if refinement_artifact is not None:
            print("refinement_artifact:")
            print(
                json.dumps(
                    refinement_artifact.to_dict(), indent=2, ensure_ascii=False
                )
            )
