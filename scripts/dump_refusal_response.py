"""Send the failing refiner prompt to OpenRouter and dump the full response.

Used once to inspect which fields (if any) carry a refusal/safety signal
that llm_client.py is currently ignoring. Run with the live env:

    $env:NER_RUN_LIVE_TESTS="1"
    .\.venv\Scripts\python.exe scripts\dump_refusal_response.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ner.agents.shared import render_prompt, serialize_for_prompt
from src.ner.runtime.bootstrap import _resolve_api_key
from src.ner.runtime.llm_client import (
    STRUCTURED_OUTPUT_GUIDED_JSON,
    LLMClient,
    LLMConfig,
)
from src.ner.schemas.artifacts import CandidateQuadruplet, GroupedCandidateSet
from src.ner.storage.config_store import ConfigStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_DIR = PROJECT_ROOT / "settings"
RUN_DIR = PROJECT_ROOT / "data" / "ner_runs" / "runs" / "run_sonnet_full_20260416"
BILL_ID = "2024__RI S   2540"
TARGET_GROUP_ID = 7945457209652008633
TARGET_CANDIDATE_ID = 3473583289579683607


def _load_target_group() -> GroupedCandidateSet:
    grouped_path = RUN_DIR / "grouped" / f"{BILL_ID}.json"
    for group_payload in json.loads(grouped_path.read_text(encoding="utf-8")):
        if int(group_payload["group_id"]) == TARGET_GROUP_ID:
            return GroupedCandidateSet.from_dict(group_payload)
    raise RuntimeError(f"group {TARGET_GROUP_ID} not found")


def _load_target_candidate() -> CandidateQuadruplet:
    candidates_dir = RUN_DIR / "candidates" / BILL_ID
    for chunk_file in candidates_dir.glob("*.json"):
        if chunk_file.name.endswith(".raw.json"):
            continue
        for candidate_payload in json.loads(chunk_file.read_text(encoding="utf-8")):
            if int(candidate_payload["candidate_id"]) == TARGET_CANDIDATE_ID:
                return CandidateQuadruplet.from_dict(candidate_payload)
    raise RuntimeError(f"candidate {TARGET_CANDIDATE_ID} not found")


def main() -> None:
    config_store = ConfigStore()
    config_store.load(
        base_config_path=SETTINGS_DIR / "config.yml",
        ner_config_path=SETTINGS_DIR / "ner_config.yml",
        prompt_config_path=SETTINGS_DIR / "ner_prompts.json",
    )

    llm_client = LLMClient(
        LLMConfig(
            base_url=config_store.llm_config["base_url"],
            api_key=_resolve_api_key(config_store.llm_config),
            model_name=config_store.llm_config["model_name"],
            temperature=float(config_store.llm_config["temperature"]),
            max_tokens=int(config_store.llm_config["max_tokens"]),
            max_retries=1,
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

    grouped_set = _load_target_group()
    candidate = _load_target_candidate()

    refiner_cfg = config_store.prompt_config("granularity_refiner")
    prompt = render_prompt(
        refiner_cfg["prompt_template"],
        grouped_candidate_set_json=serialize_for_prompt(grouped_set),
        candidate_pool_json=serialize_for_prompt([candidate]),
    )

    client = llm_client._client  # raw openai SDK client
    cfg = llm_client._config

    kwargs = {
        "model": cfg.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ner_output",
                "schema": refiner_cfg["output_schema"],
                "strict": True,
            },
        },
    }

    print("=" * 80)
    print("Calling OpenRouter with the failing refiner input...")
    print("=" * 80)
    response = client.chat.completions.create(**kwargs)

    print("=" * 80)
    print("response.model_dump() (full Pydantic dump):")
    print("=" * 80)
    print(json.dumps(response.model_dump(), indent=2, default=str, ensure_ascii=False))

    print("=" * 80)
    print("response.model_extra (vendor-specific extras OpenAI SDK collected):")
    print("=" * 80)
    print(json.dumps(getattr(response, "model_extra", None), indent=2, default=str, ensure_ascii=False))

    choice = response.choices[0]
    print("=" * 80)
    print("choices[0].model_extra:")
    print("=" * 80)
    print(json.dumps(getattr(choice, "model_extra", None), indent=2, default=str, ensure_ascii=False))

    print("=" * 80)
    print("choices[0].message.model_extra:")
    print("=" * 80)
    print(json.dumps(getattr(choice.message, "model_extra", None), indent=2, default=str, ensure_ascii=False))

    print("=" * 80)
    print("choices[0].message.refusal:")
    print("=" * 80)
    print(repr(getattr(choice.message, "refusal", "<no attribute>")))

    llm_client.close()


if __name__ == "__main__":
    main()
