"""Corpus-level execution for the skill-driven NER pipeline.

- Config file paths are passed in by the caller (entry point); defaults
  match the standard ``settings/`` layout so library tests still work.
- Iterates the corpus, running the agent loop per bill via ``src/agent/``.
- Persists outputs, raw conversations, config snapshots, and usage stats.
- Does not define tools, build prompts, or parse output schemas.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

from src.agent import ToolRegistry, UsageStats, make_read_section_tool, run_tool_loop
from src.agent.loop import CreditsExhaustedError, run_tool_loop_async
from src.ner.schemas.artifacts import BillRecord
from src.ner.storage.corpus_store import CorpusStore

from .messages import build_messages, load_skill
from .schemas import parse_agent_response

logger = logging.getLogger(__name__)

_RUN_LOG_FORMAT = "%(asctime)s  %(name)s  %(levelname)s  %(message)s"

_USAGE_SUMMARY_NUMERIC_KEYS = (
    "total_calls",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_tokens",
    "total_reasoning_tokens",
    "total_cached_tokens",
    "total_cost_usd",
    "total_elapsed_ms",
)


def run_corpus(
    *,
    project_root: Path,
    config_path: str,
    base_config_path: str,
    skill_path: str,
    run_id: str | None = None,
    max_bills: int | None = None,
    resume: bool = False,
    bill_id_filter: list[str] | None = None,
) -> Path:
    """Run skill-driven NER on the corpus.

    Args:
        project_root: Project root for resolving config paths.
        config_path: Path to skill NER config, relative to project root.
        base_config_path: Path to shared project config, relative to project root.
        skill_path: Path to the skill methodology file, relative to project root.
        run_id: Run identifier. Auto-generated when omitted.
        max_bills: Maximum number of bills to process (cost control).
        resume: When True, skip bills whose output files already exist.
        bill_id_filter: When provided, only process bills whose IDs are in
            this list. Applied before max_bills.

    Returns:
        Path to the run directory containing all outputs.
    """

    config = _load_config(project_root, config_path)
    effective_model = config["model_name"]
    effective_run_id = run_id or f"skill_{uuid.uuid4().hex[:12]}"

    run_dir = _init_run_dir(project_root, config, effective_run_id)
    run_log_handler = _attach_run_log_handler(run_dir)

    logger.info(
        "Skill NER run starting: run_id=%s  model=%s  max_bills=%s  resume=%s",
        effective_run_id,
        effective_model,
        max_bills,
        resume,
    )

    # Snapshot config and skill
    _snapshot_config(project_root, config, run_dir)
    skill_content = load_skill(project_root, skill_path)
    (run_dir / "skill_snapshot.md").write_text(skill_content, encoding="utf-8")

    # API clients (sync for preflight, async for concurrent bill processing)
    llm_config = _load_llm_config(project_root, base_config_path)
    api_key = _resolve_api_key(llm_config)
    client_kwargs = dict(
        base_url=llm_config["base_url"],
        api_key=api_key,
        max_retries=config.get("max_retries", 2),
        timeout=config.get("request_timeout_seconds", 120),
    )
    sync_client = OpenAI(**client_kwargs)
    async_client = AsyncOpenAI(**client_kwargs)

    # Preflight (sync -- single request before fan-out)
    preflight_result = _run_preflight(sync_client, effective_model)
    (run_dir / "preflight.json").write_text(
        json.dumps(preflight_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Corpus
    corpus_path = _resolve_project_path(
        project_root,
        _load_base_config(project_root, base_config_path).get("input_path", ""),
    )
    corpus_store = CorpusStore(corpus_path)
    bills = corpus_store.load()
    if bill_id_filter is not None:
        allowed = set(bill_id_filter)
        bills = [b for b in bills if b.bill_id in allowed]
    if max_bills is not None:
        bills = bills[:max_bills]

    concurrency = config.get("concurrency", 10)
    logger.info(
        "Processing %d bill(s)  concurrency=%d", len(bills), concurrency,
    )

    usage_stats = UsageStats()
    outputs_dir = run_dir / "outputs"
    raw_dir = run_dir / "raw_responses"
    outputs_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)

    # Filter bills for resume before dispatching
    if resume:
        pending_bills = []
        for bill in bills:
            if (outputs_dir / f"{bill.bill_id}.json").exists():
                logger.info("Resuming: skip %s (output exists)", bill.bill_id)
            else:
                pending_bills.append(bill)
    else:
        pending_bills = list(bills)

    corpus_t0 = time.perf_counter()
    try:
        processed = asyncio.run(
            _run_corpus_async(
                client=async_client,
                model=effective_model,
                config=config,
                skill_content=skill_content,
                bills=pending_bills,
                usage_stats=usage_stats,
                outputs_dir=outputs_dir,
                raw_dir=raw_dir,
                concurrency=concurrency,
            )
        )
    except CreditsExhaustedError:
        logger.error("Credits exhausted. Stopping.")
        processed = 0
    finally:
        _persist_usage_summary(usage_stats, run_dir)
        logging.getLogger().removeHandler(run_log_handler)
        run_log_handler.close()

    corpus_elapsed = time.perf_counter() - corpus_t0
    logger.info(
        "Skill NER run complete: %d/%d bill(s) processed  "
        "total_elapsed=%.1fs  run_dir=%s",
        processed, len(pending_bills), corpus_elapsed, run_dir,
    )
    return run_dir


def _process_bill(
    *,
    client: OpenAI,
    model: str,
    config: dict[str, Any],
    skill_content: str,
    bill: BillRecord,
    usage_stats: UsageStats,
    outputs_dir: Path,
    raw_dir: Path,
) -> None:
    """Run the agent loop for a single bill and persist results (sync).

    Args:
        client: Connected OpenAI client.
        model: Model identifier for the completions API.
        config: Loaded skill_ner_config.yml dict.
        skill_content: The methodology prompt text.
        bill: The bill record to process.
        usage_stats: Shared usage accumulator.
        outputs_dir: Directory for canonical output files.
        raw_dir: Directory for raw conversation logs.
    """

    bill_t0 = time.perf_counter()

    max_chars = config.get("runtime", {}).get("max_bill_text_chars", 100_000)
    bill_text = bill.text[:max_chars] if len(bill.text) > max_chars else bill.text

    logger.info(
        "Processing bill %s  text_length=%d",
        bill.bill_id,
        len(bill_text),
    )

    registry = ToolRegistry()
    schema, handler = make_read_section_tool(bill_text)
    registry.register("read_section", handler, schema)

    messages = build_messages(skill_content, bill.bill_id, len(bill_text))

    final_response = run_tool_loop(
        client=client,
        model=model,
        messages=messages,
        tools=registry.definitions(),
        tool_executor=registry.execute,
        usage_stats=usage_stats,
        max_turns=config.get("max_tool_turns", 50),
        temperature=0.0,
        max_tokens=config.get("max_tokens", 16384),
    )

    bill_elapsed_s = time.perf_counter() - bill_t0

    raw_path = raw_dir / f"{bill.bill_id}.json"
    raw_path.write_text(
        json.dumps(messages, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    quadruplets = parse_agent_response(final_response)

    payload: dict[str, Any] = {
        "bill_id": bill.bill_id,
        "source_bill_id": bill.source_bill_id,
        "year": bill.year,
        "state": bill.state,
        "text_length": len(bill_text),
        "elapsed_seconds": round(bill_elapsed_s, 2),
        "quadruplets": quadruplets,
    }

    output_path = outputs_dir / f"{bill.bill_id}.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "Bill %s: extracted %d quadruplet(s)  elapsed=%.1fs",
        bill.bill_id,
        len(quadruplets),
        bill_elapsed_s,
    )


async def _run_corpus_async(
    *,
    client: AsyncOpenAI,
    model: str,
    config: dict[str, Any],
    skill_content: str,
    bills: list[BillRecord],
    usage_stats: UsageStats,
    outputs_dir: Path,
    raw_dir: Path,
    concurrency: int,
) -> int:
    """Fan out bill processing with bounded concurrency.

    Args:
        client: Connected async OpenAI client.
        model: Model identifier for the completions API.
        config: Loaded skill_ner_config.yml dict.
        skill_content: The methodology prompt text.
        bills: Bills to process (already filtered for resume).
        usage_stats: Shared usage accumulator (thread-safe).
        outputs_dir: Directory for canonical output files.
        raw_dir: Directory for raw conversation logs.
        concurrency: Maximum number of bills processed in parallel.

    Returns:
        Number of bills successfully processed.
    """

    semaphore = asyncio.Semaphore(concurrency)
    credits_exhausted = asyncio.Event()
    pbar = tqdm(total=len(bills), desc="Skill NER bills", unit="bill")

    async def _guarded(bill: BillRecord) -> bool:
        if credits_exhausted.is_set():
            pbar.update(1)
            return False
        async with semaphore:
            if credits_exhausted.is_set():
                pbar.update(1)
                return False
            try:
                await _process_bill_async(
                    client=client,
                    model=model,
                    config=config,
                    skill_content=skill_content,
                    bill=bill,
                    usage_stats=usage_stats,
                    outputs_dir=outputs_dir,
                    raw_dir=raw_dir,
                )
                pbar.update(1)
                return True
            except CreditsExhaustedError:
                credits_exhausted.set()
                logger.error(
                    "Credits exhausted during bill=%s. "
                    "Remaining bills will be skipped.",
                    bill.bill_id,
                )
                pbar.update(1)
                return False
            except Exception:
                logger.exception(
                    "Bill %s failed (non-fatal). Continuing.",
                    bill.bill_id,
                )
                pbar.update(1)
                return False

    results = await asyncio.gather(*[_guarded(bill) for bill in bills])
    pbar.close()
    return sum(results)


async def _process_bill_async(
    *,
    client: AsyncOpenAI,
    model: str,
    config: dict[str, Any],
    skill_content: str,
    bill: BillRecord,
    usage_stats: UsageStats,
    outputs_dir: Path,
    raw_dir: Path,
) -> None:
    """Async variant of :func:`_process_bill` for concurrent dispatch.

    Args:
        client: Connected async OpenAI client.
        model: Model identifier for the completions API.
        config: Loaded skill_ner_config.yml dict.
        skill_content: The methodology prompt text.
        bill: The bill record to process.
        usage_stats: Shared usage accumulator (thread-safe).
        outputs_dir: Directory for canonical output files.
        raw_dir: Directory for raw conversation logs.
    """

    bill_t0 = time.perf_counter()

    max_chars = config.get("runtime", {}).get("max_bill_text_chars", 100_000)
    bill_text = bill.text[:max_chars] if len(bill.text) > max_chars else bill.text

    logger.info(
        "Processing bill %s  text_length=%d",
        bill.bill_id,
        len(bill_text),
    )

    registry = ToolRegistry()
    schema, handler = make_read_section_tool(bill_text)
    registry.register("read_section", handler, schema)

    messages = build_messages(skill_content, bill.bill_id, len(bill_text))

    loop_result = await run_tool_loop_async(
        client=client,
        model=model,
        messages=messages,
        tools=registry.definitions(),
        tool_executor=registry.execute,
        usage_stats=usage_stats,
        max_turns=config.get("max_tool_turns", 50),
        temperature=0.0,
        max_tokens=config.get("max_tokens", 16384),
        context_label=bill.bill_id,
    )

    bill_elapsed_s = time.perf_counter() - bill_t0

    raw_path = raw_dir / f"{bill.bill_id}.json"
    raw_path.write_text(
        json.dumps(messages, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    quadruplets = parse_agent_response(loop_result.response)

    payload: dict[str, Any] = {
        "bill_id": bill.bill_id,
        "source_bill_id": bill.source_bill_id,
        "year": bill.year,
        "state": bill.state,
        "text_length": len(bill_text),
        "agent_turns": loop_result.turns,
        "agent_tool_calls": loop_result.tool_calls,
        "elapsed_seconds": round(bill_elapsed_s, 2),
        "quadruplets": quadruplets,
    }

    output_path = outputs_dir / f"{bill.bill_id}.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "Bill %s: extracted %d quadruplet(s)  turns=%d  "
        "tool_calls=%d  elapsed=%.1fs",
        bill.bill_id,
        len(quadruplets),
        loop_result.turns,
        loop_result.tool_calls,
        bill_elapsed_s,
    )


# ---------------------------------------------------------------------------
# Config and bootstrap helpers
# ---------------------------------------------------------------------------

def _load_config(project_root: Path, config_path: str) -> dict[str, Any]:
    """Load the skill NER config from the caller-specified path."""

    full_path = _resolve_project_path(project_root, config_path)
    with open(full_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_base_config(project_root: Path, base_config_path: str) -> dict[str, Any]:
    """Load the shared project config from the caller-specified path."""

    full_path = _resolve_project_path(project_root, base_config_path)
    with open(full_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_llm_config(project_root: Path, base_config_path: str) -> dict[str, Any]:
    """Extract the ``llm`` block from the shared project config."""

    base = _load_base_config(project_root, base_config_path)
    return base.get("llm", {})


def _resolve_api_key(llm_config: dict[str, Any]) -> str:
    """Resolve the API key from env var, keyring, or config literal.

    Resolution order mirrors ``src/ner/runtime/bootstrap._resolve_api_key``.

    Args:
        llm_config: The ``llm`` block from the loaded project config.

    Returns:
        Resolved API key string.

    Raises:
        RuntimeError: If no API key can be resolved.
    """

    api_key_env_var = str(llm_config.get("api_key_env_var", "")).strip()
    if not api_key_env_var:
        return str(llm_config.get("api_key", ""))

    api_key = os.environ.get(api_key_env_var, "").strip()
    if api_key:
        return api_key

    keyring_service = str(llm_config.get("keyring_service", "")).strip()
    keyring_username = str(llm_config.get("keyring_username", "")).strip()
    if keyring_service and keyring_username:
        try:
            import keyring as _keyring

            secret = _keyring.get_password(keyring_service, keyring_username)
            if secret and secret.strip():
                return secret.strip()
        except Exception:
            pass

    raise RuntimeError(
        f"API key could not be resolved: environment variable "
        f"'{api_key_env_var}' is not set and no keyring secret was found "
        f"(service='{keyring_service}', username='{keyring_username}')"
    )


def _init_run_dir(
    project_root: Path,
    config: dict[str, Any],
    run_id: str,
) -> Path:
    """Create the run directory under the configured storage base_dir.

    Args:
        project_root: Project root for resolving relative paths.
        config: Loaded skill_ner_config dict.
        run_id: Stable run identifier.

    Returns:
        Path to the created run directory.
    """

    base_dir = _resolve_project_path(
        project_root,
        config.get("storage", {}).get("base_dir", "data/skill_ner_runs"),
    )
    run_dir = base_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _snapshot_config(
    project_root: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> None:
    """Save a snapshot of the effective config to the run directory."""

    snapshot_path = run_dir / "config_snapshot.yml"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def _run_preflight(client: OpenAI, model: str) -> dict[str, Any]:
    """Run a minimal health probe to verify the model is reachable.

    Args:
        client: Connected OpenAI client.
        model: Model identifier to probe.

    Returns:
        Dict with probe results for persistence.
    """

    logger.info("Preflight check: model=%s", model)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Return a JSON object with exactly one field named "
                    "'status' and set it to the string 'ok'."
                ),
            }
        ],
        temperature=0.0,
        max_tokens=32,
    )
    content = response.choices[0].message.content or ""
    logger.info("Preflight response: %s", content)
    return {
        "model": model,
        "raw_probe_response": content,
    }


def _persist_usage_summary(usage_stats: UsageStats, run_dir: Path) -> None:
    """Write cumulative usage stats, merging with any previous session.

    Args:
        usage_stats: The session's usage accumulator.
        run_dir: Run directory for the usage file.
    """

    session_summary = usage_stats.summary_dict()
    summary_path = run_dir / "usage_summary.json"

    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                previous = json.load(f)
            for key in _USAGE_SUMMARY_NUMERIC_KEYS:
                session_summary[key] = previous.get(key, 0) + session_summary[key]
            session_summary["total_cost_usd"] = round(
                session_summary["total_cost_usd"], 8
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(
                "Could not merge previous usage_summary.json; overwriting"
            )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(session_summary, f, indent=2, ensure_ascii=False)

    logger.info(
        "Usage summary persisted: calls=%d  tokens=%d  cost=$%.6f  file=%s",
        session_summary["total_calls"],
        session_summary["total_tokens"],
        session_summary["total_cost_usd"],
        summary_path,
    )


def _attach_run_log_handler(run_dir: Path) -> logging.FileHandler:
    """Attach a file handler to the root logger for the run directory.

    Args:
        run_dir: Run directory where the log file is created.

    Returns:
        The handler instance for later removal.
    """

    log_path = run_dir / "run.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_RUN_LOG_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    return handler

def _resolve_project_path(project_root: Path, configured_path: str) -> Path:
    """Resolve a configured path relative to the project root.

    Args:
        project_root: Project root for resolving relative paths.
        configured_path: Path string from config.

    Returns:
        Absolute path for the configured location.
    """

    candidate = Path(configured_path)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate

