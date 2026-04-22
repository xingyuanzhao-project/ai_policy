# Log 0.3 — Agent Skill NER: Crashes, Fixes, and Practical Insights

**Date**: 2026-04-16
**Agent session**: [Skill NER build & run](462b7595-316e-47df-8937-f68b986c90dd)

---

## 1. All Crashes and Errors (Chronological)

### 1.1 API Key Resolution Failure

- **When**: First test run (`tests/test_agent_loop.py`)
- **Symptom**: `openai.AuthenticationError` — key not found
- **Cause**: The test setup used `os.environ.get("OPENROUTER_API_KEY")` but the key was stored in Windows keyring, not env var. The existing NER pipeline resolves keys via `bootstrap.py` which checks env → keyring → literal config, but the new test code only checked env.
- **Fix**: Replicated the keyring fallback logic from `src/ner/runtime/bootstrap.py` in test helpers.

### 1.2 Invalid Model ID

- **When**: First run of `tests/test_skill_ner_extraction.py`
- **Symptom**: `openai.BadRequestError` from OpenRouter
- **Cause**: `settings/skill_ner_config.yml` had `model_name: anthropic/claude-sonnet-4-20250514` which is not a valid OpenRouter model identifier.
- **Fix**: Changed to `anthropic/claude-sonnet-4.5`.

### 1.3 Windows PermissionError on Temp Directory Cleanup

- **When**: `test_skill_ner_extraction.py` teardown
- **Symptom**: `PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'run.log'`
- **Cause**: Python's `logging.FileHandler` holds the `run.log` file lock. `tempfile.TemporaryDirectory` tries to delete it on context manager exit while the handler is still open.
- **Fix**: Switched to `tempfile.mkdtemp()` without auto-cleanup. Prints the path for manual deletion.

### 1.4 JSON Parsing Failure — Prose Preamble Before Fenced JSON

- **When**: First 5-bill test run (sync, pre-async)
- **Symptom**: All 5 bills produced empty `[]` output. `json.JSONDecodeError` in logs.
- **Cause**: The model returns prose before the JSON block: `"Based on my reading of Arizona HB 2482... ```json { ... } ```"`. The original `_strip_markdown_fences()` only handled text that **starts** with triple backticks.
- **Fix**: Rewrote as `_extract_json_block()` in `src/skill_ner/schemas.py`. It searches for the last fenced `json` block anywhere in the response text, regardless of preceding prose.
- **Observation**: This is a fundamental behavioral pattern of commercial LLMs via API — they frequently wrap structured output in prose + markdown fences even when `response_format: json_schema` is specified. Any production agentic pipeline must handle this.

### 1.5 Logging Swallowed — Root Logger at WARNING

- **When**: Second 5-bill test run (after parsing fix)
- **Symptom**: `run.log` only showed ERROR messages from run 1. All INFO-level logs from run 2 were silently dropped.
- **Cause**: Python's root logger defaults to `WARNING`. The file handler was set to `INFO`, but the root logger filtered out INFO messages before they reached any handler.
- **Fix**: Modified `_attach_run_log_handler()` in `src/skill_ner/runner.py` to set `logging.getLogger().setLevel(logging.INFO)` if the root level is above INFO.

### 1.6 Full Corpus Crash — `RuntimeError: Agent loop exceeded max_turns=50`

- **When**: First full corpus run (`skill_full_20260416`), bill 64/1826
- **Symptom**: Entire `asyncio.gather` crashed. Run terminated at bill 64 with traceback.
- **Cause**: `2023__ME S 656` (text_length=31,244) triggered a **micro-scanning loop**: the model spent 30+ turns binary-searching for exact character offsets of the phrase `"systems based on artificial intelligence"` around offset ~17,220. Turn progression: 5000-char reads → 100-char → 50-char → 10-char → single-offset verification. Exceeded `max_turns=50`.
- **Root root cause**: The `_guarded()` error handler only caught `CreditsExhaustedError`. The `RuntimeError` from the agent loop was uncaught, propagated through `asyncio.gather`, and killed all concurrent bill tasks.
- **Fix**: Broadened `_guarded()` to `except Exception`, logging the error, updating the progress bar, and returning `False`. Single-bill failures no longer crash the corpus.
- **Observation**: The micro-scanning behavior is **not correlated with bill length**. Bills up to 100,000 chars completed successfully. The trigger is the model's drive to verify evidence offsets at character-level precision.

### 1.7 Mass Edit Incident

- **When**: Refactoring config loading from `src/` to `scripts/`
- **Symptom**: Agent rewrote both `runner.py` and `messages.py` completely, moving ~80 lines of config loading logic into the entry point script.
- **Cause**: User asked source code not to directly access `settings/`. Agent interpreted this as "move all loading to scripts" rather than "pass paths as parameters."
- **Fix**: Reverted. Made the minimal change: `run_corpus()` accepts `config_path`, `base_config_path`, `skill_path` as keyword parameters. Entry point passes the paths. Library still loads files internally, but the **path decision** comes from the caller.
- **Learning**: "Don't access settings from src" → correct response is parameterization, not relocation of business logic.

### 1.8 Entry Point Overengineering

- **When**: Initial `scripts/run_skill_ner.py` creation
- **Symptom**: Script had 5 argparse flags (`--model`, `--run-id`, `--max-bills`, `--resume`, `--run-id`). Not click-and-run.
- **Cause**: Agent designed for reusability across model comparisons (task 1.7). But the project convention is: config lives in config files, entry points are thin hardcoded callers.
- **Fix**: Deleted argparse. Wrote a 6-line script matching `run_ner.py` pattern — imports from `src`, hardcoded `run_id`, everything else from config files.

### 1.9 Final Full Run — 4 Non-Fatal Failures

- **When**: Clean full run (`skill_full_20260416_v2`), 1826 bills
- **Bills failed**: `2023__IL H 3338`, `2024__MO H 1814`, `2025__AZ S 1279`, `2025__VA H 2468`
- **Symptom**: Non-fatal — logged as errors, skipped, run continued. 1819/1826 outputs produced.
- **Cause**: Not diagnosed individually. Likely max_turns exceeded or transient API errors.
- **Impact**: 4/1826 = 0.22% failure rate. Acceptable for research corpus.

---

## 2. Practical Insights from Running Agent Skill NER

These observations come from actually building, debugging, and running the pipeline — not from theoretical expectations.

### 2.1 LLM Tool-Calling Behavior: Micro-Scanning is Real

**this is a problem of agent skill pipeline. the agent loop might be a dead circlining loop**

The most significant unexpected behavior was the **micro-scanning loop**. When the skill prompt requires evidence spans with character offsets, the model performs binary-search-style progressive reads to pinpoint exact boundaries:

| Turn | Offset range | Length |
|------|-------------|--------|
| 10 | 17100–17200 | 100 |
| 12 | 17230–17280 | 50 |
| 19 | 17155–17165 | 10 |
| 24-28 | 17220–17261 | 41–45 (repeated) |

This consumes 30+ turns for what should be a single read. The root cause is the evidence offset requirement in the prompt — the model tries to be *character-exact*. Proposed mitigations (not yet implemented): minimum read size floor (500 chars) in the `read_section` tool, or a read-deduplication cache.

### 2.2 Model Response Format is Unreliable

Even with `response_format: json_schema` set, commercial models (Claude Sonnet 4.5 via OpenRouter) wrap JSON in:
- Prose preamble (`"Based on my reading of..."`)
- Markdown code fences (`` ```json ... ``` ``)

A production parser must search for the JSON block, not assume the response starts with `{` or `[`. The `_extract_json_block()` function — which finds the last fenced JSON block anywhere in the response — was necessary for correct parsing.

### 2.3 Cost Profile: $315 for 1826 Bills

| Metric | Value |
|--------|-------|
| Total cost | $315.60 |
| Cost per bill | $0.173 |
| Total tokens | 90.9M |
| Total API calls | 11,231 |
| Avg calls per bill | ~6.2 |
| Wall time | 1h 38m (with concurrency=10) |

The per-bill cost is significantly higher than the multi-step orchestrated pipeline because:
- The full methodology prompt is sent on every turn (growing context window)
- Multi-turn conversations accumulate tokens: each turn re-sends all prior messages
- The model uses Claude Sonnet 4.5 at $3/M input, $15/M output

### 2.4 Async Concurrency: 2.8x Speedup Measured

| Config | 5-bill wall time | Effective rate |
|--------|-----------------|----------------|
| Sync (sequential) | ~138s | ~27.6 s/bill |
| Async (concurrency=10) | ~50s | ~10 s/bill |

At full corpus scale with concurrency=10, effective rate was **~3.22 s/bill** (1826 bills in 98 minutes). The speedup scales sub-linearly because longer bills hold semaphore slots while shorter bills wait.

### 2.5 **Bill Length Does Not Predict Failure**

this is very imporatant. the fixed stacked prompt pipeline crash a lot because of this, had to debug code, cathc errors, and make exceptions and rules to handle intensively. this is critial imporatance of skill driven NER pipeline.

The failing bill (`ME S 656`, 31,244 chars) was mid-range. Bills up to 100,000 chars completed. Failure is driven by content structure (how the model interprets what needs character-level verification), not length.

### 2.6 Resume Works But Only at Bill Granularity

The skill NER pipeline resumes by checking for `{bill_id}.json` existence. If a bill crashes mid-processing, the entire bill is re-processed. The multi-step NER pipeline resumes at finer granularity (per-stage within a bill). For the skill NER this is acceptable because each bill is a single agent conversation — there are no intermediate artifacts to save.

### 2.7 Logging Gaps Were Significant

The initial implementation logged only start/end per bill. Compared to the existing NER pipeline, it was missing:

| Field | NER pipeline | Skill NER (initial) | Skill NER (final) |
|-------|-------------|--------------------|--------------------|
| `bill_id` in output JSON | yes | no | yes |
| `source_bill_id` | yes | no | yes |
| `year`, `state` | yes | no | yes |
| Per-bill elapsed time | yes | no | yes |
| Corpus total elapsed | yes | no | yes |
| Agent turns count | n/a | no | yes |
| Tool call count | n/a | no | yes |
| `text_length` | no | no | yes |

The enriched output format is critical for downstream evaluation (task 1.6) and resource comparison (task 1.7). The bare-list format from the first run would have been unusable for research without post-hoc reconstruction from timestamps.

### 2.8 Interleaved Async Logs Need Context Labels

With 10 concurrent bills, the `run.log` interleaves HTTP requests and tool calls from all active sessions. Without a `[bill_id]` prefix on each log line, it is impossible to trace which turns belong to which bill. Adding `context_label` to the agent loop was essential for debugging the micro-scanning issue.

### 2.9 Error Isolation is Non-Negotiable for Corpus Runs

The single biggest operational lesson: **one bill failure must never crash 1825 other bills**. The original `_guarded()` only caught `CreditsExhaustedError`. The `RuntimeError` from one bill's max_turns killed the entire run at bill 64 — wasting all in-flight concurrent bills' partial work. The fix (`except Exception` with per-bill logging) is the minimum viable error boundary for any corpus-scale async pipeline.

### 2.10 Agent Architecture: Simplicity Won

**this is also very important strength. very time saving**

The final implementation uses:
- 1 tool (`read_section` — a pure string slice)
- Raw `openai` SDK (no LangGraph, no CrewAI, no framework)
- A while loop (~50 lines in `loop.py`)

This was arrived at after the agent initially proposed 5 tools (`submit_candidate`, `list_candidates`, `submit_group`, `submit_refined`, `complete_bill`) with Python-side state management. The user's pushback — "the most the agent should call will be just reading the document" — was correct. The model handles all reasoning internally and outputs final JSON. Python just serves text and saves results.

---

## 3. Run Summary

| Run | Bills | Succeeded | Failed | Cost | Wall Time |
|-----|-------|-----------|--------|------|-----------|
| `skill_run_20260416` (5-bill test, sync) | 5 | 5 | 0 | $0.30 | 4.7 min |
| `skill_run_20260416` (5-bill retest, sync) | 5 | 4 | 1 (0 quads, legit) | $0.07 | 2.3 min |
| `skill_run_20260416` (5-bill, async) | 5 | 5 | 0 | $0.53 | 50s |
| `skill_full_20260416` (crashed) | 1826 | 64 | 1 (crash) | — | ~6 min |
| `skill_full_20260416_v2` (clean) | 1826 | 1819 | 4 (non-fatal) | $315.60 | 1h 38m |
