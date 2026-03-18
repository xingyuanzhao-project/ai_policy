# NER Implementation Plan

## Step 1 - Build the Framework, Contracts, and Stores

### Module: Project Skeleton
- [ ] Create `src/ner/` as the bounded context for the proposed NER multi-agent system.
- [ ] Split the code into `agents/`, `orchestration/`, `storage/`, `runtime/`, and `schemas/`.
- [ ] Keep NER system code separate from analysis scripts and paper-writing materials.

### Module: Shared Agent Contract
- [ ] Define one common agent interface with explicit `input schema`, `output schema`, `raw response`, and `parsed response`.
- [ ] Implement shared schema objects for `BillRecord`, `ContextChunk`, `SpanRef`, `CandidateQuadruplet`, `GroupedCandidateSet`, `RefinedQuadruplet`, and `RefinementArtifact`.
- [ ] Define `BillRecord` as the canonical raw-bill schema and `ContextChunk` as the canonical derived inference-unit schema.
- [ ] Fix one canonical field order for all field-wise outputs: `entity`, `type`, `attribute`, `value`.
- [ ] Freeze `field_score_matrix` as a required schema field of `GroupedCandidateSet`; row order follows `candidate_ids` and column order follows `field_order`.
- [ ] Freeze one canonical refinement relation label set for `RefinementArtifact`: `support`, `overlap`, `conflict`, `duplicate`, `refinement`.
- [ ] Validate every parsed agent output against the declared schema before handing it to the next stage.
- [ ] Do not allow `bill -> chunk -> agent` handoffs to use free-form dicts when a shared schema exists.

### Module: State Design
- [ ] Keep these artifact types distinct throughout the pipeline: `bill records`, `context chunks`, `candidate quadruplets`, `grouped candidate sets`, `refined quadruplets`, and `refinement artifacts`.
- [ ] Assign stable ids for `bill_id`, `chunk_id`, `candidate_id`, `group_id`, and `refined_id`.
- [ ] Ensure every `ContextChunk` keeps a stable `bill_id` reference to its source `BillRecord`.
- [ ] Ensure `GroupedCandidateSet.candidate_ids` and `GroupedCandidateSet.field_order` are the only legal row/column interpretation for `field_score_matrix`.
- [ ] Use candidate ids to recover candidate content and evidence rather than duplicating evidence blobs inside grouped sets.
- [ ] Treat `ContextChunk` as a derived artifact created by `Inference Unit Builder`, not as a raw corpus record loaded by `Corpus Store`.

### Module: Config Store
- [ ] Store prompt templates, model settings, and chunking settings in versioned config files.
- [ ] Validate required config values at startup before any agent runs.
- [ ] Keep per-run config snapshots so one run can be reproduced against the exact same prompts and parameters.

### Module: Corpus Store
- [ ] Read bill metadata and raw bill text from local project data.
- [ ] Preserve bill ids and raw bill text as first-class fields.
- [ ] Do not mix raw corpus records with derived `ContextChunk` artifacts.

### Module: Artifact Store
- [ ] Persist zero-shot candidate pools keyed by run id, bill id, and chunk id.
- [ ] Persist grouped candidate sets with their `field_order` and `field_score_matrix` keyed by run id and bill id.
- [ ] Persist optional refinement artifacts separately from final refined outputs.
- [ ] Keep raw LLM responses and parsed artifacts side by side for debugging and recovery.

### Module: Final Output Store
- [ ] Store refined bill-level quadruplets in a normalized format.
- [ ] Preserve `source_group_id`, `source_candidate_ids`, and field-linked evidence in final outputs.
- [ ] Keep final outputs separate from candidate pools, grouped sets, and refinement artifacts.

### Module: Inference Unit Builder
- [ ] Convert each raw bill into `ContextChunk` objects that can be processed by the `Zero-shot Annotator`.
- [ ] Give every derived chunk a stable `chunk_id` and preserve source offsets for all extracted spans.
- [ ] Emit schema-valid `ContextChunk` objects that retain the source `bill_id`.
- [ ] Keep `bill record`, derived `ContextChunk`, and `final refined output` as separate data objects.

### Module: Local LLM Binding
- [ ] Bind the framework to the existing local LLM runtime rather than adding a new remote-model dependency.
- [ ] Expose one shared model client that all three agents use for prompt execution.
- [ ] Centralize model settings, prompt settings, and structured-output parsing behavior in config rather than inside agent files.

### Test: Contract and Store Round-Trip
- [ ] Verify that `BillRecord` and `ContextChunk` can round-trip through storage without losing `bill_id`, `chunk_id`, or source offsets.
- [ ] Verify that derived `ContextChunk` objects can be serialized and deserialized without losing `chunk_id` or source offsets.
- [ ] Verify that candidate, grouped, refined, and refinement artifacts can round-trip through storage without shape drift.
- [ ] Verify that canonical field order `entity`, `type`, `attribute`, `value` remains consistent between shared contracts and stored score matrices.
- [ ] Verify that `field_score_matrix` is stored and reloaded as a declared `GroupedCandidateSet` schema field rather than as an ad-hoc array, dict, or dataframe.
- [ ] Verify that every non-null relation entry in `RefinementArtifact` uses the canonical relation label set.
- [ ] Verify that raw corpus records stay distinct from derived context chunks across store boundaries.
- [ ] Verify that the `Zero-shot Annotator` receives schema-valid `ContextChunk` objects rather than ad-hoc dict payloads.

## Step 2 - Build and Validate the Agents

### Module: Shared Agent Components
- [ ] Implement one shared prompt execution wrapper used by all three agents.
- [ ] Implement one shared structured-output parser and schema-validation path used by all three agents.
- [ ] Implement shared helpers for stable id assignment and artifact serialization.
- [ ] Reuse the shared schema objects from `Shared Agent Contract` rather than letting agents define local ad-hoc payloads.

### Module: Zero-shot Annotator
- [ ] Implement the `Zero-shot Annotator` to read one context chunk at a time.
- [ ] Make the agent emit `CandidateQuadruplet` objects with fields allowed to be missing.
- [ ] Make the agent attach field-linked `SpanRef` evidence to every emitted candidate.
- [ ] Ensure the agent proposes candidates only; it must not finalize grouped relations or refined outputs.

### Module: Eval Assembler
- [ ] Implement the `Eval Assembler` to read the candidate quadruplet pool produced by the `Zero-shot Annotator`.
- [ ] Make the agent group related candidates into `GroupedCandidateSet` objects.
- [ ] Make the agent emit one `field_score_matrix` per grouped set with row order aligned to `candidate_ids` and column order aligned to `field_order`.
- [ ] Ensure the agent groups and scores candidates only; it must not finalize `support / overlap / conflict / duplicate / refinement`.

### Module: Granularity Refiner
- [ ] Implement the `Granularity Refiner` to read `GroupedCandidateSet` objects plus referenced `CandidateQuadruplet` objects with field-linked evidence.
- [ ] Make the agent emit `RefinedQuadruplet` objects with preserved evidence and source ids.
- [ ] Make the agent emit `RefinementArtifact` only as an optional refinement-side artifact, not as the primary final output.
- [ ] Ensure every non-null relation entry in `RefinementArtifact` uses the canonical relation label set.
- [ ] Ensure the agent refines grouped candidates into final structured outputs without re-running raw chunk annotation.

### Test: Agent Validation
- [ ] Verify that each of the three agents can run with the current local LLM and current config.
- [ ] Verify that `Zero-shot Annotator` can emit candidates with partially missing fields without breaking schema validation.
- [ ] Verify that `Eval Assembler` emits grouped sets whose score-matrix row order matches `candidate_ids` and whose column order matches `field_order`.
- [ ] Verify that `Granularity Refiner` emits refined quadruplets with source ids and field-linked evidence.
- [ ] Verify that `Granularity Refiner` emits `RefinementArtifact` objects whose non-null relation entries use only the canonical relation label set.

## Step 3 - Build the Proposed Orchestrator

### Module: Stage Ordering
- [ ] Implement explicit orchestration order: `zero-shot annotator -> eval assembler -> granularity refiner`.
- [ ] Keep stage ordering explicit in orchestration code rather than implicit inside agent logic.
- [ ] Make the `Granularity Refiner` read grouped candidates and referenced candidates rather than re-reading raw chunks as its primary input.

### Module: Failure and Resume Control
- [ ] Stop a stage when schema parsing fails instead of silently continuing with invalid artifacts.
- [ ] Persist stage outputs immediately so one failed bill does not require a full restart.
- [ ] Support rerunning one bill, one chunk, or one grouped set for debugging.
- [ ] Keep resume logic stage-aware so a completed annotation stage does not need to be rerun when only refinement fails.

## Step 4 - Build the Agent Main Loop

### Module: Runtime Bootstrap
- [ ] Load configs, initialize the local LLM client, and open all stores.
- [ ] Read the bill corpus and build the bill/chunk worklist.
- [ ] Register the shared field order and schema validators before starting inference.

### Module: Target Inference Loop
- [ ] Run the `Zero-shot Annotator` over each context chunk and aggregate chunk-level candidates into a bill-level candidate pool.
- [ ] Run the `Eval Assembler` on the bill-level candidate pool to produce grouped candidate sets and score matrices.
- [ ] Run the `Granularity Refiner` on each grouped candidate set using candidate ids to recover the referenced candidates and evidence.
- [ ] Persist outputs after each stage rather than waiting until the end of the whole bill.

### Module: Bill-Level Assembly
- [ ] Merge chunk-level candidates into one bill-level candidate pool before grouping.
- [ ] Merge refined group outputs into bill-level refined quadruplets.
- [ ] Preserve mappings from each final quadruplet back to its source group, source candidates, and source spans.
- [ ] Keep chunk boundaries available for debugging even after bill-level assembly.

### Module: Runtime Entry Points
- [ ] Provide one entry point for full-corpus runs.
- [ ] Provide one entry point for single-bill debugging.
- [ ] Provide one entry point for single-group or single-chunk debugging after annotation.
- [ ] Ensure all entry points use the same config loading and orchestrator code path.

## Step 5 - Verify the System End to End

### Test: Pipeline Validation
- [ ] Run the full three-stage pipeline on a small set of existing bills.
- [ ] Confirm that stage transitions use only declared artifacts and do not rely on hidden global state.
- [ ] Confirm that the pipeline can complete end-to-end without manual artifact repair.
- [ ] Confirm that optional refinement artifacts are produced only by the refinement stage.

### Test: Traceability Validation
- [ ] Confirm that every final refined quadruplet can be traced back to its `source_group_id` and `source_candidate_ids`.
- [ ] Confirm that every field-level evidence span in final outputs maps back to a valid `SpanRef` with `chunk_id`.
- [ ] Confirm that grouped candidate sets do not duplicate evidence payloads when evidence can be recovered from referenced candidates.
- [ ] Confirm that refinement decisions can be inspected alongside preserved evidence and optional refinement artifacts.
