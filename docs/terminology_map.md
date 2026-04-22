# Terminology Map

Purpose: lock in the user's terminology choices so agent drafts do not drift. Each entry lists the user's original word, its concrete meaning as used in the source docs, and the rewordings introduced by me (the agent) in `docs/mpsa_draft_v1.md` and the lit_rev....mds files. The user can review each entry and decide what to keep, revert, or rename; future agents must consult this file before rewording.

Format (per user request):

```
### user's word
**concrete meaning**
- my substitute 1 (`file:line`)
- my substitute 2 (`file:line`)
```

example:
i use the word "quadruplet" to describe the output of the NER pipeline.

Agent use word like combination, sets or others.

Source docs consulted: `docs/NER_method.md`, `docs/NER_agent_architecture_proposed_method.md`, `docs/NER_agent_architecture_references.md`, `docs/NER_plan.md`, `docs/apsa_2026_abstract_entity.md`, `docs/todo.md`.

Agent-authored files scanned for substitutes: `docs/mpsa_draft_outline.md` (prior outline draft; `mpsa_draft_v1.md` referenced in the paragraph above does not exist on disk), `docs/lit_rev_ner.md`, `docs/lit_rev_eval.md`, `docs/lit_rev_theory.md`, `docs/proposed_methods_eval.md` (added later; agent-drafted evaluation report), `docs/mpsa_draft_v6.md` (added later; current numbered MPSA draft). Note: `mpsa_draft_v6.md` is under active editing; line numbers cited below reflect the state of the file at the time of the mapping pass.

Entries below are included only when the agent introduced a substitute. Terms that the agent preserved verbatim (e.g. `granularity error`, `Regulatory Targets`, `Regulated Entities`, `CandidateQuadruplet` / `GroupedCandidateSet` / `RefinedQuadruplet` as schema names, `Self-Annotator` / `TRF Extractor` / `Demonstration Discriminator` / `Overall Predictor`, `CMAS`, `ContextChunk`) are not listed.

---

### quadruplet
**4-tuple `<entity, type, attribute, value>` with field-linked evidence spans; user's name for the canonical output unit of the NER pipeline (`docs/NER_agent_architecture_proposed_method.md:8`, also canonical field order fixed in `docs/NER_plan.md:14`).**
- `{entity, type, attribute, value}` — curly braces substituted for user's angle brackets (`docs/mpsa_draft_outline.md:125`, `docs/mpsa_draft_outline.md:134`, `docs/mpsa_draft_outline.md:160`, `docs/mpsa_draft_outline.md:165`, `docs/mpsa_draft_outline.md:170`, `docs/mpsa_draft_outline.md:182`, `docs/lit_rev_eval.md:217`)
- `(entity, type, attribute, value)` — parenthesised bracket form (`docs/proposed_methods_eval.md:8`)
- "entity, type, attribute, and value" — prose comma-plus-and form (abstract, intro, method, data) (`docs/mpsa_draft_v6.md:23`, `:35`, `:125`, `:169`)
- "The entity field ... The type field ... The attribute field ... The value field ..." — four-sentence expansion (`docs/mpsa_draft_v6.md:103`)
- "entity-attribute-value triplets" / "entity-attribute-value triplet extraction" — 3-field variant when paraphrasing the cited paper \cite{xu2025zeroshot} (`docs/mpsa_draft_v6.md:83`, `:89`)
- "triplets (target, entity, attribute)" — 3-field variant introduced once inline before reverting to quadruplet in the next bullet (`docs/mpsa_draft_outline.md:124`)

### Zero-shot Annotator
**Stage-1 agent: reads context chunks, proposes candidate quadruplets with field-linked evidence; fields may be missing at this stage (`docs/NER_agent_architecture_proposed_method.md:3-12`; also `docs/NER_plan.md:78-82`).**
- "Extraction of entities" — subsection heading (`docs/mpsa_draft_outline.md:157`)
- "Candidate annotation" — subsection heading under the multi-turn pipeline section (`docs/mpsa_draft_v6.md:119`)
- "the zero-shot annotator" — lowercase prose form referring to the stage-1 agent (`docs/mpsa_draft_v6.md:121`)

### Eval Assembler
**Stage-2 agent: groups related candidates into `GroupedCandidateSet` and emits a `field_score_matrix` per group; does not finalize relation labels (`docs/NER_agent_architecture_proposed_method.md:36-61`; also `docs/NER_plan.md:84-88`).**
- "Grouping and scoring candidates" — subsection heading (`docs/mpsa_draft_outline.md:162`)
- "Candidate grouping and scoring" — subsection heading (`docs/mpsa_draft_v6.md:123`)
- "the grouping and scoring agent" — prose agent name used in place of `Eval Assembler` (`docs/mpsa_draft_v6.md:125`)
- "Grouped Candidate Set" — spaced form of the schema class `GroupedCandidateSet` (`docs/mpsa_draft_v6.md:125`)

### Granularity Refiner
**Stage-3 agent: reads grouped sets plus referenced candidates with evidence, emits `RefinedQuadruplet` and an optional `RefinementArtifact` (a data-class name for the intermediate field-wise relation matrix — an output product of the stage, not a label for the stage itself) using the canonical relation labels `support / overlap / conflict / duplicate / refinement` (`docs/NER_agent_architecture_proposed_method.md:63-106`; also `docs/NER_plan.md:90-95`).**
- "Refinement of candidates sets" — subsection heading, applied per user comment (`docs/mpsa_draft_outline.md:167`)
- "Refinement" — shorter subsection heading (`docs/mpsa_draft_v6.md:127`)
- "the refiner" — prose agent name used in place of `Granularity Refiner` (`docs/mpsa_draft_v6.md:129`)
- "Refined Quadruplet" — spaced form of the schema class `RefinedQuadruplet` (`docs/mpsa_draft_v6.md:129`)
- "refinement artifact" — lowercase spaced form of the schema class `RefinementArtifact` (`docs/mpsa_draft_v6.md:129`)

comment: 
"Refinement of candidates" is not accurate name. call it ""Refinement of candidates sets". edit in all referenced files.
`RefinementArtifact` is not the accurate mapping. artifact is the intermediate product of the process, not the process itself. 

resolution: subsection heading renamed to "Refinement of candidates sets" at `docs/mpsa_draft_outline.md:167`. `RefinementArtifact` is preserved only as a data-class name for the emitted relation matrix, not as a label for the stage.

### multi-turn stateful extraction pipeline
**User's name for the fixed-plan three-stage NER method: the first of two proposed extraction methods; not conversational, not a baseline. Acceptable surface forms per user: `multi-turn stateful extraction pipeline`, `multi-turn NER pipeline`, `Multi-turn stateful pipeline`, `multi-turn pipeline`, `fixed-plan three-stage method`, `non agent orchestrated three-stage NER`.**
- "multi-turn NER pipeline" (`docs/mpsa_draft_outline.md:10`, `docs/mpsa_draft_outline.md:152`)
- "fixed-plan three-stage method" — applied per user comment; replaces prior "orchestrated three-stage baseline" / "orchestrated three-stage NER" / "orchestrated multi-turn NER" (`docs/mpsa_draft_outline.md:12`, `docs/mpsa_draft_outline.md:147`, `docs/mpsa_draft_outline.md:252`, `docs/mpsa_draft_outline.md:262`; `docs/lit_rev_eval.md:217`, `docs/lit_rev_eval.md:228`)
- "Multi-turn stateful pipeline" (`docs/mpsa_draft_outline.md:154`, `docs/mpsa_draft_outline.md:259`)
- "multi-turn pipeline" (`docs/mpsa_draft_outline.md:126`)
- "multi-turn stateful extraction pipeline" — applied per user comment; replaces prior "multi-turn conversational extraction pipeline" (`docs/apsa_2026_abstract_entity.md:7`)
- "Multi-turn stateful prompting" — applied per user comment; replaces prior "Multi-turn conversational prompting" (`docs/apsa_2026_abstract_entity.md:62`)
- `orchestrated` — bare standalone form used as an extractor run-ID / method name (`docs/proposed_methods_eval.md:1`, `:24`, `:41`, `:46`, `:54`, `:61`, `:67`, `:69`, `:70`, `:78`, `:97`, `:99`, `:103`). Also appears as residual at `docs/lit_rev_eval.md:349` ("the orchestrated extractor") and `docs/mpsa_draft_outline.md:211` ("orchestrated NER vs skill-driven NER"). Flag: the user's earlier comment above (item 2.1) asks the word "orchestrated" be avoided for this method; these occurrences pre- or post-date the clean-up pass that replaced the compound phrases. `docs/mpsa_draft_v6.md` does not use "orchestrated" for this method (the only "orchestration" occurrence at `:43` refers to generic pipeline-plumbing, not to the method label).
- "multi-turn stateful NER pipeline" — full descriptive form in the abstract (`docs/mpsa_draft_v6.md:23`)
- "fixed-plan multi-stage pipeline" — introduction / method-review form that swaps "three-stage" for the broader "multi-stage" (`docs/mpsa_draft_v6.md:47`, `:97`, `:263`)
- "fixed-plan multi-turn pipeline" — pipeline-section form compounding "fixed-plan" with "multi-turn" (`docs/mpsa_draft_v6.md:113`)
- "Multi-stage NER pipeline" — literature-review subsection heading (`docs/mpsa_draft_v6.md:81`)
- "multi-stage pipeline" — prose short form (`docs/mpsa_draft_v6.md:87`)
- "Multi-turn NER pipeline" — subsection heading (`docs/mpsa_draft_v6.md:115`)
- "multi-turn pipeline with scoring" / "multi-turn pipeline" — prose short forms (`docs/mpsa_draft_v6.md:88`, `:117`, `:205`, `:209`, `:215`, `:221`, `:233`, `:235`, `:241`, `:245`)
- "stateful three-stage design" — prose description of the pipeline's internal shape (`docs/mpsa_draft_v6.md:117`)
- "fixed-plan three-stage method" — reused from the already-accepted list, also present in v6 (`docs/mpsa_draft_v6.md:111`, `:231`)

comment:
1. "multi-turn conversational extraction pipeline" is not correct name. the edit agent carried stale memory, which is for other tasks. this is not conversational. 
acceptable names:
    multi-turn stateful extraction pipeline
    multi-turn NER pipeline
    Multi-turn stateful pipeline
    multi-turn pipeline

2. "orchestrated three-stage baseline" is a very conflating name, conflating 2 things:
    2.1 the orchestrated can be conflated with the agent self orchestrating pipeline. should avoid use the world orchestrated to describe this method.
    2.2 the three-stage pipeline is also a proposed method, so it is not a baseline.
acceptable names:
fixed-plan three-stage method
non agent orchestrated three-stage NER

resolution: (1) "conversational" dropped from `docs/apsa_2026_abstract_entity.md:7` and `docs/apsa_2026_abstract_entity.md:62`. (2) "orchestrated three-stage baseline" / "orchestrated three-stage NER" / "orchestrated multi-turn NER" replaced by "fixed-plan three-stage method" across `docs/mpsa_draft_outline.md` and `docs/lit_rev_eval.md`.

### Agentic Skills driven NER
**User's name for the single-agent, tool-calling approach that uses a skill file plus on-demand bill reading (`docs/todo.md:1`, section heading "From orchestrated NER to Agentic Skills driven NER"; contrast with the multi-turn pipeline above).**
- "skill driven agentic approach" (`docs/mpsa_draft_outline.md:10`, `docs/mpsa_draft_outline.md:126`, `docs/mpsa_draft_outline.md:172`, `docs/mpsa_draft_outline.md:260`)
- "skill-driven agentic NER" (`docs/mpsa_draft_outline.md:147`, `docs/mpsa_draft_outline.md:253`, `docs/mpsa_draft_outline.md:262`; `docs/lit_rev_eval.md:217`, `docs/lit_rev_eval.md:228`)
- "Agentic / skill-driven workflows" — section heading (`docs/mpsa_draft_outline.md:109`; `docs/lit_rev_ner.md:155`, also in the header summary `docs/lit_rev_ner.md:1`)
- `skill_driven` — underscore-lowercase identifier form used as an extractor run-ID / method name (`docs/proposed_methods_eval.md:1`, `:24`, `:42`, `:46`, `:55`, `:61`, `:67`, `:69`, `:70`, `:78`, `:97`, `:99`, `:103`)
- "skill-driven extractor" — hyphenated prose form paired with "orchestrated extractor" (`docs/lit_rev_eval.md:349`)
- "skill-driven agentic approach" — hyphenated variant of the existing "skill driven agentic approach" substitute (`docs/mpsa_draft_v6.md:23`)
- "skill-driven agent" — bare noun form for the method's running instance; dominant surface form in v6 (`docs/mpsa_draft_v6.md:133`, `:203`, `:209`, `:215`, `:221`, `:233`, `:235`, `:241`, `:245`)
- "agentic skill-driven pipeline" — word-order-reversed compound (`docs/mpsa_draft_v6.md:47`, `:97`, `:263`)
- "skill-driven agentic pipeline" — same compound in the canonical order (`docs/mpsa_draft_v6.md:113`)
- "Skill-driven agentic approach" — subsection heading under the Pipeline section (`docs/mpsa_draft_v6.md:131`)
- "Agentic and skill-driven workflows" — literature-review subsection heading (conjunction "and" in place of the outline's "/") (`docs/mpsa_draft_v6.md:91`)
- "skill-driven agentic NER" — heading form, reused from the already-accepted list (`docs/mpsa_draft_v6.md:111`)
- "skill-driven" — bare adjective in cross-method headings (`docs/mpsa_draft_v6.md:231`)

### llm as judge
**User's evaluation mechanism: use an LLM to decide semantic coverage because direct string match is unreliable at this granularity (`docs/todo.md:11`, "use llm as judge").**
- "LLM-as-judge" — hyphenated (`docs/mpsa_draft_outline.md:13`, `docs/mpsa_draft_outline.md:202`, `docs/mpsa_draft_outline.md:255`, `docs/mpsa_draft_outline.md:256`, `docs/mpsa_draft_outline.md:284`)
- "LLM as judge" — title-cased section heading (`docs/lit_rev_eval.md:69`)
- "LLM-as-a-Judge" / "LLM-as-a-judge" — citations and subsection headings (`docs/lit_rev_eval.md:86`, `docs/lit_rev_eval.md:137`, `docs/lit_rev_eval.md:197`, `docs/lit_rev_eval.md:203`); these are quoted paper titles, kept as cited
- "LLM-judge" — hyphenated short form used in test-subsection headings (`docs/proposed_methods_eval.md:5`, `:12`, `:19`, `:26`; also `docs/mpsa_draft_v6.md:201`)
- bare "judge" — used as the subject of sentences describing the LLM grader (`docs/proposed_methods_eval.md:1`, `:10`, `:17`, `:24`, `:28`, `:29`, `:31`, `:46`, `:93`, `:97`; also `docs/mpsa_draft_v6.md:159`, `:209`, `:215`, `:221`, `:227`)
- "LLM-as-judge protocol" / "LLM-as-judge scorer" — hyphenated form reused in v6 evaluation prose (`docs/mpsa_draft_v6.md:23`, `:149`, `:153`)

### granular policy-design extraction
**User's abstraction for the granular variables the extractor should produce: target (domain/technology), regulated entity (developer/deployer/agency), instrument (ban, mandate, disclosure, audit, procurement), enforcement, exemptions/safe harbors, definition scope, risk triggers (`docs/apsa_2026_abstract_entity.md:30-31`). "primitives" is deprecated per user comment; use "variables". User-accepted surface forms (per the user comment below): `policy design variables`, `granular entities and relations`, `granular entities and relations extraction`, `granular policy wise extraction`. User-rejected: `bill-interior variables` (must be replaced when encountered).**
- "granular policy-design variables" — hyphenated, "variables" form (`docs/mpsa_draft_outline.md:14`, `docs/mpsa_draft_outline.md:46`, `docs/mpsa_draft_outline.md:292`; `docs/mpsa_draft_v6.md:23`)
- "policy-design variables" — applied per user comment; replaces prior "policy-design primitives" (`docs/mpsa_draft_outline.md:86`)
- "document-level policy-design variables" (`docs/apsa_2026_abstract_entity.md:26`)
- "policy design variables" — applied per user comment; replaces prior "policy design primitives" inside the scare-quoted phrase (`docs/apsa_2026_abstract_entity.md:30`)
- "granular entities and relations" — user-accepted surface form (per comment below); appears in v6 at `docs/mpsa_draft_v6.md:35`, `:39`, `:51`, `:73`, `:233`, `:263`. Replaced the earlier v6 draft's `bill-interior variables` / `bill-interior questions` during active editing; `bill-interior` is no longer on disk in v6 but was present in the earlier revision this mapping first scanned.
- user-rejected: `bill-interior variables` / `bill-interior questions` — explicitly disallowed by the user comment below; no occurrences currently on disk in v6 at the time of this pass (verified by grep).

comment: primitives is not the top choice of word. others are acceptable. try replace "primitives" with others.

resolution: "primitives" replaced by "variables" at `docs/mpsa_draft_outline.md:86` and `docs/apsa_2026_abstract_entity.md:30`. Unrelated "primitives" in `docs/qa_app_eval.md` and `docs/qa_app_desc.qmd` refer to tool-call primitives for the QA agent, not to this concept; left untouched.

comment (added with v6 mapping pass):
- policy design variables
- bill-interior variables is not acceptable word. when encountered, must be replaced.
- recommended word: granular entities and relations, granular entities and relations extraction, granular policy wise extraction

resolution: (1) the accepted surface forms listed above are recorded in the concrete-meaning line and in the substitute list. (2) `bill-interior variables` was already replaced in `docs/mpsa_draft_v6.md` before this pass; grep for `bill-interior` in v6 returns 0 matches. If future agent drafts reintroduce it, it must be replaced with one of `granular entities and relations`, `granular entities and relations extraction`, or `granular policy wise extraction`.

### Regulatory Mechanisms
**Second dimension in `docs/NER_method.md:23`: by what mechanism the bill regulates the targeted entities (distinct from the target itself). Note: later docs (`docs/NER_method.md:74-75`, `docs/apsa_2026_abstract_entity.md:38-39`) restate the second dimension as `Regulated Entities` instead — an internal user-side shift that agent drafts need to preserve rather than paper over.**
- "how it is regulated" (`docs/mpsa_draft_outline.md:124`, `docs/mpsa_draft_outline.md:134`)
- "regulatory mechanism" — lowercase, in "Target-entity vs regulatory mechanism ambiguity" (`docs/mpsa_draft_outline.md:238`; also `docs/mpsa_draft_v6.md:191` same heading, plus prose uses at `docs/mpsa_draft_v6.md:35`, `:57`, `:103`)
- "Entity relations, aka how the entities are regulated, promoted or punished?" (`docs/mpsa_draft_outline.md:268`)

### field_score_matrix
**`GroupedCandidateSet` schema field (`docs/NER_agent_architecture_proposed_method.md:58`; frozen as required in `docs/NER_plan.md:15`): row order = `candidate_ids`, column order = `field_order` (entity, type, attribute, value), values in [0, 1].**
- "per-field score matrix" — space instead of underscore, descriptive prose form (`docs/mpsa_draft_outline.md:165`; `docs/mpsa_draft_v6.md:125`)

---

## Agent-only jargon from the evaluation docs (no user-side canonical term)

Entries below record terminology introduced by the agent in `docs/lit_rev_eval.md` and carried into / renamed by `docs/proposed_methods_eval.md`. None of the user's source docs (`docs/NER_method.md`, `docs/NER_agent_architecture_proposed_method.md`, `docs/NER_agent_architecture_references.md`, `docs/NER_plan.md`, `docs/apsa_2026_abstract_entity.md`, `docs/todo.md`) contain a canonical term for these concepts — they were added when the agent designed the evaluation pipeline. Listed so the user can decide whether to keep, rename, or map them to a user-side term.

### Test 1 / Test 2 / Test 3 / Test 4
**Agent-coined labels for the four evaluation tests. Concrete mapping (as established in `docs/lit_rev_eval.md`): Test 1 = Stage 3, per-quadruplet grounding verdict (`lit_rev_eval.md:353`); Test 2 = Stage 4, set-to-label coverage (`lit_rev_eval.md:364`); Test 3 = Stage 6, cross-method pairwise comparison (`lit_rev_eval.md:382`); Test 4 = Stage 8, judge bias audit (`lit_rev_eval.md:396`). Originally written in `lit_rev_eval.md` as parenthetical aliases on the Stage headings; `proposed_methods_eval.md` promotes the Test numbering to the primary heading.**
- "Test 1" / "Test 2" / "Test 3" / "Test 4" — primary section headings and in-prose references (`docs/proposed_methods_eval.md:5`, `:12`, `:19`, `:26`, `:35`, `:48`, `:63`, `:80`; also used throughout the Analysis and Verdict prose)
- "test 1, LLM-as-judge test" / "test 2, LLM-as-judge test" / "test 3, LLM-as-judge bias test part 1" / "test 4, LLM-as-judge bias test part 2" — original parenthetical aliases on Stage headings (`docs/lit_rev_eval.md:353`, `:364`, `:382`, `:396`)
- "Per-quadruplet grounding (Test 1)" / "Set-to-label coverage (Test 2)" / "Cross-method pairwise comparison (Test 3)" / "Judge bias audit (Test 4)" — v6 results-section subsection headings with the Test label in parentheses (`docs/mpsa_draft_v6.md:207`, `:213`, `:219`, `:225`). In-prose label uses appear at `docs/mpsa_draft_v6.md:201` ("four LLM-judge tests"), `:205` ("Test 2").

### pre-filter / rule-based pre-filter
**Agent-coined short name in `docs/proposed_methods_eval.md` for the Stage 2 filter defined in `docs/lit_rev_eval.md:345-351` as "Rule-based fabrication filter". Concrete meaning: drops quadruplets whose required fields are empty, whose evidence spans are missing, or whose span text is not literally present in the bill.**
- "rule-based pre-filter" (`docs/proposed_methods_eval.md:1`; `docs/mpsa_draft_v6.md:205`)
- "pre-filter" — bare form in prose and as adjective in "pre-filter failures" (`docs/proposed_methods_eval.md:7`, `:46`; `docs/mpsa_draft_v6.md:209`)

### strict coverage / permissive coverage
**Agent-coined aggregation thresholds for Test 2 / Stage 4 three-way output. Concrete meaning: "strict coverage" counts only `covered` verdicts; "permissive coverage" counts `covered` + `partially_covered`. Not present in `docs/lit_rev_eval.md`'s Stage 4 definition — introduced only in the evaluation write-up.**
- "strict-coverage" / "strict coverage" — column label and prose form (`docs/proposed_methods_eval.md:17`, `:52`, `:55`, `:61`, `:97`, `:103`; also "strict definition" / "strict-coverage gap" in prose at `docs/mpsa_draft_v6.md:215`, `:233`)
- "permissive definition" / "permissive coverage" — prose form (`docs/proposed_methods_eval.md:61`, `:97`; `docs/mpsa_draft_v6.md:215`)

### grounded-but-uncited / "novel" pile / novel-claim audit
**Agent-coined paraphrases for the concept defined in `docs/lit_rev_eval.md:374-379` as Stage 5 "Novel-claim bookkeeping" — quadruplets Stage 3 grounded as `entailed` but that no Stage 4 verdict cited as support for any NCSL label. `lit_rev_eval.md` already uses `novel` and `cited` as the two bucket names; `proposed_methods_eval.md` adds further surface forms.**
- "grounded-but-uncited ('novel') quadruplets" — compound modifier (`docs/proposed_methods_eval.md:24`; also "grounded-but-uncited quadruplets" at `docs/mpsa_draft_v6.md:235`)
- "the 'novel' pile" — noun phrase for the bucket (`docs/proposed_methods_eval.md:99`)
- "novel-claim audit sample" / "novel-claim audit" — the 50-row stratified human-verification sample (`docs/proposed_methods_eval.md:99`, `:103`; also "Novel-claim audit" as plot caption title at `docs/mpsa_draft_v6.md:237`)
- "novel-type breakdown" — plot caption (`docs/proposed_methods_eval.md:101`)
- "novelty count" / "raw novelty count" / "novelty" signal — prose framing of the Stage-5 bucket (`docs/mpsa_draft_v6.md:151` section heading, `:153`, `:235`)

### extractor / extractors
**Agent-coined umbrella noun spanning both NER methods (i.e. covers the `multi-turn stateful extraction pipeline` entry AND the `Agentic Skills driven NER` entry above). User's source docs do not use "extractor" — they name the two methods separately.**
- "two extractors" / "the two extractors" / "extractor methods" (`docs/proposed_methods_eval.md:1`, `:21`, `:26`, `:28`, `:93`, `:97`, `:103`)
- "the extractor" — generic singular (`docs/lit_rev_eval.md:84`, `:241`, `:301`, `:329`, `:343`, `:347`, `:360`, `:372`, `:380`, `:414`)
- "orchestrated extractor" / "skill-driven extractor" — paired compound forms (`docs/lit_rev_eval.md:349`)
- v6 prose uses: "extractor of their own" (`docs/mpsa_draft_v6.md:39`), "The extractor is decoupled from the AI-bill domain" (`:43`), "the extractor is Claude Sonnet 4.5" used as a model-family handle inside the judge-bias rationale (`:159`), "extractor-level tests" (`:233`).

---

## Agent-only jargon introduced in `docs/mpsa_draft_v6.md` (no user-side canonical term)

Entries below record terminology introduced by the agent in the v6 draft when naming internal components of the two pipelines. None of the user's source docs (`docs/NER_method.md`, `docs/NER_agent_architecture_proposed_method.md`, `docs/NER_agent_architecture_references.md`, `docs/NER_plan.md`) contain these exact labels as canonical terms — they were added when the agent wrote the Pipeline section of the paper draft. Listed so the user can decide whether to keep, rename, or map them to code-module names.

### inference-unit builder
**Agent-coined name for the pre-stage that splits a bill into `ContextChunk` units for the multi-turn pipeline. Concrete meaning: the chunking step that produces the input units Stage 1 (Zero-shot Annotator) reads.**
- "the bill is split into context chunks by an inference-unit builder" (`docs/mpsa_draft_v6.md:117`)
- User-side related terms: `ContextChunk` is the schema object this step emits (preserved verbatim, so not listed as a substitute above). User source docs describe the chunking step but do not give the *builder* a dedicated noun; this label is new in v6.

### section-reader tool
**Agent-coined name for the per-bill tool registered to the skill-driven agent that returns a slice of the bill text for a given start/end character offset. Concrete meaning: the tool-calling primitive that lets the single-agent pipeline read sections on demand instead of receiving the whole bill up front.**
- "through a section-reader tool rather than being handed the whole bill up front" (`docs/mpsa_draft_v6.md:133`)
- "#### Section reader tool" — subsection heading (`docs/mpsa_draft_v6.md:143`)
- "A section-reader tool is registered per bill" (`docs/mpsa_draft_v6.md:145`)
- User-side related terms: the Agentic Skills driven NER method (see entry above) is described in `docs/todo.md:1` and referenced via the Anthropic skills line in `docs/NER_agent_architecture_references.md` and `docs/lit_rev_ner.md`, but those docs do not name the tool. The v6 name `section-reader tool` is new.

### main agent loop
**Agent-coined name for the conversation driver of the skill-driven pipeline — the loop that dispatches tool calls, appends results to the message history, and re-enters the model until the model emits a final answer. Concrete meaning: the control-flow component of the single-agent pipeline.**
- "#### Main agent loop" — subsection heading (`docs/mpsa_draft_v6.md:135`)
- "The loop runs a multi-turn conversation with the model until the model returns a final answer rather than another tool call" (`docs/mpsa_draft_v6.md:137`)
- User-side related terms: the Anthropic skills references (`schluntz2024building`, `zhang2025equipping`, `anthropic2025skills`) describe this pattern generically; user source docs do not give it a dedicated label.
