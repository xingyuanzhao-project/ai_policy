---
title: ""
---

# Abstract

Framework (one paragraph, ~200 words), following the draft_4.tex flow problem -> proposal -> application -> evaluation -> claim.

- Problem: state AI bill research relies on document-level theme / keyword labels that treat "AI bill" as a single comparable unit and collapse heterogeneous governance instruments. Coarse labels cannot answer what sectors, technologies, or applications are regulated, or which actors carry obligations.
- Proposal: two extraction methods. The first is a multi-turn NER pipeline synthesized from the literature on cooperative multi-agent zero-shot NER, with three stages of candidate annotation, grouping, and refinement. The second is a skill driven agentic approach enabled by recent LLM tool-calling, where one agent follows a skill file and reads bill sections on demand to produce quadruplets in one conversation.
- Application: 1,200+ US state AI bills from NCSL covering 2023-2025.
- Model setup: Claude Sonnet 4.5 through OpenRouter with multi-turn tool calling; compared against a fixed-plan three-stage method.
- Evaluation: coverage of existing human theme / keyword labels via LLM-as-judge (direct match is unreliable at this granularity); resource cost reported in tokens, dollars, wall time.
- Contribution (three framing points from todo 2.1): (a) the pipeline produces granular policy-design variables that current labels cannot reveal, (b) it is corpus-agnostic and applies to other messy policy corpora with only a taxonomy swap, (c) compute hours replace weeks of manual coding while preserving traceable spans.

# Introduction

## Background

State legislatures have become a focal venue for AI lawmaking in the United States. Between 2022 and 2025, state AI legislation has surged \cite{ncsl2025ai}, with the NCSL tracker recording more than 1,200 state AI bills in 2025 alone \cite{ncsl2025ailegislation2025}. A surge of this magnitude unfolding across many sub domains including heath care, education, and ethical use of AI marks a starting point in which AI governance in the United States is being written.

Treating the AI bill as a single undivided unit without granular entity and relation extraction may be too coarse for downstream research, because the bills are bundles of heterogeneous governance instruments \cite{yoo2020regulation, kuteynikov2022key, sheikin2024principles, wang2024regulating}. Two bills can both count as "policy regulating AI" while doing fundamentally different things, such as one setting procurement rules, yet another establishing constraints on certain technologies. Collapsing this bundle into a single label breaks the downstream analysis in three ways. First is measurement error. The labels might target different procedures or entities but may measured as same category. The second problem is that the reduced effect magnitude when multiple type of relations about one concept are collapsed into one. Third is that the mechanism behind the regulated body and the method of regulation will not be revealed. Without extracting the entity and the relations, the downstream analysis will be hard to understand what policy design moved, which obligations, which targets, which enforcement.

### Why existing methods fall short

The author gathered bill corpora from NCSL and still found the annotations are too coarse to answer the question "what does it target and who does it regulate?". To address this, one may resort to dictionary or keyword based methods, but this does not seprate the target from the regulated entity on the relation to AI regulation {young2012affective, hopkins2010method}. For example, a bill may trigger the keyword "AI" in some sections, and mentioned "agent" in other sections as in chemical agents, but this may build a false positive detection of AI agent regulation.\cite{ri2024s2540}


- Rule-based NER and topic modeling do not generalize across 50 states' drafting conventions.
- Manual coding of 1,200+ bills per year is expensive and unstable across coders.

- NER exists. LLMs exists. agentic flow exists. we are in a differnet period now.. we should use them.

## How this research benefits the field?

- future research rely on clear unit, entiy etc.
- applying llms in policy research is a new frontier.

### Contribution 1. Granular policy-design variables (framing a)

- Output = quadruplets with source spans, not document-level labels.
- Enables mechanism-level research (which regulated entities, which targets, which instruments, which exemptions) instead of bill counts.

### Contribution 2. Generalizable pipeline (framing b)

- Skill file + taxonomy are swappable; the runner is corpus-agnostic.
- Transferable to federal bills, other regulatory texts, or non-English legislation with only taxonomy work.

### Contribution 3. Resource-efficient extraction (framing c)

- Full-corpus run cost / time recorded in `data/skill_ner_runs/runs/.../usage_summary.json`.
- Compute hours replace weeks of manual coding; cost profile reported alongside accuracy.

### Contribution 4. Open artifacts

- Quadruplet dataset with spans for 1,200+ state AI bills.
- Reusable skill file (`settings/skills/ner_extraction.md`) and runner (`src/skill_ner/`).
- Live extraction demo (`src/qa/web_app.py`) for inspection and replication.

# Literature Review
Theory side:

- many states are introducing AI bills, but the focus are all over the place.
- there is no way to reasearch unless clearly define and draw the scope
- they are treated like buzzwords, not solid concepts

Method side:
## Non-LLM methods for policy-text analysis

### Rule-based, keyword, and dictionary labeling

- Theme tagging, keyword lists, regex pattern matching on bill text.
- Strength: cheap, reproducible, auditable.
- Limit: heterogeneous state drafting defeats fixed patterns; cannot separate target from regulated entity.

### Topic modeling and unsupervised clustering

- LDA / BERTopic style clustering on bill corpora.
- Limit: produces themes, not policy-design variables; no evidence spans.

### Manual expert coding

- Hand-coded subsets by NCSL, Brookings, state analysts.
- Limit: coverage-accuracy trade-off; stops at document level; hard to scale across states / years.


## LLM methods for entity extraction 

general references: @references/references_methods.md
more suitable: @NER_agent_architecture_references.md

### Prompt-based NER

- Zero-shot / few-shot NER with GPT-4, Gemini, Claude; structured output.
- Works on short, well-bounded text; struggles on long legislative documents.

### Multi-agent and two-stage decomposition

- Cooperative agents; locate-then-type pipelines; planner-worker designs.
- Relevant references: #1-5, #7 in `docs/apsa_2026_abstract_entity.md`.

### Agentic / skill-driven workflows

- Tool-augmented agents that read long documents in sections and refine across turns.
- Strength: adapts to variable document structure; robust to drafting variation.
- Limit: cost, reliability, and evaluation on real policy corpora largely untested.

Method side:

- There are dataset for corpus, but not for what they are and what they are about.
- current papers focusing on "theme" or "area", rahter than concrest "entity" or "target" being regulated.
- there are existing methods to extract, but not entirely suitable for this research.
- llm empowers us, especially with agents

## Our take on existing methods

- triplets (target, entity, attribute) can cover both the target entity being regulated and how it is regulated, rather than a simple word theme or keyword extraction.
- such quadruplets (entity, type, attribute, value) should be anchored to source spans for every bill, providing evidence for human review to open the black box of llm.
- Based on the latest llm developments, both multi-turn pipeline and skill driven agentic approach supporting tool calls should be able to handle the complexity of the task.


# Method

## Task Definitions

- Input: full text of a state AI bill.
- Output: list of quadruplets {entity, type, attribute, value} with evidence spans, covering both the target entity being regulated and how it is regulated.
- Requirement: every quadruplet must trace back to a verbatim passage in the bill; no free generation of unsupported entities.

## Model Setup

- Model: Claude Sonnet 4.5 through OpenRouter
- Parameters: temperature 0.0; `max_tokens = 16384` (Sonnet 4.5 ceiling, see `docs/log_0.31_ner_crashes.md` C3).

## Pipeline




### From fixed-plan three-stage method to skill-driven agentic NER

frame this part as "A is this, from the syntheizing from the literature. 
however, recent development in llm allwos B, so we also incroprate B"

### Multi-turn NER pipeline @src/ner/

Multi-turn stateful pipeline. The bill is split into context chunks. Three stages pass explicit artifacts between them, each stage a separate LLM call with its own prompt and its own parsed output schema.


#### Extraction of entities

- Agent reads the chunked bill
- Produces candidate quadruplets(entity, type, attribute, value), and also supporting evidence spans for each field of the quadruplet.

#### Grouping and scoring candidates

- Agent reads the bill-level pool of candidate quadruplets produced across all chunks
- Groups candidates that refer to the same quadruplet, and produces a per-field score matrixfor each field of the quadruplet (entity, type, attribute, value) for each candidate in the group based on the evidence linked to the candidates.

#### Refinement of candidates sets

- Agent reads each grouped candidate set together
- Based on both evidence and scores, one refined quadruplet(entity, type, attribute, value) per group that merges the grouped members, preserving the supporting evidence spans for each field.

### Skill driven agentic approach @src/skill_ner/

The agent will be enpowerd with tool calls capability to read the bill section by section. And a skill file will be used to guide the agent's behavior.



#### main agent loop

- Agent runs a multi-turn conversation with the model until the model returns a final answer instead of another tool call.
- Each turn, the agent will decide whether and how to execute any tool calls the model issued and appends the results back into the message history for the next turn.
- Produces one JSON payload of quadruplets(entity, type, attribute, value) per bill, parsed from the final assistant message.

#### agentic skill definition

- A markdown skill file is loaded as the system prompt and defines the quadruplet schema, the extraction process, per-field quality criteria, and the required output JSON format.
- Directs the agent on which entities to target, which to skip, and how to anchor each field of the quadruplet to an evidence span in the bill text.

#### enabling tool calls and how is called

- A `read_section` tool is registered per bill with an OpenAI function schema taking `start_offset` and `end_offset`, and a handler that slices the bill text at the requested offsets.
- The agent read the files and understand the scope and strucutre, then decide which sections to read and how to read them by themselves.
- Each model turn can emit tool calls that the loop dispatches to the handler and returns to the model as a tool-role message for the next turn of the conversation.

## Evaluation Design (framework for todo 1.6 and 1.7)

### Gold reference

- Existing human theme / keyword labels at bill level (NCSL tags where available).
- No pre-existing entity-level gold standard; this is an acknowledged limitation.
- Direct string match is unreliable because extracted quadruplets are finer-grained than bill-level labels.
- Use LLM-as-judge to test whether the set of extracted quadruplets semantically covers each human label.

### Cross method evaluation

#### Performance comparison

- compare the performance of the two methods, using the llm as judge to evaluate the quality of the extracted quadruplets.

#### Resource comparison
- Tokens, dollars, wall time per bill: orchestrated NER vs skill-driven NER.
- Projected manual-coding hours vs observed compute hours (framing c).

# Data

## Data Source

- Corpus: state AI legislation tracked by NCSL, years 2023, 2024, 2025.
- Scraper: selenium based crawler
- Metadata file `data/ncsl/us_ai_legislation_ncsl_meta.csv` contains, per bill, the fields `state`, `year`, `bill_id`, `bill_url`, `title`, `status`, `date_of_last_action` if available, `author`, with partisanship information if available, `topics`, `summary`, `history`, and `text`. Bill text is scraped from `bill_url` and stored separately, joined to metadata by `bill_id`.

## Descriptive Statistics

- Bill count per year and per state (plot).
- Text-length distribution per bill (plot from `scripts/desc_stat.py`).
- Coverage of existing theme / keyword labels (plot).
- Party composition coverage where available.
- Cross-tabulation bill count x status (introduced / passed / enacted).

## Challenges in the Data

### Heterogeneity across jurisdictions

- Different drafting conventions, cross-references, section numbering across 50 states.
- Same concept encoded with different language (e.g. "automated decision system" vs "AI system" vs "algorithmic tool").
- This cannot be simply sovled by rule-based methods, llm can do it better.

### Target-entity vs regulatory mechanism ambiguity

- given dataset classification does not support this. show some examples.

### Keyword-based inclusion false positives

- Bills that mention AI in a single definition clause but whose body is unrelated (e.g. environmental / chemical policy) reach the corpus through keyword filtering.
- Reference: `2024__RI S 2540` case in `docs/log_0.31_ner_crashes.md` J2.

# Results

## Baseline vs Proposed Method

- Baseline 1: existing theme / keyword labels (document-level).
- Proposed 1: fixed-plan three-stage method, run `run_sonnet_full_20260416`.
- Proposed 2: skill-driven agentic NER, run `skill_full_20260416_v2`.
- Metrics to report:
  - coverage of human labels by extracted quadruplets judged by LLM-as-judge,
  - count of novel entities recovered beyond theme labels by LLM-as-judge
  - Resource table from `usage_summary.json`: calls, tokens, dollars, elapsed per pipeline.
  - qualitative evaluation of the error analysis
    - Multi-turn stateful pipeline: where it fails and why it fails there
    - Skill driven agentic approach: where it fails and why it fails there

#### Discussion: skill-driven vs fixed-plan three-stage method vs theme / keyword labels

## Downstream Analysis (todo 2.3 and 2.4)

### Distribution of targeted entities and entity types

### Entity relations, aka how the entities are regulated, promoted or punished?

### temporal drift of regulatory focus

### Focus vs bill outcome

## Live Demonstration (todo 3)

- Flask app (`src/qa/web_app.py`) accepts a bill / text, returns extracted quadruplets and evidence spans.
- Deployment bootstrap via `src/qa/bootstrap_app.py`.
- Role in the paper: reproducibility demo + invitation for other researchers to apply the pipeline to new bills.

# Discussion

## Limitations

- No entity-level gold standard; LLM-as-judge is the evaluator for label coverage.
- Single commercial model family (Claude) dominates; cross-family comparison is scoped for follow-up work.
- Corpus limited to US state AI bills and to English text.
- Target vs regulated-entity disambiguation still partially depends on analyst judgment at the analysis stage.
- Keyword-based corpus inclusion produces false positives that the pipeline cannot fully reject without a relevance pre-check.

## Implications

### Granular policy-design variables (framing a)

- Downstream research can test mechanism hypotheses (who is regulated, with which instrument, under which conditions) instead of only bill counts.
- Reconnects quantitative legislative research to the substantive content of bills.

### Scalability to other messy policy corpora (framing b)

- Corpus-agnostic pipeline; swapping the skill file and taxonomy is sufficient for a new domain.
- Transferable to federal bills, other regulatory texts (financial, environmental, data protection), or non-English legislation.
- The skill-driven architecture extends naturally to relevance pre-checks (see J2 / J3 in `docs/log_0.31_ner_crashes.md`) without adding a new orchestration stage.

### Resource profile vs manual and keyword methods (framing c)

- Hours of compute replace weeks of manual coding while preserving traceable spans.
- Agentic extraction is more expensive per bill than keyword labelling but produces policy-design granularity that keywords cannot.
- Trade-off positioning: keyword (low $, low info), manual coding (high $, high info, slow), agentic (medium $, high info, fast).

### Downstream uses beyond this paper

- Distant supervision for fine-tuning smaller open-source models on the extracted quadruplets.
- Policy-design indices (strictness, coverage, enforcement, private right of action).
- Diffusion studies across states and over time using entity-level patterns.
- Integration with the companion definition paper to connect "how AI is defined" with "what AI policy targets".
