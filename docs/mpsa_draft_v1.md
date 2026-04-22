---
title: ""
---

# Abstract

State AI bill research largely relies on document-level theme and keyword labels that treat each bill as a single undivided unit and collapse heterogeneous governance instruments into one row \cite{depaula2024regulating, depaula2025evolving}. Coarse labels cannot answer which sectors, technologies, or applications are regulated, or which actors carry obligations. We propose two extraction methods to unpack the bill. The first is a multi-turn stateful NER pipeline synthesized from recent work on cooperative multi-agent zero-shot NER \cite{wang2025cooperative}, with three stages of candidate annotation, grouping, and refinement. The second is a skill-driven agentic approach enabled by recent LLM tool calling, in which one agent follows a skill file and reads bill sections on demand to produce quadruplets in a single conversation \cite{schluntz2024building, zhang2025equipping}. Both methods extract quadruplets of entity, type, attribute, and value anchored to evidence spans, and run on 1,826 U.S. state AI bills from NCSL covering 2023 to 2025 \cite{ncsl2025ai}. We evaluate coverage of NCSL topic labels using an LLM-as-judge protocol, and report tokens, dollars, and wall time for each method. The pipeline produces granular policy-design variables that current labels cannot reveal, transfers to other messy policy corpora with only a taxonomy swap, and replaces weeks of manual coding with hours of compute while keeping traceable spans.

# Introduction

## Background

State legislatures have become a focal venue for AI lawmaking in the United States. Between 2022 and 2025, state AI legislation has surged \cite{ncsl2025ai}, with the NCSL tracker recording more than 1,200 state AI bills in 2025 alone \cite{ncsl2025ailegislation2025}. A surge of this magnitude unfolding across many sub domains including heath care, education, and ethical use of AI marks a starting point in which AI governance in the United States is being written.

Treating the AI bill as a single undivided unit without granular entity and relation extraction may be too coarse for downstream research, because the bills are bundles of heterogeneous governance instruments \cite{yoo2020regulation, kuteynikov2022key, sheikin2024principles, wang2024regulating}. Two bills can both count as "policy regulating AI" while doing fundamentally different things, such as one setting procurement rules, yet another establishing constraints on certain technologies. Collapsing this bundle into a single label breaks the downstream analysis in three ways. First is measurement error. The labels might target different procedures or entities but may measured as same category. The second problem is that the reduced effect magnitude when multiple type of relations about one concept are collapsed into one. Third is that the mechanism behind the regulated body and the method of regulation will not be revealed. Without extracting the entity and the relations, the downstream analysis will be hard to understand what policy design moved, which obligations, which targets, which enforcement.

### Why existing methods fall short

The author gathered bill corpora from NCSL and still found the annotations are too coarse to answer the question "what does it target and who does it regulate?". To address this, one may resort to dictionary or keyword based methods, but this does not seprate the target from the regulated entity on the relation to AI regulation {young2012affective, hopkins2010method}. For example, a bill may trigger the keyword "AI" in some sections, and mentioned "agent" in other sections as in chemical agents, but this may build a false positive detection of AI agent regulation.\cite{ri2024s2540}

Rule-based NER and topic modeling carry the same limitation in a different form. The fifty states do not share a drafting convention, and a definitions clause in one state is a preamble in another and a section cross-reference in a third. A fixed pattern or a topic cluster built on bill-level text cannot follow that variation, and the output remains a theme rather than a set of policy-design variables with evidence \cite{blei2003latent, roberts2014structural, grootendorst2022bertopic}. Manual expert coding is the method that in principle can reach the bill interior, and NCSL, Brookings, and state analysts have coded subsets by hand, but coding more than a thousand bills per year is expensive, slow, and unstable across coders \cite{krippendorff2018content}, and even then the output stops at the document level rather than producing entity-and-relation content inside each bill.

The timing has changed. Named entity recognition, large language models, and agentic workflows that can read long documents in sections all exist, and recent work shows that cooperative multi-agent prompting and skill-driven tool use can produce structured extractions from long text without fine-tuning \cite{wang2025cooperative, zhang2025equipping}. This paper takes the step these developments make available.

## How this research benefits the field?

The contribution of this paper is to replace document-level labels with bill-interior variables that future research can use directly. Downstream work on diffusion, adoption, and legislative content has been held at the document level because the measurement instrument stopped there. With quadruplets anchored to evidence spans, the same research questions can be asked at the level of what is regulated and how. The paper also treats the application of LLMs in policy research as a first-class method rather than a convenience, with resource cost and evaluation reported alongside the output so the method is usable rather than illustrative.

### Contribution 1. Granular policy-design variables (framing a)

The pipeline outputs quadruplets of entity, type, attribute, and value with a source span for each field, rather than one label per bill. This lets downstream work test mechanism-level hypotheses, such as which regulated entities are named, which targets are set, which instruments are used, and which exemptions apply, without re-reading the bill text.

### Contribution 2. Generalizable pipeline (framing b)

The skill file and the taxonomy are swappable, and the runner is corpus-agnostic. Moving to federal bills, to other regulatory text in finance or environment, or to non-English legislation requires taxonomy work rather than pipeline work. The same extractor drives a different corpus once the skill file is updated.

### Contribution 3. Resource-efficient extraction (framing c)

Full-corpus runs are recorded in `data/ner_runs/runs/run_sonnet_full_20260416/usage_summary.json` and `data/skill_ner_runs/runs/skill_full_20260416_v2/usage_summary.json`. The multi-turn pipeline ran at 21,586 LLM calls for US\$228.78, and the skill-driven pipeline ran at 11,231 calls for US\$315.60. Weeks of manual coding are replaced by hours of compute while the cost profile is reported alongside accuracy, not substituted for it.

### Contribution 4. Open artifacts

The paper releases the quadruplet dataset with spans for the 1,826 state AI bills, the reusable skill file at `settings/skills/ner_extraction.md`, the runner at `src/skill_ner/`, and a live extraction demo at `src/qa/web_app.py` so any reader can submit a bill and inspect the extracted quadruplets and their evidence spans.

# Literature Review

## Theory side

The qualitative legal and political-science literature on U.S. AI governance converges on one observation. States are now the main site of AI legislation, and they are producing bills faster than a shared framework can form, so the bills as a group are fragmented across sectors, uneven across states, and often built around definitions that drift from one statute to the next \cite{yoo2020regulation, kuteynikov2022key, sheikin2024principles, wang2024regulating, defranco2024assessing, agbadamasi2025navigating}. The content-focused work that does look inside bills reports a moving target. The 2019 to 2023 window is dominated by health, education, and advisory bodies, and the 2024 window shifts substantially toward generative AI and synthetic content, with private-sector regulation dropping in share, while definitions remain inconsistent across states \cite{oduro2022obligations, depaula2024regulating, depaula2025evolving}. The one quantitative political-economy study of state AI adoption reports that economic conditions and unified Democratic government predict adoption, while ideology and neighbor adoption do not, and that partisan structure emerges specifically around consumer-protection AI bills rather than around AI regulation in general \cite{parinandi2024investigating}. Across this literature, the empirical unit is either the bill as a whole, a hand-coded subset, or a roll call. What is inside the bill, at the level of which entities carry obligations and which mechanism is used, is treated as context rather than as a variable.

## Method side

### Non-LLM methods for policy-text analysis

#### Rule-based, keyword, and dictionary labeling

Theme tagging, keyword lists, and regex patterns applied to bill text are cheap, reproducible, and auditable, and the political-science text-as-data literature has used them for two decades \cite{grimmer2013text, young2012affective, hopkins2010method}. Recent infrastructure in this family, including the state legislation topic tracker of \cite{garlick2023laboratories} and its extension in \cite{dee2025policy}, shows how far labeling scales when the label set is fixed in advance. The limit is the same as in the rule-based family more generally. Heterogeneous state drafting defeats fixed patterns, and a label says whether a bill is on a topic but cannot separate the target from the regulated entity or name the instrument.

#### Topic modeling and unsupervised clustering

LDA, structural topic models, and BERTopic-style clustering extract recurring themes from a corpus without a fixed label set \cite{blei2003latent, quinn2010how, grimmer2010bayesian, roberts2014structural, lucas2015computer, grootendorst2022bertopic}. A recent policy application uses the same family to summarize AI policy documents at scale \cite{pham2026using}. The output is a theme distribution rather than a policy-design variable, and no evidence span is produced, so these methods inform descriptive work but are not a measurement instrument for what a bill actually requires.

#### Manual expert coding

Hand-coded subsets such as those produced by NCSL, Brookings, and state analysts, backed by standard inter-coder reliability protocols \cite{krippendorff2018content}, are the high-quality end of the existing toolkit. They reach into bill interiors but trade coverage for accuracy, stop at the document level, and scale poorly across fifty states and multiple years.

### LLM methods for entity extraction

#### Prompt-based NER

Zero-shot and few-shot entity extraction with commercial LLMs, including GPT-4, Gemini, and Claude, has become practical for well-bounded text. Clinical NER \cite{hu2023improving, islam2025llm}, non-English clinical reports with long-context prompts \cite{akcali2025automated}, and cyber threat intelligence \cite{feng2025promptbart} all report workable accuracy from prompt engineering alone. The common limit is document length. Legislative bills are long, structurally varied, and cross-referenced, and a single-pass prompt has no way to read one section and then decide what to read next.

#### Multi-agent and two-stage decomposition

A second family splits NER across cooperating agents or across a locate-then-type pipeline, so that each sub-step operates on a narrower input. Wang et al.'s cooperative multi-agent framework is the representative example for zero-shot NER, with self-annotator, type-related feature extractor, demonstration discriminator, and overall predictor each doing one sub-step \cite{wang2025cooperative}. Two-stage locate-then-type models \cite{ye2023decomposed}, entity structure discovery \cite{xu2025zeroshot}, API entity and relation joint extraction \cite{huang2023api}, relation-classification agent architectures \cite{berijanian2025comparative}, automatic labeling for sensitive text \cite{deandrade2025promptner}, and benchmark construction for scholarly NER \cite{otto2023gsapner} round out the family. These designs handle long, varied inputs better than a single pass but still require a fixed plan, so they do not decide on the fly which bill section to look at next.

#### Agentic and skill-driven workflows

Retrieval-augmented generation \cite{lewis2020retrieval}, the ReAct pattern that interleaves reasoning and acting \cite{yao2023react}, and tool-use training \cite{schick2023toolformer} laid the groundwork for agents that read a document in sections and refine across turns. Recent agentic RAG surveys \cite{singh2025agentic}, search-augmented reasoning \cite{li2025searcho1, jin2025searchr1}, and the Anthropic skills line that loads a markdown skill file as system prompt and grants structured tool access \cite{schluntz2024building, zhang2025equipping, anthropic2025skills, anthropic2025sdk} together define the design space for a single agent that can read a bill in sections and produce quadruplets in one conversation. Cost, reliability, and evaluation on real policy corpora in this family are still largely untested.

## Our take on existing methods

The paper treats the quadruplet as the output unit. An entity is the target being regulated or the actor carrying an obligation, the type is a taxonomy label, the attribute names the mechanism, and the value carries the specific content, with an evidence span anchored to every field \cite{xu2025zeroshot}. The quadruplet covers both what is regulated and how, which a theme or a keyword label cannot. Both methods proposed below run without fine-tuning, so the existing literature on prompt-based and agentic extraction is directly applicable and the corpus does not need hand-labeled training data.

# Method

## Task Definitions

The input to the pipeline is the full text of one state AI bill. The output is a list of quadruplets with fields entity, type, attribute, and value, each carrying an evidence span that points back to a verbatim passage in the bill. Every quadruplet must be traceable to bill text, and no field is produced without a span, so the output is auditable by a human reader without re-running the model.

## Model Setup

The model is Claude Sonnet 4.5 reached through OpenRouter. Temperature is 0.0 and `max_tokens` is set to 16,384, which is Sonnet 4.5's hard completion ceiling.\footnote{Token-ceiling selection and the failure modes that pushed it to the model's physical ceiling are documented in \texttt{docs/log\_0.31\_ner\_crashes.md} C3.} Running through OpenRouter lets both methods call the same weights, so a cost and accuracy comparison between the two methods is not confounded by model choice.

## Pipeline

### From fixed-plan three-stage method to skill-driven agentic NER

The multi-turn pipeline is built by synthesizing the cooperative multi-agent NER literature \cite{wang2025cooperative, xu2025zeroshot}, which is the closest published match to the project's constraint of no fine-tuning, long legal text, and decomposed extraction. The three stages and their input-output contracts follow that literature directly. Recent developments in tool calling and skill files on commercial LLMs \cite{yao2023react, schluntz2024building, zhang2025equipping, anthropic2025skills} make a second design available. A single agent, following a skill file and reading bill sections through a tool, can produce quadruplets in one conversation. The paper implements both methods and reports the two side by side, so the comparison is between a synthesized fixed-plan pipeline and a recent agentic design on the same corpus rather than between an LLM method and a non-LLM baseline.

### Multi-turn NER pipeline @src/ner/

The multi-turn pipeline is a stateful three-stage design. The bill is split into context chunks by an inference-unit builder, and each of the three stages is a separate LLM call with its own prompt, its own parsed output schema, and its own persisted artifact. Stage outputs are passed between stages by candidate identifier rather than by inlined payload, so evidence spans are recovered from the candidate pool rather than duplicated.

#### Extraction of entities

The zero-shot annotator reads one context chunk at a time and emits `CandidateQuadruplet` objects with four fields and four field-linked evidence maps. Missing fields are allowed at this stage, because the annotator's job is to propose candidates with their supporting spans, not to finalize the quadruplet. Each candidate keeps a stable `candidate_id` that later stages use to recover the candidate and its evidence.

#### Grouping and scoring candidates

The eval assembler reads the bill-level pool of chunk candidates and groups candidates that refer to the same underlying quadruplet into a `GroupedCandidateSet`. For each grouped set it emits a per-field score matrix with row order aligned to `candidate_ids` and column order aligned to the canonical field order of entity, type, attribute, and value. This stage produces the grouping and the scores only, and does not finalize any refinement relation.

#### Refinement of candidates sets

The granularity refiner reads each grouped candidate set together with the referenced candidates and their evidence, and emits one `RefinedQuadruplet` per group. The relation labels used between grouped candidates are the canonical set support, overlap, conflict, duplicate, and refinement, stored as a field-wise matrix on an optional `RefinementArtifact`. Field-linked evidence is preserved on the refined output, so the final quadruplet is inspectable without re-reading raw chunks.

### Skill driven agentic approach @src/skill_ner/

The skill-driven agent runs as a single conversation over one bill. A markdown skill file is loaded as the system prompt and defines the quadruplet schema, the extraction process, and the output JSON. The agent reads bill sections on demand through a registered `read_section` tool rather than being handed the whole bill up front.

#### Main agent loop

The loop runs a multi-turn conversation with the model until the model returns a final answer rather than another tool call. Each turn, the loop dispatches any tool calls the model issued, appends the tool results to the message history, and re-enters the model. The output is one JSON payload of quadruplets per bill, parsed from the final assistant message.

#### Agentic skill definition

The skill file carries the quadruplet schema, per-field quality criteria, and the required output JSON. It directs the agent on which entities to target, which to skip, and how to anchor each field to an evidence span. The skill file is the only instruction the agent receives, so swapping the skill file is how the extractor is pointed at a different taxonomy.

#### Enabling tool calls and how is called

A `read_section` tool is registered per bill with an OpenAI function schema that takes `start_offset` and `end_offset`, and a handler that slices the bill text at the requested offsets. The agent reads an index of the bill first, decides which sections to request, and calls the tool one or more times across turns. Each tool call produces a tool-role message that re-enters the next model turn, so the model sees only the sections it requested rather than the full text.

## Evaluation Design

### Gold reference

The reference for coverage evaluation is NCSL's existing human topic labels at bill level \cite{ncsl2025ai}. No pre-existing entity-level gold standard exists for this corpus, and constructing one by hand at scale is not feasible within the paper, which is an acknowledged limitation. Direct string match between extracted quadruplets and NCSL labels is unreliable because the quadruplets are finer-grained than the labels. The paper therefore uses an LLM-as-judge protocol \cite{zheng2023judging, laskar2025improving, ho2025llm} to decide, for each bill and each NCSL label, whether the method's quadruplets jointly account for that label.

### Cross method evaluation

#### Performance comparison

Both methods run on the same 1,703-bill evaluation intersection, and the LLM judge runs at Gemini 2.5 Pro with temperature 0.0. Four tests are reported. The first is per-quadruplet grounding against the bill text, which isolates an item-level accuracy signal that does not depend on the label set. The second is set-to-label coverage against NCSL topics, which isolates a reference-aligned signal. The third is a cross-method pairwise comparison in which the judge sees both methods' quadruplet sets for the same bill and picks one, with swap-averaging across presentation order to cancel position bias \cite{zheng2023judging}. The fourth is a judge bias audit on a pooled 100-row sample with four perturbations, which sets a trust bound on the first three tests rather than ranking the methods \cite{ye2024justice, guerdan2025validating, tan2024judgebench}.

#### Resource comparison

For each method, the paper reports calls, prompt and completion tokens, dollar cost, and elapsed wall time per bill, drawn from `usage_summary.json` in each run directory. Projected manual-coding hours are computed from a conservative coder throughput and compared against the observed compute hours. The cost table sits alongside the accuracy tables rather than in a separate appendix, so the trade-off between the two methods and against human coding is visible on the same page.

# Data

## Data Source

The corpus is state AI legislation tracked by the National Conference of State Legislatures for 2023, 2024, and 2025 \cite{ncsl2025ai, ncsl2023ailegislation2023, ncsl2024ailegislation2024, ncsl2025ailegislation2025}, with underlying data supplied by LexisNexis State Net \cite{lexisnexis2025statenet}. Bill text is scraped from each bill's canonical URL by a Selenium crawler, and a metadata file at `data/ncsl/us_ai_legislation_ncsl_meta.csv` records, per bill, the fields `state`, `year`, `bill_id`, `bill_url`, `title`, `status`, `date_of_last_action` when available, `author` with partisanship when available, `topics`, `summary`, `history`, and `text`. Bill text is stored separately from metadata and joined by `bill_id`. The merged corpus covers 1,879 rows; 53 empty-text rows are filtered out; and 1,826 bills enter the pipeline.

## Descriptive Statistics

The bill count is distributed across years as 137 in 2023, 480 in 2024, and 1,262 in 2025. Among 2025 bills, 192 have a status beginning with `Enacted` across 45 states, with California carrying the highest count at 24. Bill-length distribution and per-state counts are produced by `scripts/desc_stat.py`; coverage of existing NCSL theme labels is reported alongside the method results in §Results; and party composition is reported where the `author` field carries partisanship.

## Challenges in the Data

### Heterogeneity across jurisdictions

Drafting conventions, section numbering, and cross-references differ across the fifty states, and the same concept appears under different language, such as "automated decision system" in one bill, "AI system" in another, and "algorithmic tool" in a third. Fixed-pattern rule-based extraction cannot follow this variation, and the LLM methods handle it by reading the defining clause in context rather than matching a surface string.

### Target-entity vs regulatory mechanism ambiguity

The same bill can name an entity in one clause and then specify a mechanism that acts on that entity two clauses later, and a keyword method cannot tell which role a phrase is playing. The pipeline handles this by producing entity and attribute as two separate fields of the same quadruplet, each with its own evidence span, so the role each phrase plays is recoverable from the span and not just from the label.

### Keyword-based inclusion false positives

A bill can reach the corpus because a keyword filter matched one "artificial intelligence" string, while the body of the bill regulates something else entirely. The Rhode Island Clean Air Preservation Act \cite{ri2024s2540} is the concrete instance used in this paper. It reached the NCSL corpus through one definitions-section match, and the body regulates stratospheric aerosol injection and chemical and biological agents, where the word "agent" refers to a chemical agent rather than an AI agent.\footnote{The false-positive trace and the structural tells in the extracted quadruplet are recorded in \texttt{docs/log\_0.31\_ner\_crashes.md} J2.} Both methods are affected by this corpus-inclusion bias, because both methods trust the corpus filter at intake; handling the false positive would require a relevance pre-check that is outside the extraction step.

# Results

## Baseline vs Proposed Method

The comparison covers three columns. The baseline column is the existing NCSL theme and keyword labels at document level \cite{ncsl2025ai}. The two proposed methods are the multi-turn pipeline, run at `run_sonnet_full_20260416`, and the skill-driven agent, run at `skill_full_20260416_v2`. Both ran over the same corpus and are evaluated on a 1,703-bill intersection. A rule-based pre-filter drops quadruplets whose required fields are empty, whose evidence spans are missing, or whose span text is not literally present in the bill; after this filter the multi-turn pipeline retains 10,429 of 10,876 quadruplets (95.89%) and the skill-driven agent retains 8,926 of 9,555 (93.42%). Only these surviving quadruplets enter the LLM-judge tests below.

### Per-quadruplet grounding

The first test asks the judge whether each surviving quadruplet is supported by the bill text. The multi-turn pipeline records 85.05% entailed, 11.55% neutral, and 3.39% contradicted. The skill-driven agent records 78.32% entailed, 18.07% neutral, and 3.61% contradicted. The contradiction rate, which is the sharpest failure signal in this test, is indistinguishable across methods. The entailed-rate gap is confounded with volume, because the multi-turn pipeline was judged on 1,503 more quadruplets than the skill-driven agent. The separator in this test is the neutral rate, which is higher for the skill-driven agent, consistent with its tendency to paraphrase values; this also explains why every pre-filter failure on the skill-driven agent is of type `span_not_literal` (629 of 629) while the multi-turn pipeline's pre-filter failures are dominated by missing fields and missing spans.

![Grounding verdict distribution by method.](../output/evals/v1/plots/02_grounding_distribution.png)

### Set-to-label coverage

The second test asks the judge whether, for each bill and each NCSL topic label on that bill, the method's surviving quadruplets jointly account for the label. On the permissive definition that counts both covered and partially covered, the skill-driven agent leads by 3.33 pp (87.04% vs 84.03%). On the strict definition that counts only covered, the skill-driven agent leads by 32.06 pp (81.50% vs 49.44%), and its error rate is roughly forty times smaller (0.03% vs 1.30%), driven by the multi-turn pipeline's longer supporting lists overflowing the judge prompt. Because NCSL topic labels are the reference for the downstream analyses below, the strict-coverage gap is the outcome-relevant number.

![Coverage by method and cost-vs-coverage.](../output/evals/v1/plots/01_coverage_by_method.png)

### Cross-method pairwise comparison

The third test shows the judge both methods' quadruplet sets for the same bill and asks it to pick one. Each of the 1,693 bills in the intersection is judged twice with presentation order swapped, so position bias is canceled by averaging \cite{zheng2023judging}. Both presentation orders prefer the skill-driven agent, with a swap-averaged win rate of 75.99% against 22.68% for the multi-turn pipeline. A count-normalised variant that adjusts for the fact that the multi-turn pipeline writes more quadruplets cuts the raw gap roughly in half, and the skill-driven agent still leads by 3.4× in normalised points (0.3791 vs 0.1126).

![Pairwise win rate.](../output/evals/v1/plots/04_pairwise_winrate.png)

### Judge bias audit

The fourth test perturbs the coverage prompt on a pooled 100-row sample and measures how often the judge flips its verdict. Position flips 3%, verbosity flips 4%, self-preference flips 7%, and authority flips 14%. The first three are at or near the noise range for this kind of audit. Authority is the outlier; a 14% sensitivity to a prefix that claims a senior expert insists on a verdict means any prompt that carries authority-aligned language has to be treated as a rerun condition. Inspection of the frozen prompts confirms that neither the grounding prompt nor the coverage prompt carries authority cues, so the results above are not contaminated by this channel.

![Bias audit scorecard.](../output/evals/v1/plots/06_bias_scorecard.png)

#### Discussion: skill-driven vs fixed-plan three-stage method vs theme and keyword labels

Of the three extractor-level tests, two separate the two methods and both point the same way. Set-to-label coverage puts the skill-driven agent 32.06 pp ahead on the strict definition, and cross-method pairwise comparison puts the skill-driven agent ahead 75.99% to 22.68% after swap-averaging. Per-quadruplet grounding does not separate the two methods on the contradiction signal. The judge diagnostic finds no trust failure on the prompts in use. Against NCSL's own document-level labels, both proposed methods produce bill-interior variables that the labels themselves cannot carry, so the label-level comparison is not a head-to-head but a statement that the proposed methods add a dimension the baseline does not have.

Under the audit of the grounded-but-uncited quadruplets, the multi-turn pipeline retains 6,319 entries and the skill-driven agent retains 1,730. The multi-turn pipeline's audit sample splits into bill-relevant specifics that NCSL's topic tags do not name and bill-adjacent entries that are off-topic relative to AI policy, such as tax credits, demonstrated-mastery assessments, and nuclear energy in bills that happen to mention AI. The skill-driven agent's audit entries are AI-topical by construction. The raw novelty count is therefore not promoted to a quality claim for the multi-turn pipeline; a sizable share of that advantage is extraction from the non-AI portions of bills that pass the keyword filter, which is the same mechanism that lowered the multi-turn pipeline's coverage rate.

![Novel-claim audit: type breakdown.](../output/evals/v1/plots/05_novel_type_breakdown.png)

### Resource comparison

The multi-turn pipeline issued 21,586 LLM calls, 55.34 million tokens, at US\$228.78 over a cumulative 27.4 hours of LLM time. The skill-driven agent issued 11,231 calls, 90.91 million tokens, at US\$315.60 over 16.2 hours. The skill-driven agent is cheaper in calls and time but more expensive in tokens and dollars, because each conversation carries the full running context across turns. Against a conservative manual-coding throughput of ten bills per hour, 1,826 bills would cost roughly 183 coder-hours; the extractor replaces that with pipeline time measured in a workday's worth of compute, while keeping per-field evidence spans that manual coding would not.

## Downstream Analysis

### Distribution of targeted entities and entity types

The quadruplet's entity and type fields, pooled across bills, produce the distribution of what is being regulated. The analysis reports the top entity types by frequency, with per-state and per-year breakdowns, and attaches an example bill and span for each type so the category is auditable without re-reading the corpus.

### Entity relations, aka how the entities are regulated, promoted or punished

The attribute and value fields, conditional on entity and type, give the distribution of mechanisms that act on each entity class. A regulated entity may be required to disclose, prohibited from deploying, assessed for impact, or exempted under named conditions, and each appears as a separate attribute value in the pooled output.

### Temporal drift of regulatory focus

Pooling by year across the 2023 to 2025 window shows which targets and which mechanisms gain or lose share over time, including the 2024 shift toward generative AI and synthetic content reported independently by \cite{depaula2025evolving}. The analysis treats the year-to-year changes as descriptive rather than causal, because the corpus is the population of state AI bills rather than a sample.

### Focus vs bill outcome

Joining the quadruplet distribution to the bill's `status` field maps which entity-mechanism combinations are enacted rather than only introduced. The joint distribution is reported as a cross-tabulation, and selected cells with enough mass are inspected through the cited bills to avoid reading patterns off small counts.

## Live Demonstration

A Flask application at `src/qa/web_app.py` lets any reader submit a bill URL, a bill identifier, or pasted text and receive the extracted quadruplets together with evidence spans. Deployment is bootstrapped by `src/qa/bootstrap_app.py` and the same code path is exercised by a 100-question evaluation harness that scores the question-answering pipeline against hand-authored ground truth \cite{zheng2023judging}. The agentic version of the app passes 79 of 100 questions, against 54 for a single-pass RAG baseline and 59 for a self-query baseline, at a mean latency of 6.65 seconds. The app supports reproducibility on this paper, and serves as an invitation for other researchers to run the pipeline on new bills.

![QA app: overall pass rate across versions.](../output/evals_app/_comparison/01_overall_pass_rate.png)

![QA app: pass rate by difficulty.](../output/evals_app/_comparison/02_pass_rate_by_difficulty.png)

# Discussion

## Limitations

Four limits carry across the results. First, there is no pre-existing entity-level gold standard for the corpus, so the evaluation uses NCSL topic labels plus an LLM-as-judge protocol; a hand-labeled entity gold is follow-up work, and the judge diagnostic in §Results bounds how much trust the current protocol can carry. Second, both proposed methods run on Claude Sonnet 4.5; a cross-family comparison is scoped for follow-up rather than included here. Third, the corpus is U.S. state AI legislation in English, so claims about the pipeline's portability to other jurisdictions or languages are method-level rather than empirical. Fourth, target-versus-regulated-entity disambiguation still depends on analyst judgment at the analysis step, because a single span can appear in either role; the pipeline surfaces the two fields separately, but the separation is still read by a human.

## Implications

### Granular policy-design variables (framing a)

Once the bill is represented as a set of quadruplets with evidence, research questions that were forced to the document level, such as diffusion, adoption, and partisan structure, can be asked at the mechanism level. Instead of testing whether a state passed an AI bill, work can ask which kinds of regulated entities the state names, which instruments it uses on them, and under which conditions. This re-connects the quantitative legislative literature to the substantive content of the bill.

### Scalability to other messy policy corpora (framing b)

The runner is corpus-agnostic, and the skill file plus the taxonomy is how the extractor learns what to extract. Moving to federal bills, financial regulation, environmental regulation, or non-English legislation is taxonomy work rather than pipeline work. The skill-driven architecture also extends to a relevance pre-check for the keyword-filtered false positive described in §Data, because adding a pre-check is a change to the skill file rather than a new orchestration stage.

### Resource profile vs manual and keyword methods (framing c)

Keyword labeling is cheap but low-information. Manual coding is high-information but slow and expensive, and stops at the document level. The extractor as run here sits between the two on cost and above both on information, producing 19,000 grounded quadruplets for 1,826 bills at the combined dollar cost of both pipelines on the order of US\$544 and a combined wall time on the order of 44 hours of LLM work, plus US\$305.72 of judge time for the evaluation. The trade-off summary, for a reviewer, is that keyword labels give a topic for the dollar, manual coding gives document-level depth for the coder-week, and the extractor gives per-field quadruplets with spans for the compute-day.

### Downstream uses beyond this paper

The quadruplet corpus is usable in three further ways. First, as distant supervision for fine-tuning smaller open-source NER models on the extracted quadruplets. Second, as input to policy-design indices that aggregate fields such as strictness, coverage, enforcement, and private right of action into bill-level measures without re-reading the bills. Third, as a substrate for diffusion studies across states and over time at the entity level rather than at the bill level. The paper's companion definition paper connects how states define AI to what state AI policy targets, and the two together support joint definition-and-target analysis on the same corpus.
