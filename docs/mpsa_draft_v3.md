---

## title: ""

# Abstract

State AI bill research largely relies on document-level theme and keyword labels that treat each bill as a single undivided unit and collapse heterogeneous governance instruments into one row \cite{depaula2024regulating, depaula2025evolving}. Coarse labels cannot answer which sectors, technologies, or applications are regulated, or which actors carry obligations. We propose two extraction methods that unpack the bill. The first is a multi-turn stateful NER pipeline synthesized from recent work on cooperative multi-agent zero-shot NER \cite{wang2025cooperative}, with three stages of candidate annotation, grouping, and refinement. The second is a skill-driven agentic approach enabled by recent LLM tool calling, in which one agent follows a skill file and reads bill sections on demand to produce quadruplets in a single conversation \cite{schluntz2024building, zhang2025equipping}. Both methods extract quadruplets of entity, type, attribute, and value anchored to evidence spans, and run on 1,826 U.S. state AI bills from NCSL covering 2023 to 2025 \cite{ncsl2025ai}. We evaluate coverage of NCSL topic labels using an LLM-as-judge protocol, and report tokens, dollars, and LLM time for each method. The pipeline produces granular policy-design variables that current labels cannot reveal, transfers to other messy policy corpora with a taxonomy swap, and replaces weeks of manual coding with hours of compute while keeping traceable spans.

# Introduction

## Background

State legislatures have become a focal venue for AI lawmaking in the United States. Between 2022 and 2025, state AI legislation has surged \cite{ncsl2025ai}, with the NCSL tracker recording more than 1,200 state AI bills in 2025 alone \cite{ncsl2025ailegislation2025}. A surge of this magnitude unfolding across many sub domains including heath care, education, and ethical use of AI marks a starting point in which AI governance in the United States is being written.

Treating the AI bill as a single undivided unit without granular entity and relation extraction may be too coarse for downstream research, because the bills are bundles of heterogeneous governance instruments \cite{yoo2020regulation, kuteynikov2022key, sheikin2024principles, wang2024regulating}. Two bills can both count as "policy regulating AI" while doing fundamentally different things, such as one setting procurement rules, yet another establishing constraints on certain technologies. Collapsing this bundle into a single label breaks the downstream analysis in three ways. First is measurement error. The labels might target different procedures or entities but may measured as same category. The second problem is that the reduced effect magnitude when multiple type of relations about one concept are collapsed into one. Third is that the mechanism behind the regulated body and the method of regulation will not be revealed. Without extracting the entity and the relations, the downstream analysis will be hard to understand what policy design moved, which obligations, which targets, which enforcement.

### Why existing methods fall short

The author gathered bill corpora from NCSL and still found the annotations are too coarse to answer the question "what does it target and who does it regulate?". To address this, one may resort to dictionary or keyword based methods, but this does not seprate the target from the regulated entity on the relation to AI regulation {young2012affective, hopkins2010method}. For example, a bill may trigger the keyword "AI" in some sections, and mentioned "agent" in other sections as in chemical agents, but this may build a false positive detection of AI agent regulation.\cite{ri2024s2540}

Rule-based NER and topic modeling may carry the same limitation in a different form. The fifty states do not share a drafting convention, and a definitions clause in one state is a preamble in another and a section cross-reference in a third. A fixed pattern or a topic cluster built on bill-level text cannot follow that variation, and the output remains a theme rather than a set of policy-design variables with evidence \cite{blei2003latent, roberts2014structural, grootendorst2022bertopic}. Manual expert coding is the method that in principle can reach the bill interior, and NCSL, Brookings, and state analysts have coded subsets by hand, but coding more than a thousand bills per year is expensive, slow, and unstable across coders \cite{krippendorff2018content}, and even then the output stops at the document level rather than producing entity-and-relation content inside each bill.

With the help of large language models (LLM) and agentic workflow, the named entity recognition and relaction extraction are made easy, and the recent works show that multi-agent prompting and skill-driven agentic tool use can produce structured extractions from long text without fine-tuning \cite{wang2025cooperative, zhang2025equipping}. This paper is built upon these recent progress of LLMs.

## How this research benefits the field?

### Contribution 2. Generalizable pipeline (framing b)

example: the authoer developed the pipeline so that other researcher can use it to (DO NOT MENITONE QUAD AT HERE FFS. IT IS NOT ABOUT PIPELINE DESC)...

comment: do not repeat pipeline here. contribution is not pipeline. say how this benefits OTHER RESEARCHER. your content bear ZERO value so i remvo.ed it. start over..

### Contribution 3. Resource-efficient extraction (framing c)

Full-corpus runs are recorded in `usage_summary.json` under each run directory, and the resource table in §Results reports calls, tokens, dollars, and elapsed LLM time per method. Hours of compute replace weeks of manual coding while the cost profile is reported alongside accuracy, not substituted for it.

comment: frame it as that we researched for SAME given type of task, COMPARES WHAT TYPE OF LLM PIPELINE IS MORE EFFICIENT, as in produce less bias and wiht less resource consumption. this can be a guide for OTHER researcehr, so it is important to mention this.

### Contribution 1. Granular policy-design variables (framing a)

### Contribution 4. Open artifacts

The pipeline outputs quadruplets of entity, type, attribute, and value with a source span for each field, rather than one label per bill. This lets downstream work test mechanism-level hypotheses, such as which regulated entities are named, which targets are set, which instruments are used, and which exemptions apply, without re-reading the bill text.

The paper releases the quadruplet dataset with spans for the 1,826 state AI bills, the reusable skill file at `settings/skills/ner_extraction.md`, the runner at `src/skill_ner/`, and a live extraction demo at `src/qa/web_app.py` so any reader can submit a bill and inspect the extracted quadruplets and their evidence spans.

comment: merge contribution 1 and 4. they are both less imporatnat and only helps for downstream research.

# Literature Review

## Theory side

The qualitative legal and political-science literature on U.S. AI governance converges on one observation. States are now the main site of AI legislation, and they are producing bills faster than a shared framework can form, so the bills as a group are fragmented across sectors, uneven across states, and often built around definitions that drift from one statute to the next \cite{yoo2020regulation, kuteynikov2022key, sheikin2024principles, wang2024regulating, defranco2024assessing, agbadamasi2025navigating}. The content-focused work that does look inside bills reports a moving target. The 2019 to 2023 window is dominated by health, education, and advisory bodies; the 2024 window shifts substantially toward generative AI and synthetic content, with private-sector regulation dropping in share; and definitions remain inconsistent across states \cite{depaula2024regulating, depaula2025evolving}. A separate strand documents the governance model the bills use, with impact-assessment and accountability obligations as the emerging common shape \cite{oduro2022obligations}. The one quantitative political-economy study of state AI adoption reports that economic conditions and unified Democratic government predict adoption, while ideology and neighbor adoption do not, and that partisan structure emerges specifically around consumer-protection AI bills rather than around AI regulation in general \cite{parinandi2024investigating}. Across this literature, the empirical unit is either the bill as a whole, a hand-coded subset, or a roll call. What is inside the bill, at the level of which entities carry obligations and which mechanism is used, is treated as context rather than as a variable.

comment: 

1. previous paper find X tech/entiy are being regualted
2. you claimed "Across this literature, the empirical unit is either the bill as a whole,....hich entities carry obligations and which mechanism is used, is treated as context rather than as a variable."

this contradicts. they did do that. they find X tech/entiy are being regualted, that is the "what entity being regualted how"

if you didnt find gap, do not fabricate one. just state the facts and what is our take on it. in another owrd, how they contribute to use. this hsould be brief.

## Method side

### Non-LLM methods for policy-text analysis

#### Rule-based, keyword, and dictionary labeling

Theme tagging, keyword lists, and regex patterns applied to bill text are cheap, reproducible, and auditable, and the political-science text-as-data literature has used them for two decades \cite{grimmer2013text, young2012affective, hopkins2010method}. Recent U.S. state legislation topic infrastructure from \cite{garlick2023laboratories} and \cite{dee2025policy} shows how far labeling scales when the label set is fixed in advance. The limit is the same as in the rule-based family more generally. Heterogeneous state drafting defeats fixed patterns, and a label says whether a bill is on a topic but cannot separate the target from the regulated entity or name the instrument.

commnet: "Heterogeneous state drafting" wtf is this. what is the orignial word? i think this is lost in chains of paraphrasing. you can keep the word "heterogeneous", but use more direct word to say it. do not be obscure.

#### Topic modeling and unsupervised clustering

LDA, structural topic models, and BERTopic-style clustering extract recurring themes from a corpus without a fixed label set \cite{blei2003latent, quinn2010how, grimmer2010bayesian, roberts2014structural, lucas2015computer, grootendorst2022bertopic}. A recent policy application uses the same family to summarize AI policy documents at scale \cite{pham2026using}. The output is a theme distribution rather than a policy-design variable, and no evidence span is produced, so these methods inform descriptive work but are not a measurement instrument for what a bill actually requires.

comment:  do not mention bert here.  do not use wording that is too harsh(the last sentence). i think we solved this problem already. are you on the same page?

#### Manual expert coding

Hand-coded subsets such as those produced by NCSL, Brookings, and state analysts, backed by standard inter-coder reliability protocols \cite{krippendorff2018content}, have the highest per-bill quality of any method in the non-LLM family. They reach into bill interiors but trade coverage for accuracy, stop at the document level, and scale poorly across fifty states and multiple years.

### LLM methods for entity extraction

#### Prompt-based NER

Zero-shot and few-shot entity extraction with commercially available LLMs or locally deployable open weights models, including GPT-4, Gemini, Claude, Llama, etc., has become practical for well-bounded text. Clinical NER \cite{hu2023improving, islam2025llm}, non-English clinical reports with long-context prompts \cite{akcali2025automated}, and cyber threat intelligence \cite{feng2025promptbart} all report workable accuracy from prompt engineering alone. For the open source models deploying on local machines, the common limit is document length. Legislative bills are long, structurally varied, and cross-referenced, and a single-pass prompt has no way to read one section and then decide what to read next. While for the commercially available models, the common limit is cost and ethical guardrails. Some models have tendency of ommiting content related to security and privacy when these entities are burried in long context.

#### Multi-stage NER pipeline

A second family splits NER across cooperating agents or across a several stage pipeline with stacked prompts, so that each sub-step operates on a narrower input. Wang et al.'s cooperative multi-agent framework is the representative example for zero-shot NER, with self-annotator, type-related feature extractor, demonstration discriminator, and overall predictor each doing one sub-step \cite{wang2025cooperative}. Two-stage locate-then-type models \cite{ye2023decomposed}, entity structure discovery \cite{xu2025zeroshot}, API entity and relation joint extraction \cite{huang2023api}, relation-classification agent architectures \cite{berijanian2025comparative}, automatic labeling for sensitive text \cite{deandrade2025promptner}, and benchmark construction for scholarly NER \cite{otto2023gsapner} round out the family. These designs handle long, varied inputs better than a single pass but still require a fixed plan, so they do not decide on the fly which bill section to look at next.

comment: "Wang et al.'s " wrong citation style. use latex here.
the @NER_agent_architecture_references.md very very clearly stated "WAHT DO THEY MEAN FOR US; WHAT WE CAN BENEFIT FROM THEM? AND WHY WE CANNOT DIRECTLY USE THEIRS". this is more important than revisting each paper and summarize what they did.

#### Agentic and skill-driven workflows

Retrieval-augmented generation \cite{lewis2020retrieval}, the ReAct pattern that interleaves reasoning and acting \cite{yao2023react}, and tool-use training \cite{schick2023toolformer} laid the groundwork for agents that read a document in sections and refine across turns. Recent agentic RAG surveys \cite{singh2025agentic}, search-augmented reasoning \cite{li2025searcho1, jin2025searchr1}, and the Anthropic skills line that loads a markdown skill file as system prompt and grants structured tool access \cite{schluntz2024building, zhang2025equipping, anthropic2025skills, anthropic2025sdk} together define the design space for a single agent that can read a bill in sections and produce quadruplets in one conversation. Cost, reliability, and evaluation on real policy corpora in this family are still largely untested.

## Our take on existing methods

comment: wrong content. this is not pipline description. this is about what we can benifit from them (multi stage and agentic). and why we cannot directly use theirs(multi stage) and why agentci can be better.

# Method

## Task Definitions

comment: this is not task definition. yours is pipeline description.

## Model Setup

The model is Claude Sonnet 4.5 reached through OpenRouter. Temperature is 0.0 and `max_tokens` is set to 16,384, which is Sonnet 4.5's hard completion ceiling.\footnote{Token-ceiling selection and the failure modes that pushed it to the model's physical ceiling are documented in \texttt{docs/log0.31nercrashes.md} C3.} Running through OpenRouter lets both methods call the same weights, so a cost and accuracy comparison between the two methods is not confounded by model choice.

comment: what 2 methods? this is cited before defined.

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

comment: no, we dont hava gold standard. that is why we test" coverage" and "novelty" instead. there are detaield documenation for this. @proposed_methods_eval.md, @lit_rev_eval.md

### Cross method evaluation

#### Performance comparison

Both methods run on the same full population of the bills, and the LLM judge runs at Gemini 2.5 Pro with temperature 0.0. Four tests are reported. The first is per-quadruplet grounding against the bill text, which gives an item-level accuracy signal that does not depend on the label set. The second is set-to-label coverage against NCSL topics, which gives a reference-aligned signal. The third is a cross-method pairwise comparison in which the judge sees both methods' quadruplet sets for the same bill and picks one, with swap-averaging across presentation order to cancel position bias \cite{zheng2023judging}. The fourth is a judge bias audit on a pooled 100-row sample with four perturbations, which sets a trust bound on the first three tests rather than ranking the methods \cite{ye2024justice, guerdan2025validating, tan2024judgebench}.

comment: "and the LLM judge runs at Gemini 2.5 Pro with temperature 0.0" epxlain why this. it was documented. 

#### Resource comparison

For each method, the paper reports calls, prompt and completion tokens, dollar cost, and cumulative LLM time, drawn from `usage_summary.json` in each run directory. Per-bill averages are computed from the totals and the 1,826-bill corpus. The cost table sits alongside the accuracy tables, so the trade-off between the two methods and against human coding is visible on the same page.

# Data

## Data Source

The corpus is state AI legislation tracked by the National Conference of State Legislatures for 2023, 2024, and 2025 \cite{ncsl2025ai, ncsl2023ailegislation2023, ncsl2024ailegislation2024, ncsl2025ailegislation2025}, with underlying data supplied by LexisNexis State Net \cite{lexisnexis2025statenet}. Bill text is scraped from each bill's canonical URL by a Selenium crawler, and a metadata file at `data/ncsl/us_ai_legislation_ncsl_meta.csv` records, per bill, the fields `state`, `year`, `bill_id`, `bill_url`, `title`, `status`, `date_of_last_action` when available, `author` with partisanship when available, `topics`, `summary`, `history`, and `text`. Bill text is stored separately from metadata and joined by `bill_id`. The merged corpus covers 1,879 rows; 53 empty-text rows are filtered out; and 1,826 bills enter the pipeline.

## Descriptive Statistics

The bill count is distributed across years as 137 in 2023, 480 in 2024, and 1,262 in 2025. Among 2025 bills, 192 have a status beginning with `Enacted` across 45 states, with California carrying the highest count at 24. Bill-length distribution and per-state counts are produced by `scripts/desc_stat.py`; coverage of existing NCSL theme labels is reported alongside the method results in §Results; and party composition is reported where the `author` field carries partisanship.



commnet: you shuold add some plots here. if not, create a scripts/desc_plots.py and delegate to subagents.

# Results

comment: this part should be ground on the 4 tests, not 9 stages. results pltos should be inserted. i dont know why you keep refusing this. is it hard for you? 

## Baseline vs Proposed Method


### Per-quadruplet grounding

The first test asks the judge whether each surviving quadruplet is supported by the bill text. The multi-turn pipeline records 85.05% entailed, 11.55% neutral, and 3.39% contradicted. The skill-driven agent records 78.32% entailed, 18.07% neutral, and 3.61% contradicted. The contradiction rate, which is the most direct failure signal in this test, is indistinguishable across methods. The entailed-rate gap is confounded with volume, because the multi-turn pipeline was judged on 1,503 more quadruplets than the skill-driven agent. The separator in this test is the neutral rate, which is higher for the skill-driven agent, consistent with its tendency to paraphrase values; this also explains why every pre-filter failure on the skill-driven agent is of type `span_not_literal` (629 of 629) while the multi-turn pipeline's pre-filter failures are dominated by missing fields and missing spans.

Grounding verdict distribution by method. (wtf is this? where is the result plots?do you know syntax?)

### Set-to-label coverage

The second test asks the judge whether, for each bill and each NCSL topic label on that bill, the method's surviving quadruplets jointly account for the label. On the permissive definition that counts both covered and partially covered, the skill-driven agent leads by 3.33 pp (87.04% vs 84.03%). On the strict definition that counts only covered, the skill-driven agent leads by 32.06 pp (81.50% vs 49.44%), and its error rate is roughly forty times smaller (0.03% vs 1.30%), driven by the multi-turn pipeline's longer supporting lists overflowing the judge prompt. Because NCSL topic labels are the reference for the downstream analyses below, the strict-coverage gap is the outcome-relevant number.

Coverage by method and cost-vs-coverage.

### Cross-method pairwise comparison

The third test shows the judge both methods' quadruplet sets for the same bill and asks it to pick one. Each of the 1,693 bills in the intersection is judged twice with presentation order swapped, so position bias is canceled by averaging \cite{zheng2023judging}. Both presentation orders prefer the skill-driven agent, with a swap-averaged win rate of 75.99% against 22.68% for the multi-turn pipeline. A count-normalised variant that adjusts for the fact that the multi-turn pipeline writes more quadruplets cuts the raw gap roughly in half, and the skill-driven agent still leads by 3.4× in normalised points (0.3791 vs 0.1126).

Pairwise win rate.

### Judge bias audit

The fourth test perturbs the coverage prompt on a pooled 100-row sample and measures how often the judge flips its verdict. Position flips 3%, verbosity flips 4%, self-preference flips 7%, and authority flips 14%. The first three are at or near the noise range for this kind of audit. Authority is the outlier; a 14% sensitivity to a prefix that claims a senior expert insists on a verdict means any prompt that carries authority-aligned language has to be treated as a rerun condition. Inspection of the frozen prompts confirms that neither the grounding prompt nor the coverage prompt carries authority cues, so the results above are not contaminated by this channel.

Bias audit scorecard.

#### Discussion: skill-driven vs fixed-plan three-stage method vs theme and keyword labels

Of the three extractor-level tests, two separate the two methods and both point the same way. Set-to-label coverage puts the skill-driven agent 32.06 pp ahead on the strict definition, and cross-method pairwise comparison puts the skill-driven agent ahead 75.99% to 22.68% after swap-averaging. Per-quadruplet grounding does not separate the two methods on the contradiction signal. The judge diagnostic finds no trust failure on the prompts in use. Against NCSL's own document-level labels, both proposed methods produce bill-interior variables that the labels themselves cannot carry, so the label-level comparison is not a head-to-head but a statement that the proposed methods add a dimension the baseline does not have.

Under the audit of the grounded-but-uncited quadruplets, the multi-turn pipeline retains 6,319 entries and the skill-driven agent retains 1,730. The multi-turn pipeline's audit sample splits into bill-relevant specifics that NCSL's topic tags do not name and bill-adjacent entries that are off-topic relative to AI policy, such as tax credits, demonstrated-mastery assessments, and nuclear energy in bills that happen to mention AI. The skill-driven agent's audit entries are AI-topical by construction. The raw novelty count is therefore not promoted to a quality claim for the multi-turn pipeline; a sizable share of that advantage is extraction from the non-AI portions of bills that pass the keyword filter, which is the same mechanism that lowered the multi-turn pipeline's coverage rate.

Novel-claim audit: type breakdown.

### Resource comparison

The multi-turn pipeline issued 21,586 LLM calls, 55.34 million tokens, at US228.78 over 27.4 cumulative hours of LLM time. The skill-driven agent issued 11,231 calls, 90.91 million tokens, at US315.60 over 16.2 hours. Normalized to the 1,826-bill corpus, the multi-turn pipeline averages 11.8 calls and US0.125 per bill, and the skill-driven agent averages 6.2 calls and US0.173 per bill. The skill-driven agent is cheaper in calls and time but more expensive in tokens and dollars, because each conversation carries the full running context across turns. The evaluation layer added US305.72 and 29,451 judge calls over 78.8 hours of cumulative judge time for the four tests combined.

#### Discussion: cost and accuracy comparison

## 

## Live Demonstration

A Flask application at `src/qa/web_app.py` lets any reader submit a bill URL, a bill identifier, or pasted text and receive the extracted quadruplets together with evidence spans. Deployment is bootstrapped by `src/qa/bootstrap_app.py`, and the same code path is exercised by a 100-question evaluation harness that scores the question-answering pipeline against hand-authored ground truth. The agentic version of the app passes 79 of 100 questions, against 54 for a single-pass RAG baseline and 59 for a self-query baseline, at a mean latency of 6.65 seconds. The app supports reproducibility on this paper, and serves as an invitation for other researchers to run the pipeline on new bills.

QA app: overall pass rate across versions.

QA app: pass rate by difficulty.

# Discussion

## Limitations

Four limits carry across the results. First, there is no pre-existing entity-level gold standard for the corpus, so the evaluation uses NCSL topic labels plus an LLM-as-judge protocol; a hand-labeled entity gold is follow-up work, and the judge diagnostic in §Results bounds how much trust the current protocol can carry. Second, both proposed methods run on Claude Sonnet 4.5; a cross-family comparison is scoped for follow-up rather than included here. Third, the corpus is U.S. state AI legislation in English, so claims about the pipeline's portability to other jurisdictions or languages are method-level rather than empirical. Fourth, target-versus-regulated-entity disambiguation still depends on analyst judgment at the analysis step, because a single span can appear in either role; the pipeline surfaces the two fields separately, but the separation is still read by a human.

## Implications

comment: this is duplication writing. do not remention every single thing. this should be higher level writing. 

### Granular policy-design variables (framing a)

Once the bill is represented as a set of quadruplets with evidence, research questions that were forced to the document level, such as diffusion, adoption, and partisan structure, can be asked at the mechanism level. Instead of testing whether a state passed an AI bill, work can ask which kinds of regulated entities the state names, which instruments it uses on them, and under which conditions. This ties quantitative legislative research back to bill content.

### Scalability to other messy policy corpora (framing b)

The runner is corpus-agnostic, and the skill file plus the taxonomy is how the extractor learns what to extract. Moving to federal bills, financial regulation, environmental regulation, or non-English legislation is taxonomy work rather than pipeline work. The skill-driven architecture also extends to a relevance pre-check for the keyword-filtered false positive described in §Data, because adding a pre-check is a change to the skill file rather than a new orchestration stage.

### Resource profile vs manual and keyword methods (framing c)

Keyword labeling is cheap but low-information. Manual coding is high-information but slow and stops at the document level. The extractor sits between the two on cost and above both on information: it produces roughly 19,000 grounded quadruplets for 1,826 bills at a combined dollar cost on the order of US544 and roughly 44 cumulative hours of LLM time, with US305.72 and 78.8 hours of judge time for the four evaluation tests. Manual coding at the depth used in \cite{depaula2024regulating, depaula2025evolving} was applied to 68 and 79 enacted bills respectively; the 1,826-bill corpus here is more than an order of magnitude larger, and the extractor produces per-field evidence spans that document-level coding does not include. The trade-off summary, for a reviewer, is that keyword labels give a topic at the dollar, manual coding gives document-level depth at the coder-week, and the extractor gives per-field quadruplets with spans at the compute-day.

### Downstream uses beyond this paper

The quadruplet corpus is usable in three further ways. First, as distant supervision for fine-tuning smaller open-source NER models on the extracted quadruplets. Second, as input to policy-design indices that aggregate fields such as strictness, coverage, enforcement, and private right of action into bill-level measures without re-reading the bills. Third, as a base for diffusion studies across states and over time at the entity level rather than at the bill level. The paper's companion definition paper connects how states define AI to what state AI policy targets, and the two together support joint definition-and-target analysis on the same corpus.