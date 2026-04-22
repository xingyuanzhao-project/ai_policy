Section structure mirrors `docs/mpsa_draft_outline.md` Method side (lines 77-121): non-LLM methods for policy-text analysis, LLM methods for entity extraction, and agentic / skill-driven workflows.

---

# Non-LLM methods for policy-text analysis

## Rule-based, keyword, and dictionary labeling

**(Grimmer & Stewart, 2013)**
### Text as Data: The Promise and Pitfalls of Automatic Content Analysis Methods for Political Texts
- https://doi.org/10.1093/pan/mps028
- method: survey of automated content-analysis methods for political text, including dictionary / keyword methods, supervised classifiers, and unsupervised topic models. Establishes four principles: (1) all quantitative language models are wrong but some are useful, (2) methods amplify human resources rather than replace reading, (3) there is no globally best method, (4) validate on every corpus. Positions dictionary methods as cheap, reproducible, and auditable but brittle when applied outside the domain of construction.
- required data: a target text corpus and a hand-built dictionary of keywords or word patterns per category; optionally a human-coded validation subset; no training labels required.

**(Young & Soroka, 2012)**
### Affective News: The Automated Coding of Sentiment in Political Texts
- https://doi.org/10.1080/10584609.2012.671234
- method: Lexicoder Sentiment Dictionary (LSD); a bag-of-words dictionary of 2,858 negative and 1,709 positive word patterns plus 2,860 and 1,721 negated forms. Scoring is a simple per-document word-pattern count; validated against human-coded news and benchmarked against nine prior dictionaries.
- required data: the pre-built LSD (or a custom dictionary) and a human-coded validation subset to check dictionary behaviour on the target corpus; no training labels.

**(Garlick, 2023)**
### Laboratories of Politics: There is Bottom-Up Diffusion of Attention in the American Federal System
- https://doi.org/10.1177/10659129211068059  (Political Research Quarterly 76(1): 29-43)
- method: dictionary approach that runs curated keyword searches over LexisNexis State Capital Universe to place state bills into 22 policy areas (1991-2008, extended to 2018). Used to measure state-level policy attention and test bottom-up federalism diffusion. Demonstrates high precision when keywords fire but limited recall because many bills never contain any dictionary keyword.
- required data: full-text state bill corpus searchable by keyword; hand-constructed keyword dictionary per policy area; validation set of hand-coded bills.

**(Dee & Garlick, 2025)**
### Policy Agendas of the American State Legislatures
- https://doi.org/10.1038/s41597-025-05621-5  (Scientific Data 12: 1276)
- method: benchmarks the legacy Garlick (2023) dictionary method against a transformer-based classifier with a three-pass decision function that codes 1.36M state bills since 2009 into 28 policy areas. The dictionary baseline covers only ~41% of bills (false negatives when keyword absent) while the transformer matches hand-coder accuracy with much broader coverage. Establishes why dictionary-only methods under-cover state legislative agendas.
- required data: Legiscan bill descriptions + titles; hand-coded validation set (Pennsylvania Policy Database Project); extended 28-category codebook.

**(NCSL, 2025)**
### NCSL Artificial Intelligence Legislation Database
- https://www.ncsl.org/financial-services/artificial-intelligence-legislation-database
- method: hybrid manual + keyword tracker. NCSL staff read each incoming bill and assign topics from a fixed taxonomy (Appropriations, Audit, Deepfake, Government Use, Health Use, Oversight/Governance, Private Right of Action, etc.). The user-facing keyword field searches bill titles and summaries only, not full text. Powered by LexisNexis State Net.
- required data: state bill metadata feed; NCSL-curated topic taxonomy; ongoing staff time for per-bill topic assignment.

## Topic modeling and unsupervised clustering

**(Blei, Ng & Jordan, 2003)**
### Latent Dirichlet Allocation
- https://www.jmlr.org/papers/v3/blei03a.html  (JMLR 3: 993-1022)
- method: LDA; three-level hierarchical Bayesian mixture model where each document is a finite mixture over latent topics and each topic is a distribution over words. Variational EM for inference. The canonical unsupervised topic model that later political-science topic models extend.
- required data: a bag-of-words document-term matrix; number of topics K as a hyperparameter; no labels.

**(Quinn, Monroe, Colaresi, Crespin & Radev, 2010)**
### How to Analyze Political Attention with Minimal Assumptions and Costs
- https://doi.org/10.1111/j.1540-5907.2009.00427.x  (American Journal of Political Science 54(1): 209-228)
- method: dynamic multinomial topic model for legislative speech. One topic per speech, dynamic prior over day-level topic proportions, jointly estimates topic keywords, topic substance, and temporal attention. First topic model built directly for political text.
- required data: 118,000+ U.S. Senate floor speeches (Congressional Record, 1997-2004) with date and speaker metadata; no topic labels.

**(Grimmer, 2010)**
### A Bayesian Hierarchical Topic Model for Political Texts: Measuring Expressed Agendas in Senate Press Releases
- https://doi.org/10.1093/pan/mpp034  (Political Analysis 18(1): 1-35)
- method: Expressed Agenda Model; hierarchical Bayesian topic model with one topic per press release and senator-level topic-attention vectors, exploiting the author structure of the corpus. Variational inference. Designed to measure how much attention each senator allocates to each topic.
- required data: 24,000+ senator press releases from 2007 with author and date metadata; no topic labels.

**(Roberts, Stewart, Tingley, Lucas, Leder-Luis, Gadarian, Albertson & Rand, 2014)**
### Structural Topic Models for Open-Ended Survey Responses
- https://doi.org/10.1111/ajps.12103  (American Journal of Political Science 58(4): 1064-1082)
- method: Structural Topic Model (STM); extends LDA to allow document-level covariates (author gender, party, treatment assignment, year) to influence both topic prevalence (how often a topic is discussed) and topic content (the words used). Supports treatment-effect estimation over estimated topics. Implemented in the `stm` R package.
- required data: a document corpus with per-document covariates; number of topics K; no topic labels.

**(Lucas, Nielsen, Roberts, Stewart, Storer & Tingley, 2015)**
### Computer-Assisted Text Analysis for Comparative Politics
- https://doi.org/10.1093/pan/mpu019  (Political Analysis 23(2): 254-277)
- method: applies the STM to multilingual political text with pipelines for language-specific pre-processing, management, translation, and validation. Demonstrates on Arabic fatwas and on Arabic + Chinese Snowden-era social-media responses, including STM over machine-translated text.
- required data: multi-language corpora; per-document covariates; optionally a machine-translation tool to unify languages before modelling.

**(Grootendorst, 2022)**
### BERTopic: Neural Topic Modeling with a Class-Based TF-IDF Procedure
- https://arxiv.org/abs/2203.05794
- method: BERTopic pipeline. (1) Embed documents with a pretrained sentence transformer (e.g. all-mpnet-base-v2); (2) reduce dimensionality via UMAP; (3) cluster with HDBSCAN; (4) extract per-cluster topic representations via class-based TF-IDF (c-TF-IDF), which treats each cluster as one concatenated document. Modular: clustering and topic extraction are decoupled.
- required data: a document corpus; a pretrained sentence-embedding model; no labels.

**(Pham, Radiya-Dixit, Gerchick, Madubuonwu & Venkatasubramanian, 2026)**
### Using AI to Make Sense of AI Policy (Brown CNTR + ACLU report; CNTR AISLE Portal)
- https://www.aclu.org/publications/using-ai-to-make-sense-of-ai-policy
- method: topic modeling over 1,804 state and federal AI-related bills (2023 - April 2025) to identify dominant themes (generative AI, task-force bills, deepfakes). Also applies pairwise text similarity to trace diffusion of template / model bills across states, and graph-theoretic analysis of intra-bill definition cycles to surface ambiguous legal language. Closest published analogue to this project's corpus.
- required data: curated keyword filter to select AI-related bills; full text scraped from Legiscan plus federal sources; definition graphs extracted per bill.

## Manual expert coding

**(Krippendorff, 2018)**
### Content Analysis: An Introduction to Its Methodology (4th edition)
- https://doi.org/10.4135/9781071878781  (Sage; also 2nd ed. 2004, 3rd ed. 2013)
- method: canonical methodology for systematic human content coding: unitizing, sampling, recording / coding, and reliability measured by Krippendorff's alpha (alpha = 1 - D_observed / D_expected, alpha >= 0.8 acceptable). Treats coders as interchangeable measurement instruments; applies to nominal, ordinal, interval, and ratio data with missing values and any number of coders.
- required data: a document corpus; a codebook with explicit mutually-exclusive category definitions; at least two independently trained coders; a reliability sample drawn from the target corpus.

**(Hopkins & King, 2010)**
### A Method of Automated Nonparametric Content Analysis for Social Science
- https://doi.org/10.1111/j.1540-5907.2009.00428.x  (American Journal of Political Science 54(1): 229-247)
- method: ReadMe; directly estimates the proportion of documents in each category rather than classifying each document. Uses a small hand-coded subset plus word-stem profiles to recover approximately unbiased category proportions even when any single-document classifier is weak. Distributed as the `ReadMe` R package.
- required data: a small hand-coded training subset (order of a few hundred documents), a larger unlabeled target corpus, and exhaustive mutually-exclusive category definitions.

**(NCSL, 2025)**  (see same entry under Rule-based, keyword, and dictionary labeling)
### NCSL Artificial Intelligence Legislation Database (manual-coding side)
- https://www.ncsl.org/financial-services/artificial-intelligence-legislation-database
- method: per-bill topic assignment by NCSL analysts over a ~20-label taxonomy. Illustrates the coverage-accuracy trade-off flagged in `docs/mpsa_draft_outline.md` (77-121): high fidelity per bill, slow to scale across 50 states and multiple years, document-level only.
- required data: ongoing analyst time; fixed topic list; bill metadata + full text.

---

# LLM methods for entity extraction

Reference anchors: `references/references_methods.md` (general) and `docs/NER_agent_architecture_references.md` (architecture-level breakdowns).

## Prompt-based NER

**(Hu et al., 2023)**
### Improving Large Language Models for Clinical Named Entity Recognition via Prompt Engineering
- https://arxiv.org/abs/2303.16416
- method: evaluates GPT-3.5 and GPT-4 on MTSamples and VAERS clinical NER using a four-component prompt framework: (1) baseline prompt with task description + output format spec, (2) annotation-guideline-derived instructions, (3) error-analysis-derived corrections, (4) a small bank of annotated few-shot examples. No fine-tuning.
- required data: task specification; annotation guidelines per entity type; a handful of hand-annotated few-shot examples; per-error-mode diagnostic notes; test clinical text.

**(Islam et al., 2025)**
### LLM-based Prompt Ensemble for Reliable Medical Entity Recognition from EHRs
- https://arxiv.org/abs/2505.08704
- method: prompt ensemble for GPT-4o and DeepSeek-R1. Combines four prompt configurations (zero-shot; few-shot with one XML-tagged document; few-shot with 100 XML-tagged sentences from 5 documents; few-shot with 5,355 category-grouped entities from 73 documents). Candidate entities are embedded with ClinicalBERT, clustered by cosine similarity (tau = 0.92), and labelled by majority vote (>=2 agreement; else `unknown`).
- required data: zero-shot base prompt; annotated document, sentence, and entity banks; ClinicalBERT (or equivalent) for embedding; test EHR text.

**(Akcali et al., 2025)**
### Automated Extraction of Key Entities from Non-English Mammography Reports Using Named Entity Recognition with Prompt Engineering
- https://doi.org/10.3390/bioengineering12020168  (Bioengineering 12(2): 168)
- method: Gemini 1.5 Pro with a 1,195-line, 165-shot, ~26,000-token prompt packaging annotation guidance and many demonstrations across five entity types (ANAT, IMP, OBS-P, OBS-A, OBS-U). Pure long-context many-shot prompting on Turkish mammography reports; no fine-tuning. Scored with relaxed-match and exact-match F1.
- required data: a large set of annotated Turkish mammography reports for in-prompt demonstrations; a long-context model; explicit annotation guidance per entity type.

## Multi-agent and two-stage decomposition

**(Wang, Zhao, Lyu, Chen, de Rijke & Ren, 2025)**
### A Cooperative Multi-Agent Framework for Zero-Shot Named Entity Recognition
- https://arxiv.org/abs/2502.18702
- method: CMAS, four cooperating agents. (1) Self-Annotator labels candidate demonstrations from an unannotated corpus. (2) Type-Related Feature (TRF) Extractor derives TRFs R^q for the target sentence and R_i for each example. (3) Demonstration Discriminator scores example helpfulness h_i via self-reflection. (4) Overall Predictor produces the final (entity mention, entity type) set using helpful TRF-tagged examples. Zero-shot; no labelled training data. Primary architecture candidate for this project (see `docs/NER_method.md`).
- required data: unannotated target corpus; entity-type label set (schema or taxonomy); no gold annotations.

**(Berijanian, Singh & Sehati, 2025)**
### Comparative Analysis of AI Agent Architectures for Entity Relationship Classification
- https://arxiv.org/abs/2506.02426
- method: benchmarks three LLM agent architectures on entity-relation classification: (a) reflective self-evaluation, (b) hierarchical task decomposition, and (c) multi-agent dynamic example generation with real-time cooperative and adversarial prompting. Compared against standard few-shot prompting and fine-tuned baselines under limited labelled data.
- required data: a relation-classification task with entity-pair contexts; a label set for relation types; domain-representative test examples; no task-specific fine-tuning required.

**(Ye, Huang, Liang & Chi, 2023)**
### Decomposed Two-Stage Prompt Learning for Few-Shot Named Entity Recognition
- https://doi.org/10.3390/info14050262  (Information 14(5): 262)
- method: locate-then-type decomposition. Stage I trains an entity-locating model with distant labels and BIOES boundaries to predict candidate spans. Stage II fills a prompt template `[entity] [MASK]` aligned with the masked-LM pre-training objective and uses an entity-typing model to predict the `[MASK]` label. Final output merges spans and types.
- required data: distantly labelled data for span supervision; a small set of typed few-shot entities for stage II; BIOES-format data.

**(Xu et al., 2025)**
### Zero-Shot Open-Schema Entity Structure Discovery
- https://arxiv.org/abs/2506.04458
- method: ZOES, a three-stage enrich-refine-unify pipeline. (i) zero-shot LLM extraction of (entity, attribute, value) triplets from context; (ii) attribute clustering via dense encoder + agglomerative clustering to induce root attributes, then value-anchored enrichment to fill missing values under each root attribute; (iii) mutual-dependency question-answer prompting refines coarse / inconsistent triplets, then merges refined triplets into per-entity attribute-value structures with type filtration. Training-free prompting pipeline. Directly adjacent to the quadruplet schema used in `docs/mpsa_draft_outline.md` and `docs/NER_agent_architecture_references.md`.
- required data: target corpus; dense encoder for attribute clustering; target entity type(s) for final filtering; no labelled data.

## Agentic / skill-driven workflows

### Foundational augmentation: Retrieval-Augmented Generation

**(Lewis et al., 2020)**
### Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- https://arxiv.org/abs/2005.11401  (NeurIPS 2020)
- method: RAG; couples a pre-trained seq2seq generator with a differentiable dense-vector retriever over a non-parametric memory (Wikipedia). Two formulations: RAG-Sequence (condition on one retrieved passage across the whole output) and RAG-Token (different passage per token). Fine-tuned end-to-end. Establishes retrieval + generation as the base building block later extended by agentic patterns.
- required data: non-parametric passage index (e.g. dense-vector Wikipedia); query-answer training pairs for fine-tuning; pre-trained seq2seq generator and pre-trained dense retriever.

### Tool-use and agent workflow patterns

**(Yao, Zhao, Yu, Du, Shafran, Narasimhan & Cao, 2023)**
### ReAct: Synergizing Reasoning and Acting in Language Models
- https://arxiv.org/abs/2210.03629  (ICLR 2023 Oral)
- method: ReAct; prompts an LLM to interleave reasoning traces (`Thought:`) with task-specific actions (`Action:`) and environment observations (`Observation:`) in a loop. Reasoning steps plan and update action plans; actions fetch external evidence (Wikipedia API, environment). Reduces hallucination on HotpotQA and Fever; outperforms imitation + RL baselines on ALFWorld (+34%) and WebShop (+10%) with 1-2 in-context examples.
- required data: 1-2 Thought / Action / Observation demonstrations; a tool or environment API the agent can call; no fine-tuning.

**(Schick, Dwivedi-Yu, Dessi, Raileanu, Lomeli, Zettlemoyer, Cancedda & Scialom, 2023)**
### Toolformer: Language Models Can Teach Themselves to Use Tools
- https://arxiv.org/abs/2302.04761  (NeurIPS 2023)
- method: self-supervised tool use. The model samples candidate API calls at each position, keeps only the calls whose returned values reduce the language-modelling loss on downstream tokens, and fine-tunes on the filtered annotated corpus. Supports calculator, QA system, two search engines, translation, calendar. Shows that a 6.7B GPT-J can match much larger models on tool-benefiting tasks.
- required data: unlabelled text corpus; a handful of in-context demonstrations per API; simple callable API endpoints; fine-tuning budget on the filtered corpus.

**(Schluntz & Zhang, 2024)**  (Anthropic Engineering Blog)
### Building Effective AI Agents
- https://www.anthropic.com/engineering/building-effective-agents
- method: catalogue of composable agentic patterns observed across dozens of production deployments. Building block: the augmented LLM (LLM + retrieval + tools + memory). Workflows: (1) prompt chaining, (2) routing, (3) parallelization (sectioning / voting), (4) orchestrator-workers (central LLM dynamically delegates to worker LLMs and synthesises), (5) evaluator-optimizer (generator + critic loop). Agents: LLM-in-a-loop that plans, uses tools, observes environmental feedback, and stops on completion or iteration cap. Recommends starting from direct LLM API calls and adding agentic complexity only when evaluation demonstrates gain.
- required data: task-specific evaluation set; clear tool / API surface per task; prompt templates; no training beyond prompt engineering.

### Agentic RAG and agentic search

**(Singh, Ehtesham, Kumar & Talaei Khoei, 2025)**
### Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG
- https://arxiv.org/abs/2501.09136
- method: survey. Distinguishes Naive RAG, Advanced RAG, and Agentic RAG. Introduces a taxonomy of Agentic RAG architectures by agent cardinality (single vs multi-agent), control structure (sequential, hierarchical, adaptive), autonomy level, and knowledge representation. Consolidates four core agentic patterns: reflection, planning, tool use, and multi-agent collaboration; reviews applications in healthcare, finance, education, and enterprise document processing. Companion repository: https://github.com/asinghcsu/AgenticRAG-Survey.
- required data: N/A (survey); synthesises published frameworks.

**(Li, Dong, Jin, Zhang, Zhou, Zhu, Zhang & Dou, 2025)**
### Search-o1: Agentic Search-Enhanced Large Reasoning Models
- https://arxiv.org/abs/2501.05366  (EMNLP 2025 Oral)
- method: augments large reasoning models (OpenAI-o1, QwQ-32B) with an agentic RAG loop. When the reasoning chain hits uncertainty, the model emits a special search token, retrieves external documents, and routes them through a separate Reason-in-Documents module that condenses the retrieved text before splicing a compact answer back into the main reasoning chain. Evaluates on GPQA, math, coding, and six open-domain QA benchmarks.
- required data: a large reasoning model backbone; a retriever plus external corpus (web or Wikipedia); Reason-in-Documents prompt; no gradient training at inference time.

**(Jin, Zeng, Yue, Yoon, Arik, Wang, Zamani & Han, 2025)**
### Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
- https://arxiv.org/abs/2503.09516  (COLM 2025)
- method: extends DeepSeek-R1-style RL to interleave step-by-step reasoning with multi-turn search engine calls. Retrieved tokens are masked from the RL loss to stabilise training; a simple outcome-based reward drives the policy. Teaches the model when to query, what to query, and how to integrate retrieved evidence. Improves accuracy by 10-41% over RAG baselines on seven QA benchmarks.
- required data: multi-hop QA datasets (7 used); a live search engine or retrieval index; an RL-capable base LLM (Qwen2.5-3B/7B, LLaMA3.2-3B).

### Anthropic Agent Skills (filesystem-based procedural knowledge)

**(Zhang, Lazuka & Murag, 2025)**  (Anthropic Engineering Blog)
### Equipping Agents for the Real World with Agent Skills
- https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- method: Agent Skills; filesystem-based folders, each containing a `SKILL.md` with required YAML frontmatter (`name`, `description`) plus optional referenced files (`reference.md`, `forms.md`) and executable scripts. Progressive disclosure in three levels: (1) metadata always loaded into the system prompt (~100 tokens per skill), (2) SKILL.md body read via Bash when the agent judges the skill relevant, (3) additional files / code loaded only when referenced. Skills turn general-purpose agents into specialists without bloating the context window, and can bundle deterministic code for operations where token-level generation is too expensive.
- required data: an agent runtime with filesystem + code-execution tools; one `SKILL.md` per skill with clear `name` and `description`; optional bundled scripts and reference files; a task-representative evaluation set for iteration.

**Anthropic (2025)**
### Agent Skills (Claude API Docs, Overview)
- https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview
- method: official specification of the Skill contract used across Claude.ai, Claude Code, the Claude Agent SDK, and the Claude Developer Platform. Each skill is a directory with `SKILL.md` (YAML frontmatter `name`, `description`). Pre-built skills ship for PDF, Excel, Word, and PowerPoint; custom skills are placed in `.claude/skills/` (project) or `~/.claude/skills/` (user). Skills are discovered at startup via metadata, loaded in full only on trigger, and extended lazily through referenced files and executable code in a VM.
- required data: agent runtime with filesystem access (e.g. Claude Agent SDK, Claude Code, claude.ai); one or more `SKILL.md` directories per capability; optional data / script dependencies bundled alongside.

**Anthropic (2025)**
### Agent Skills in the Claude Agent SDK
- https://docs.claude.com/en/agent-sdk/skills
- method: SDK-level documentation. Agent SDK (Python and TypeScript) exposes the same skill discovery model used by Claude Code, under `.claude/skills/*/SKILL.md`, alongside custom slash commands, project memory (`CLAUDE.md`), and plugins. Skills interoperate with MCP servers: MCP delivers live data / tool access, while skills encode reusable procedural knowledge.
- required data: Claude Agent SDK installation; `settingSources: ['project']` set in options; `.claude/skills/` directory populated with skill folders.
