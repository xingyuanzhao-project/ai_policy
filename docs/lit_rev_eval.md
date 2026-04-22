# non llm eval for NER and IE

## NER (span-level)

**(Tjong Kim Sang & De Meulder, 2003)**

### Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition

- [https://aclanthology.org/W03-0419/](https://aclanthology.org/W03-0419/)
- method of eval: entity-level exact-match Precision / Recall / F1. A predicted entity is counted as correct only if both its type label and its full span boundaries match a gold entity exactly. Released the `conlleval` Perl scorer that became the de facto CoNLL-2003 scoring program.
- required data: gold-standard named-entity annotations (PER, LOC, ORG, MISC) in BIO tagging on Reuters English / German; paired gold and predicted BIO sequences.

**(Chinchor & Sundheim, 1993)**

### MUC-5 Evaluation Metrics

- [https://aclanthology.org/M93-1007/](https://aclanthology.org/M93-1007/)
- method of eval: scoring each filled slot in an information-extraction template into five response buckets — Correct (COR), Partial (PAR), Incorrect (INC), Missing (MIS), Spurious (SPU) — and computing Precision = (COR + 0.5·PAR) / (COR + PAR + INC + SPU), Recall = (COR + 0.5·PAR) / (COR + PAR + INC + MIS), and F-measure. This error taxonomy is the historical origin of "strict / partial / type" matching used in NER and IE.
- required data: MUC-5 template-filling corpora (joint-venture announcements, microelectronics articles) with gold slot fills.

**(Segura-Bedmar et al., 2013)**

### SemEval-2013 Task 9: Extraction of Drug-Drug Interactions from Biomedical Texts (DDIExtraction 2013)

- [https://aclanthology.org/S13-2056/](https://aclanthology.org/S13-2056/)
- method of eval: formalised the four matching schemas still used today for NER evaluation — **Strict** (exact span + exact type), **Exact** (exact span, ignore type), **Partial** (overlapping span + type), **Type** (overlapping span + exact type) — each reported with Precision / Recall / F1. Task 9.2 (DDI extraction) adds relation scoring with exact-argument + relation-type match.
- required data: DDIExtraction 2013 corpus (DrugBank + MEDLINE) annotated with four drug entity types (`drug`, `brand`, `group`, `drug_n`) and four DDI relation types (`mechanism`, `effect`, `advise`, `int`).

**(Nakayama, 2018)**

### seqeval: A Python framework for sequence labeling evaluation

- [https://github.com/chakki-works/seqeval](https://github.com/chakki-works/seqeval)
- method of eval: Python re-implementation of the CoNLL-eval entity-level scorer. Default `mode="strict"` requires exact span + exact type match (entity-level P/R/F1 on BIO/BIOES sequences); optional relaxed span-only mode; also exposes token-level classification_report. Used as the default NER metric in HuggingFace Transformers, Flair, AllenNLP.
- required data: paired gold and predicted BIO / BIOES tag sequences with a fixed entity-type tagset.

**(Batista, 2018)**

### Named-Entity Evaluation Metrics Based on Entity-Level

- [https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/](https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/)
- method of eval: reference walk-through and open-source implementation of the MUC/SemEval-2013 matching schemas — Strict / Exact / Partial / Type — with per-schema Correct / Incorrect / Partial / Missing / Spurious counting and micro-averaged P/R/F1; later packaged as the `nervaluate` PyPI library that is widely used when CoNLL-style exact match is too harsh.
- required data: gold and predicted entity spans with type labels; no external resource, no tagset alignment required.

**(Amigo et al., 2025)**

### Evaluating Sequence Labeling on the basis of Information Theory

- [https://aclanthology.org/2025.acl-long.1351/](https://aclanthology.org/2025.acl-long.1351/) (DOI: 10.18653/v1/2025.acl-long.1351)
- method of eval: **SL-ICM (Sequence Labelling Information Contrast Model)** — an information-theoretic metric that measures how much correct (vs. noisy) information each predicted token contributes, weighted by sequence length, class base rate and token specificity. The paper proves that strict-span, token-level and partial-match F1 each violate at least one of a set of formal desiderata, while SL-ICM satisfies all of them simultaneously.
- required data: gold token/span labels on any sequence-labelling benchmark; class base rates (for the information-content term). Works on any NER / chunking / slot-filling corpus.

## Relation and end-to-end IE

**(Bekoulis et al., 2018)**

### Joint entity recognition and relation extraction as a multi-head selection problem

- [https://arxiv.org/abs/1804.07847](https://arxiv.org/abs/1804.07847)
- method of eval: codified the three standard end-to-end RE evaluation settings that all later papers cite — **Strict**: entity type AND boundaries AND relation type must all be correct; **Boundaries**: entity boundaries + relation type only (entity types ignored); **Relaxed**: one-token (head) match + relation type. Reports micro- and macro-averaged P/R/F1 under each.
- required data: end-to-end RE corpora with gold entities (type + span) and directed typed relations: ACE04, ACE05, CoNLL04, ADE, DREC.

**(Taillé et al., 2021)**

### Let's Stop Incorrect Comparisons in End-to-end Relation Extraction!

- [https://arxiv.org/abs/2009.10684](https://arxiv.org/abs/2009.10684)
- method of eval: meta-analysis showing that comparing the "Boundaries" setting of one paper with the "Strict" setting of another inflates ACE05 relation F1 by ~5 points. Proposes a minimum reporting checklist: declare (i) which Bekoulis setting is used, (ii) whether the NER span match is exact or head-only, (iii) whether entity types enter the relation score. Performs controlled ablations of pre-training (BERT) and span-level NER under each setting.
- required data: ACE05 / CoNLL04 with full entity + relation gold; NER predictions at both exact-span and head-only granularity.

**(Zhong & Chen, 2021)**

### A Frustratingly Easy Approach for Entity and Relation Extraction (PURE)

- [https://arxiv.org/abs/2010.12812](https://arxiv.org/abs/2010.12812)
- method of eval: canonical **Rel / Rel+** reporting that is now the de facto ACE/SciERC standard — **Rel**: a relation counts as correct if both entity spans are correct AND the relation type is correct (entity types ignored); **Rel+**: Rel PLUS both entity types correct. NER entity F1 is reported alongside using strict span+type match.
- required data: ACE04, ACE05, SciERC with gold entity spans + types and typed relations.

**(Duan et al., 2025)**

### SciNLP: A Domain-Specific Benchmark for Full-Text Scientific Entity and Relation Extraction in NLP

- [https://arxiv.org/abs/2509.07801](https://arxiv.org/abs/2509.07801)
- method of eval: adopts PURE-style Rel / Rel+ plus NER P/R/F1 on a new domain benchmark; adds cross-corpus transfer (train SciNLP → test SciERC and vice-versa) and downstream knowledge-graph statistics (average node degree, entity/relation coverage) as extrinsic quality checks of the extractor.
- required data: SciNLP corpus (60 full-text NLP papers, 6,409 entities, 1,648 relations) plus SciERC / similar scientific IE datasets for transfer evaluation.

# LLM as judge

## NER / RE / Extractive QA (closest to our task)

**(Laskar et al., 2025)**

### Improving Automatic Evaluation of Large Language Models (LLMs) in Biomedical Relation Extraction via LLMs-as-the-Judge

- [https://aclanthology.org/2025.acl-long.1238/](https://aclanthology.org/2025.acl-long.1238/) (DOI: 10.18653/v1/2025.acl-long.1238)
- method of eval: benchmarks 8 LLM judges scoring 5 candidate LLMs on 3 biomedical RE datasets. The judge is prompted with the source sentence, the gold relation, and the candidate's predicted relation, and outputs a binary match / no-match label; judge accuracy is meta-evaluated against human annotation. Two core findings: (i) naive LLM-as-judge is below 50% accuracy on free-form RE outputs; (ii) enforcing a **structured output schema** on the candidate lifts judge accuracy by ~15 points on average; (iii) **domain adaptation** of the judge via in-context examples from another biomedical RE dataset adds further gains.
- required data: source sentences, gold relation triples, candidate LLM free-text responses, plus 36 k human-annotated + LLM-annotated judgment labels (released). A fixed relation taxonomy is assumed, but the candidate surface form is not fixed.

**(Plum, Bernardy & Ranasinghe, 2026)**

### Do LLMs Judge Distantly Supervised Named Entity Labels Well? Constructing the judgeWEL Dataset

- [https://arxiv.org/abs/2601.00411](https://arxiv.org/abs/2601.00411)
- method of eval: uses an LLM as a *verifier* of distantly-supervised NER labels produced from Wikipedia internal links + Wikidata type inference. Several LLMs are prompted with a sentence and its candidate (entity, type) label and asked to judge correctness; the method keeps only sentences where multiple judges agree. Agreement between judges, and between judges and a human spot-check, is the validity signal. Produces a Luxembourgish NER corpus ~5× larger than the prior gold set.
- required data: weakly-labelled NER sentences (Wikipedia with linked mentions + Wikidata type mapping), a human-verified spot-check set for meta-evaluation, and a fixed NER type inventory.

**(Ho, Huang, Boudin & Aizawa, 2025)**

### LLM-as-a-Judge: Reassessing the Performance of LLMs in Extractive QA

- [https://arxiv.org/abs/2504.11972](https://arxiv.org/abs/2504.11972)
- method of eval: replaces Exact-Match / token-F1 with a single-answer LLM judge. The judge is given (question, context passage, gold answer, candidate answer) and outputs correct / incorrect. Meta-evaluates four QA datasets and multiple LLM families; correlation with human judgement rises from 0.22 (EM) and 0.40 (token-F1) to 0.85 (LLM judge). Tests for self-preference bias (using the same model as QA model and judge) and reports none.
- required data: extractive QA datasets with gold answer spans + a human-labelled subset for the correlation meta-evaluation.

## Summarization and reference-free NLG judging

**(Liu et al., 2023)**

### G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment

- [https://arxiv.org/abs/2303.16634](https://arxiv.org/abs/2303.16634)
- method of eval: LLM-as-judge framework that (i) auto-expands a short criterion (e.g. *coherence*, *faithfulness*, *consistency*, *relevance*) into chain-of-thought evaluation steps generated by GPT-4, then (ii) uses a form-filling prompt to emit a numeric score; the final score is optionally computed as the token-probability-weighted expectation over the score tokens to reduce ties. Achieves Spearman ρ ≈ 0.514 with human judgement on SummEval, beating BERTScore / GPTScore by a large margin.
- required data: source document + candidate summary; human ratings only for meta-evaluation of the judge.

**(Fu, Ng, Jiang & Liu, 2023)**

### GPTScore: Evaluate as You Desire

- [https://arxiv.org/abs/2302.04166](https://arxiv.org/abs/2302.04166)
- method of eval: reference-free likelihood scorer — scores a candidate by the conditional log-probability that a generative LLM assigns to it given an instruction + source (e.g. "Generate a factually consistent summary for the following text: {src}"). The instruction text *is* the evaluation criterion, so the same backbone can score 22 quality aspects across 37 datasets without retraining. Explored 19 backbones from 80 M to 175 B.
- required data: source text + candidate; human ratings only for correlation studies.

**(Yuan, Neubig & Liu, 2021)**

### BARTScore: Evaluating Generated Text as Text Generation

- [https://arxiv.org/abs/2106.11520](https://arxiv.org/abs/2106.11520)
- method of eval: seq2seq likelihood metric — scores a candidate by the conditional probability a BART / pre-trained seq2seq model assigns to one of (src → hyp), (ref → hyp), (hyp → ref), or (hyp ↔ ref). Different directions expose different qualities: src→hyp measures faithfulness, ref→hyp measures precision, hyp→ref measures recall. Unsupervised once BART is loaded; outperforms previous top metrics in 16 of 22 test settings.
- required data: source text and/or reference summary; optional human ratings for meta-evaluation.

## Hallucination and faithfulness

**(Huang, 2026)**

### Atomic-SNLI: Fine-Grained Natural Language Inference through Atomic Fact Decomposition

- [https://arxiv.org/abs/2601.06528](https://arxiv.org/abs/2601.06528)
- method of eval: decomposes a candidate hypothesis into atomic facts and runs three-way NLI (entailment / contradiction / neutral) per atom against the source; aggregates via the rule "hypothesis is entailed iff every atomic fact is entailed". Empirically shows current NLI models are worse at atomic than sentence-level inference, so the paper releases Atomic-SNLI training data and shows fine-tuning on it recovers atomic accuracy while preserving sentence accuracy.
- required data: source premise + candidate hypothesis; trained NLI model (e.g. DeBERTa-v3). Atomic-SNLI dataset is provided for fine-tuning.

**(Galimzianova, Boriskin & Arshinov, 2025)**

### From RAG to Reality: Coarse-Grained Hallucination Detection via NLI Fine-Tuning

- [https://aclanthology.org/2025.sdp-1.34/](https://aclanthology.org/2025.sdp-1.34/) (DOI: 10.18653/v1/2025.sdp-1.34)
- method of eval: frames hallucination detection as three-way NLI (entailment / contradiction / unverifiable) between a model claim and the retrieved reference. Fine-tunes DeBERTa-V3-large (pre-trained on five NLI corpora) on SciHal Subtask 1; this fine-tuned encoder beats LLM-prompt judges and an embedding-plus-softmax pipeline on weighted F1, illustrating the pattern that for reference-grounded hallucination detection a small fine-tuned NLI model beats zero-shot LLM prompting.
- required data: (claim, reference) pairs with 3-class entailment labels (SciHal); a pre-trained NLI encoder.

**(Hu et al., 2024)**

### RefChecker: Reference-based Fine-grained Hallucination Checker and Benchmark for Large Language Models

- [https://arxiv.org/abs/2405.14486](https://arxiv.org/abs/2405.14486)
- method of eval: two-stage pipeline. (1) **Extractor** LLM converts a candidate response into claim-triplets ⟨subject, predicate, object⟩. (2) **Checker** LLM (or NLI model) labels each triplet as entailed / neutral / contradicted against a reference. Supports three task settings — Zero Context, Noisy Context (RAG), Accurate Context. Shows claim-triplet granularity beats response / sentence / sub-sentence granularity by 4–9 points, and RefChecker overall beats prior hallucination checkers by 6.8–26.1 points with human-aligned judgments.
- required data: candidate LLM responses, a trusted reference text (document / retrieval passage / gold answer), and human-annotated triplet labels for benchmarking (11 k triplets released across 7 LLMs).

## Pairwise and general-purpose judges, and judge benchmarks

**(Zheng et al., 2023)**

### Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

- [https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)
- method of eval: introduces three judging protocols — **single-answer grading** (1–10), **pairwise comparison** (A / B / tie), and **reference-guided grading** — and systematically measures three judge biases: position bias, verbosity bias, self-enhancement bias. Mitigations include answer swapping (run A,B and B,A then average), chain-of-thought, and reference-guided prompts. Demonstrates that GPT-4 reaches ~80 % agreement with human preferences, equal to human–human agreement.
- required data: MT-Bench 80-prompt multi-turn test set + 30 k Chatbot Arena human preference battles; optional reference answers for the reference-guided variant.

**(Zhu, Wang & Wang, 2023)**

### JudgeLM: Fine-tuned Large Language Models are Scalable Judges

- [https://arxiv.org/abs/2310.17631](https://arxiv.org/abs/2310.17631)
- method of eval: distils GPT-4 pairwise judgments into a fine-tuned 7 / 13 / 33 B judge that outputs a two-number score plus a rationale; adds **swap augmentation** (against position bias), **reference support** and **reference drop** (against knowledge / format bias). Achieves >90 % agreement with the GPT-4 teacher, exceeding human–human agreement, on both the PandaLM benchmark and a new JudgeLM benchmark.
- required data: ~100 k seed tasks with pairs of candidate LLM answers and GPT-4 pairwise labels for training; pairwise test set for evaluation.

**(Wang et al., 2024)**

### PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization

- [https://arxiv.org/abs/2306.05087](https://arxiv.org/abs/2306.05087)
- method of eval: fine-tuned 7 B pairwise judge that outputs win / lose / tie plus an evaluation rationale. Explicitly targets subjective axes — conciseness, clarity, instruction adherence, comprehensiveness, formality — not just objective correctness. Reports F1 on a human-labelled test set reaching 93.75 % of GPT-3.5 and 88.28 % of GPT-4 while running locally (no API leakage).
- required data: instruction–response pairs, human-labelled pairwise preference test set, and an instruction-tuned candidate to judge.

**(Chen et al., 2025)**

### JudgeLRM: Large Reasoning Models as a Judge

- [https://arxiv.org/abs/2504.00050](https://arxiv.org/abs/2504.00050)
- method of eval: trains judge LLMs with **reinforcement learning** against judge-wise, outcome-driven rewards (not SFT on preference labels), producing chain-of-thought rationales before a pairwise verdict. Shows a negative correlation between SFT gains and the proportion of reasoning-demanding samples, motivating the RL approach. At 3–4 B it beats GPT-4 and at 7–14 B it beats DeepSeek-R1 by ~2 % F1, with especially strong gains on reasoning-heavy judging.
- required data: pairwise preference data with verifiable correctness signals (ground truth or verifier) that can define the RL outcome reward.

**(Wang et al., 2024)**

### Self-Taught Evaluators

- [https://arxiv.org/abs/2408.02666](https://arxiv.org/abs/2408.02666)
- method of eval: iterative self-improvement pipeline — starting from unlabeled instructions and a seed LLM, the method (i) samples two contrasting responses (one designed to be inferior), (ii) asks the current judge to produce a chain-of-thought verdict, (iii) retrains on the self-labelled preference data; repeat. No human preference labels. Lifts Llama3-70B-Instruct from 75.4 → 88.3 (88.7 with majority vote) on RewardBench, matching top supervised reward models.
- required data: pool of unlabeled instructions + a capable seed LLM; no human preference labels.

**(Tan et al., 2024)**

### JudgeBench: A Benchmark for Evaluating LLM-based Judges

- [https://arxiv.org/abs/2410.12784](https://arxiv.org/abs/2410.12784)
- method of eval: meta-benchmark for LLM judges where the gold preference is **objective correctness** rather than crowdsourced preference. Pipeline converts existing hard benchmarks (knowledge, reasoning, math, coding) into challenging response pairs with verifiable labels. Evaluates prompted judges, fine-tuned judges, multi-agent judges, and reward models on pairwise accuracy. Shows that on hard tasks, even GPT-4o-class judges barely beat random guessing — exposing that preference-aligned benchmarks (e.g. MT-Bench) over-estimate judge quality.
- required data: existing objective benchmarks (MMLU-Pro, LiveBench, HumanEval, etc.) from which verifiable response pairs can be derived.

## Evaluation without a gold standard

**(Seitl et al., 2024)**

### Assessing the quality of information extraction

- [https://arxiv.org/abs/2404.04068](https://arxiv.org/abs/2404.04068)
- method of eval: **MINEA (Multiple Infused Needle Extraction Accuracy)** — injects synthetic "needles" (artificial entities with name / description / keywords that match a target schema) into each document so that 10–30 % of the enriched text consists of needles, then runs the LLM extractor on the enriched document and measures how many needles were successfully extracted; the needles create a synthetic ground truth that enables absolute quality measurement without labelled data. MINEA can combine several decision rules (type match, name match, property match) into one score and is reported total and per schema type. Complemented by similarity, relevance, and an incompleteness score (proportion of entities with missing property values).
- required data: unstructured / semi-structured documents + an extraction schema (entity types + properties); a generator for plausible synthetic needles; no gold corpus required.

**(Xu, Lu, Schoenebeck & Kong, 2024)**

### Benchmarking LLMs' Judgments with No Gold Standard (GEM / GRE-bench)

- [https://arxiv.org/abs/2411.07127](https://arxiv.org/abs/2411.07127)
- method of eval: **GEM (Generative Estimator for Mutual Information)** — estimates the mutual information between a candidate judgement and one or more *peer* reference judgements using a generative LLM, so the "reference" does not have to be gold. Provably more robust than GPT-4-Examiner against rephrasing / elongation that strategically inflates scores. **GRE-bench** applies GEM to ICLR peer-review generation using a new year's papers each round, which side-steps data contamination.
- required data: candidate output + one or more peer reference outputs (no gold) + a generative LLM for conditional likelihood estimation; human ratings only for meta-evaluation.

**(Mahbub et al., 2026)**

### A Multi-Stage Validation Framework for Trustworthy Large-scale Clinical Information Extraction using Large Language Models

- [https://arxiv.org/abs/2604.06028](https://arxiv.org/abs/2604.06028)
- method of eval: six-stage pipeline for IE under weak supervision — (1) **prompt calibration** on a small dev set; (2) **rule-based plausibility filtering** (e.g. regex, term lists); (3) **semantic grounding** check that the extracted entity is actually mentioned in the source; (4) **targeted confirmatory evaluation** by an independent higher-capacity judge LLM; (5) **selective expert review** of high-uncertainty cases; (6) **external predictive-validity** analysis (does the extracted signal predict a downstream outcome?). Applied to substance-use disorder extraction from 919 k clinical notes; reports Gwet's AC1 = 0.80 between judge LLM and expert, relaxed F1 = 0.80 for the primary LLM against the judge-as-reference, and AUC = 0.80 for downstream prediction.
- required data: unstructured source documents + an extraction schema; a small expert-reviewed sample (not a full gold corpus); a downstream outcome variable for external-validity; an independent judge LLM of higher capacity than the primary extractor.

## Judge bias and meta-evaluation

**(Ye et al., 2024)**

### Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge (CALM)

- [https://arxiv.org/abs/2410.02736](https://arxiv.org/abs/2410.02736)
- method of eval: **CALM** — automated bias-quantification framework covering 12 bias types (position, verbosity, self-preference, authority, bandwagon, sentiment, distraction, refinement-aware, fallacy oversight, sycophancy, diversity, chain-of-thought). For each type CALM applies a principle-guided perturbation (swap A/B, pad candidate with filler, prepend authoritative tone, flip sentiment, etc.) to the same item and measures how much the judge verdict changes. Produces per-judge bias scorecards. Finds that even advanced judges retain significant biases on specific tasks.
- required data: a seed set of (prompt, candidate pair) items + automatic perturbation templates; no new human labels required.

**(Guerdan et al., 2025)**

### Validating LLM-as-a-Judge Systems under Rating Indeterminacy

- [https://arxiv.org/abs/2503.05965](https://arxiv.org/abs/2503.05965)
- method of eval: shows that when multiple ratings are legitimately "reasonable" for the same item (rating indeterminacy), forced-choice validation of a judge against a single aggregated human label selects judges that are up to 31 % worse than optimal. Proposes eliciting **multi-label "response sets"** from both humans and the judge and computing set-valued agreement. Provides a theoretical map between different human–judge agreement measures (Cohen's κ, Krippendorff's α, set-valued agreement) under different elicitation and aggregation schemes. 11 rating tasks × 9 commercial LLMs.
- required data: validation corpus with **multiple** human ratings per item (ideally set-valued) + the judge's full response distribution (e.g. top-k or CoT verdicts).

**(Han, Titericz Junior, Balough & Zhou, 2025)**

### Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement

- [https://arxiv.org/abs/2510.09738](https://arxiv.org/abs/2510.09738)
- method of eval: two-step judge benchmark. (1) **Correlation filter** — Pearson / Spearman between judge scores and human scores, discards judges with weak alignment. (2) **Human-likeness test** — compute Cohen's κ between judge and human and convert to a z-score against the human–human κ distribution, partitioning surviving judges into *human-like* ( |z| < 1, matches natural inter-human variance) and *super-consistent* ( z > 1, over-agrees — may indicate oversimplification). Evaluates 54 LLMs (1 B – 405 B) on RAG / Agentic response-correctness tasks; finds 27 reach Tier 1 and that training strategy matters more than size.
- required data: RAG / Agentic correctness evaluation set with gold answers + **multiple** human ratings per item (to estimate the human–human κ distribution).

# Synthesis for this study

This section applies the review above to the task in `docs/mpsa_draft_outline.md`: extracting `{entity, type, attribute, value}` quadruplets with evidence spans from 1,200+ NCSL state AI bills, using two methods (fixed-plan three-stage method vs skill-driven agentic NER), and evaluating them against bill-level NCSL theme / keyword labels under an explicit no-entity-gold constraint.

## Requirements the judge must satisfy

Derived from outline lines 195-212 and 247-260.

- Inputs available: bill full text, extracted quadruplets with evidence spans, bill-level NCSL theme / keyword labels.
- No entity-level gold standard (outline line 200).
- Three distinct judging tasks:
  1. **Coverage** — does the *set* of quadruplets semantically cover each coarse human label (outline lines 202, 255)?
  2. **Grounding** — is each quadruplet, especially novel ones, actually supported by the bill text (outline line 256)?
  3. **Cross-method comparison** — fixed-plan three-stage method vs skill-driven agentic NER (outline lines 208, 249-259).
- Direct string match is unreliable because quadruplets are finer-grained than labels (outline line 201).

## Fitness of each reviewed method

### NER / RE / extractive QA judges

- **(Laskar et al., 2025)**: assumes a gold relation triple to compare against. Full protocol inapplicable; transferable finding only — structured candidate schema + in-corpus in-context examples lift judge accuracy ~15 points.
- **(Plum, Bernardy & Ranasinghe, 2026)**: per-span verifier that needs a KG-derived candidate inventory. We have no such inventory. Inapplicable.
- **(Ho, Huang, Boudin & Aizawa, 2025)**: needs a gold answer span. Inapplicable as a protocol, but is the strongest evidence in the review for using an LLM judge over EM / F1 at all (ρ = 0.85 vs 0.22 / 0.40).

### Summarization / reference-free NLG scorers

- **(Liu et al., 2023) G-Eval; (Fu et al., 2023) GPTScore; (Yuan et al., 2021) BARTScore**: score NLG quality on dimensions or likelihoods. None operationalise "set of fine-grained facts → coarse-label coverage." Applicable only as inspiration for form-filling + token-probability-weighted scoring style.

### Hallucination / faithfulness (grounding direction)

- **(Huang, 2026) Atomic-SNLI**: hypothesis → atomic facts → per-atom NLI, aggregated. Conceptually applicable *in reverse* (treat human label as hypothesis, quadruplet set as source premise), but no paper in the review implements this direction.
- **(Galimzianova, Boriskin & Arshinov, 2025)**: fine-tuned DeBERTa-v3 NLI beats LLM-prompt judges for reference-grounded hallucination. Directly usable as the per-quadruplet grounding check if we accept training a small NLI head.
- **(Hu et al., 2024) RefChecker**: claim-triplet granularity proven to beat response / sentence by 4-9 points. We already emit structured quadruplets, so the extractor stage is skipped and the checker stage is reused. Best off-the-shelf fit for grounding.

### Pairwise / general-purpose judges

- **(Zheng et al., 2023) MT-Bench**: three protocols (single-answer, pairwise, reference-guided) with three bias mitigations (swap, CoT, reference-guided). Correct framework for cross-method comparison; not a content judge for our coverage task.
- **(Zhu, Wang & Wang, 2023) JudgeLM; (Wang et al., 2024) PandaLM; (Chen et al., 2025) JudgeLRM; (Wang et al., 2024) Self-Taught Evaluators**: all require training a custom judge from preference data we do not have. Not cost-justified for a single paper.
- **(Tan et al., 2024) JudgeBench**: a warning — preference-aligned judge benchmarks overstate judge quality on hard tasks. Take the warning: MT-Bench-style agreement alone is not sufficient validity evidence.

### No-gold evaluation

- **(Seitl et al., 2024) MINEA**: inject synthetic needles, measure recall. Useful as a complementary sanity check; inapplicable as primary judge because injection distorts legal text and only measures recall of the injected entities.
- **(Xu, Lu, Schoenebeck & Kong, 2024) GEM / GRE-bench**: mutual-information estimator with a peer reference instead of gold. Usable for the cross-method comparison as a relative ranking; does not give absolute coverage.
- **(Mahbub et al., 2026)**: six-stage IE validation under weak supervision — prompt calibration → rule filtering → semantic grounding → confirmatory judge LLM → expert review → external predictive validity. Closest one-to-one map to our setting and the only paper in the review that assumes "no full gold corpus, small expert sample, higher-capacity judge LLM." Best overall framework.

### Judge bias and meta-evaluation

- **(Ye et al., 2024) CALM**: 12-bias perturbation scorecard. Use as final bias audit on the judge.
- **(Guerdan et al., 2025)**: forced-choice validation against a single aggregated human label selects judges up to 31% worse than optimal. Relevant whenever the label-to-quadruplet mapping is genuinely indeterminate.
- **(Han et al., 2025) Judge's Verdict**: correlation filter + human-likeness z-test. Use as the final judge-selection gate against the expert-coded sample.

## Why no single paper suffices; what is missing in Mahbub

No single reviewed method simultaneously covers coverage, grounding, and cross-method comparison under our constraints (no per-item gold, granularity mismatch, evidence spans available). Each method covers a subset.

**(Mahbub et al., 2026)** is the closest single-paper fit — it is the only framework in the review that assumes exactly our setting. Three gaps remain:

- **Gap 1 — asymmetric set-to-label coverage.** Mahbub's judge evaluates one extracted entity at a time against the source. It does not evaluate whether a *set* of fine-grained quadruplets jointly covers a single coarse bill-level label. Our granularity mismatch (quadruplets vs theme / keyword labels) requires a coverage-direction judge that no paper in the review implements directly.
- **Gap 2 — cross-method comparison protocol.** Mahbub evaluates one extractor. Comparing two methods on the same bill with position-bias and verbosity-bias control is not part of its pipeline.
- **Gap 3 — judge-family independence.** Mahbub calls for a "higher-capacity" judge without enforcing family separation from the extractor. Since our extractor is Claude Sonnet 4.5, self-preference bias (Zheng et al., 2023) makes family separation a hard requirement that must be written into the protocol, not left implicit.

## Recommended modification

Adopt Mahbub's six-stage skeleton. Graft four components from other reviewed papers to close the three gaps. Resulting pipeline:

### Stage 1 — Prompt freezing (**Mahbub et al., 2026**'s stage 1; setup, not an evaluation) (Pre-Processing)

- **What happens.** Stage 1 writes the Stage 3 and Stage 4 judge prompt strings, plus their JSON response schemas, to `output/evals/v1/prompts/` and freezes them for every later stage. No quadruplet is evaluated; no judge call is issued at scale.
- **Comparison.** None. This is a pre-flight setup step.
- **Procedure.** Draw 30 NCSL bills (seed-fixed), render two fully-worked example triples — each triple is `(bill excerpt, candidate quadruplet, correct verdict)` drawn from those bills — and paste the examples into the Stage 3 and Stage 4 system prompts before writing them to disk.
- **Parameters.** `stages.stage1.dev_sample_bills = 30`, `stages.stage1.seed = 20260417`.
- **Cited frameworks.**
  - **(Mahbub et al., 2026)** use prompt calibration as stage 1 of their six-stage IE validation pipeline under weak supervision. The role is to lock the prompt once on a small dev sample so later stages are reproducible and cache-replayable. We adopt the same role.
  - **(Laskar et al., 2025)** compared LLM-judge prompts on quadruplet-grade tasks in their own benchmark: prompts that both (i) demanded a structured JSON response matching a fixed schema and (ii) included in-context examples drawn from the task's own corpus outperformed prompts lacking either ingredient. We adopt both ingredients because their within-paper comparison came out cleanly in favour of the combination: the structured-JSON demand is already satisfied by the extractor's output format (rows keyed by `refined_id`), and Stage 1 adds the in-corpus in-context examples. The accuracy delta Laskar et al. measured is specific to their corpus; it is not imported as a claim about this pipeline.

### Stage 2 — Rule-based fabrication filter (**Mahbub et al., 2026**'s stage 2, narrowed; filter, not an evaluation) (Pre-processing)

- **What happens.** Stage 2 removes mechanically broken quadruplets from the extractor's output so Stage 3 does not spend tokens on obvious failures.
- **Comparison per quadruplet.** The quadruplet's evidence-span `text` field is compared against the bill's full text on disk; the four required fields (`entity`, `type`, `attribute`, `value`) are checked for non-emptiness.
- **Pass rule.** A quadruplet passes iff (a) every required field is non-empty, (b) at least one evidence span has a non-empty `[start, end)` offset range, and (c) every non-empty span's `text` appears somewhere in the bill text. Offset correctness is *not* required — the orchestrated extractor reports whole-bill offsets while the skill-driven extractor reports chunk-relative offsets, so offsets are used only as starting hints, not as the matching criterion.
- **Output.** Per-quadruplet pass / fail plus the pass set used by Stages 3 and 4, in `output/evals/v1/results/stage2_plausibility.json`.
- **Cited framework.** **(Mahbub et al., 2026)**'s stage 2 is rule filtering; their rule set additionally required the entity and value surface forms to appear literally inside the evidence span. Those two sub-rules are dropped here because they are literal-grounding approximations that Stage 3's NLI judge does better, and they over-filter extractors that paraphrase values by design (documented in the `src/eval/stages/stage2_plausibility.py` module header).

### Stage 3 — Per-quadruplet grounding verdict (**Hu et al., 2024**'s RefChecker checker; item-level evaluation) (test 1, LLM-as-judge test)

- **What happens.** Stage 3 evaluates every quadruplet that Stage 2 passed.
- **Comparison per quadruplet.** The judge LLM reads (a) the four quadruplet fields (`entity`, `type`, `attribute`, `value`) laid out in the prompt as the "claim under test", and (b) the bill text excerpt around that quadruplet's evidence spans, and returns one of three verdicts.
- **Verdict.** `entailed` (bill text supports the claim), `contradicted` (bill text denies it), or `neutral` (bill text is silent about it).
- **Output.** Per-method verdict rates in `output/evals/v1/results/stage3_grounding.json`; per-quadruplet verdict in the Stage 3 per-item cache.
- **Cited frameworks.**
  - **(Hu et al., 2024)** proposed RefChecker: an LLM-based checker that (i) decomposes a model claim into atomic `(subject, predicate, object)` triplets and (ii) assigns each triplet a three-way NLI verdict against reference text. Their full pipeline includes a decomposer step that is unnecessary here because the extractor already emits atomic quadruplets, so we adopt only the three-way verdict schema and their checker-prompt pattern.
  - **(Zheng et al., 2023)** showed in their MT-Bench experiments that LLM judges systematically prefer outputs produced by their own model family when evaluating open-ended text. The judge here is Gemini 2.5 Pro and the extractor is Claude Sonnet 4.5 — different vendor, different training lineage — so the self-preference channel is closed by construction.
- **Upgrade path.** **(Galimzianova et al., 2025)** fine-tuned DeBERTa-v3 NLI heads on legal-IE data and reported lower cost per verdict than an LLM checker on their benchmark. Stage 3's verdict is drop-in replaceable by a fine-tuned NLI head without changing the cache schema.

### Stage 4 — Set-to-label coverage (**Huang, 2026**'s Atomic-SNLI with direction inverted; set-level evaluation) (test 2, LLM-as-judge test)

- **What happens.** Stage 4 evaluates how well each method's surviving quadruplets account for the NCSL topic tags that human curators assigned to the bill.
- **Comparison per `(bill, NCSL topic label)` pair.** The judge reads (a) the list of quadruplets from one method that passed both Stage 2 and Stage 3 for this bill and (b) one NCSL topic label (e.g. "health", "civil_rights"), and answers: does any subset of those quadruplets jointly cover the label?
- **Verdict.** `covered`, `partially_covered`, or `not_covered`. On `covered` or `partially_covered`, the judge also returns the IDs of the quadruplets in the supporting subset; those IDs feed Stages 5, 6, and 8.
- **Output.** Per-method coverage rate, per-label breakdown, per-state breakdown in `output/evals/v1/results/stage4_coverage.json`.
- **Cited frameworks.**
  - **(Huang, 2026)** proposes Atomic-SNLI: check whether each extracted atomic claim is entailed by a coarser reference label — direction label → atomic claim. That direction is backward here: NCSL labels are deliberately coarse, and a fine-grained quadruplet is rarely entailed by a topic tag on its own. The direction is inverted to claim set → label: a label is covered iff some subset of the quadruplets collectively entails it.
  - **(Zheng et al., 2023)** self-preference argument — same mitigation as Stage 3; judge family stays separate from extractor family.

### Stage 5 — Novel-claim bookkeeping (bookkeeping over Stages 3 and 4; no judge call) (Records artifacts, neither test nor processing)

- **What happens.** Stage 5 flags every quadruplet that Stage 3 grounded as `entailed` but that no Stage 4 verdict cited as a supporting quadruplet for any NCSL label.
- **Comparison per quadruplet.** The quadruplet's ID is checked against the union of the `supporting_ids` lists across every Stage 4 verdict for that bill. In the union → `cited`; outside the union → `novel`.
- **Output.** Per-method novel count, per-entity-type distribution of novel quadruplets, per-state count, and a 50-row stratified audit sample, in `output/evals/v1/results/stage5_novelty.json`.
- **Parameters.** No YAML knobs. Audit sample size is hardcoded at 50 rows, stratified across entity types, in `src/eval/stages/stage5_novelty.py` (`_AUDIT_SAMPLE_SIZE = 50`).
- **Why no outside citation.** **(Mahbub et al., 2026)**'s pipeline has no equivalent step because their ground truth was dense. Ours is sparse — NCSL tags one or two topics per bill while the extractor may surface a dozen distinct facts — so the "beyond the reference" slice is a first-class quantity rather than a residual and gets its own accounting.

### Stage 6 — Cross-method pairwise comparison (**Zheng et al., 2023**'s MT-Bench, three protocols; pairwise evaluation) (test 3, LLM-as-judge bias test part 1)

- **What happens.** Stage 6 compares the two NER methods head-to-head on the same bill.
- **Three comparison protocols, all from (Zheng et al., 2023).**
  - **Single-answer grading.** Not a judge call. Stage 4's coverage rate and Stage 5's novel count for each method are read and compared directly.
  - **Pairwise with answer swap.** The judge is called twice per bill — once with quadruplets presented in `(method A, method B)` order, once in `(B, A)` order — and the two verdicts are averaged. Averaging attenuates *position bias*, which Zheng et al. identified in their MT-Bench study as a systematic judge preference for whichever option is shown first.
  - **Reference-guided.** The union of the bill's NCSL topic labels is added to the judge's prompt as a reference answer, and the same pairwise call is issued with the reference present.
- **Verdict per judge call.** `A wins`, `B wins`, or `tie`.
- **Verbosity-bias correction.** Zheng et al. separately identified *verbosity bias* — LLM judges tend to prefer longer outputs. Stage 6 logs the quadruplet count per method per bill and reports both (a) the raw win rate and (b) a count-normalised rate in which a method's win is discounted when its quadruplet set was longer than its opponent's.
- **Output.** Raw win rate, swap-averaged win rate, count-normalised score, in `output/evals/v1/results/stage6_pairwise.json`.
- **Parameters.** `stages.stage6.pairwise_sample_bills = null` (the full intersection of bills across methods; an integer would cap deterministically), `stages.stage6.swap = true`, `stages.stage6.seed = 20260417`. Judge call volume = `len(intersection) × (2 if swap else 1)`. The current run uses all 1,693 common bills, i.e. 3,386 pairwise judge calls.



### Stage 8 — Judge bias audit (**Ye et al., 2024**'s CALM, four-perturbation subset; judge evaluation, not extractor evaluation)(test 4, LLM-as-judge bias test part 2)

- **What happens.** Stage 8 audits the judge, not the extractors. It measures how often the judge changes its Stage 4 verdict when the Stage 4 prompt is cosmetically perturbed.
- **Comparison per `(bill, NCSL label, bias type)`.** The judge's original Stage 4 verdict for that `(bill, label)` — read from the Stage 4 cache — is compared against the judge's verdict on a perturbed version of the same prompt. The two verdicts are tallied as `flipped` or `unchanged`.
- **Four perturbations (subset of (Ye et al., 2024)'s CALM catalogue).**
  - `position` — reverses the order of the supporting quadruplets.
  - `verbosity` — pads the system prompt with irrelevant boilerplate.
  - `self_preference` — prefixes the system prompt with a self-favouring claim ("you are a state-of-the-art model…").
  - `authority` — prefixes the user prompt with an authority appeal ("a senior expert insists the correct answer is X").
- **Output.** Flip rate per bias type per method, plus an overall flip rate across the four biases, in `output/evals/v1/results/stage8_bias.json`.
- **Parameters.** `stages.stage8.sample_items = 100`, `stages.stage8.calm_subset = [position, verbosity, self_preference, authority]`, `stages.stage8.seed = 20260417`. Sampling pools Stage 4 baseline rows across both methods before drawing, so `sample_items` is a pooled total, not per method. Judge call volume = `sample_items × len(calm_subset)` = 400 calls with the current settings.
- **What the flip rate means.** Low flip rate = judge decisions are internally consistent under cosmetic manipulation. This is a consistency diagnostic, not an accuracy claim — a judge that never flips can still be systematically wrong.
- **Cited framework.** **(Ye et al., 2024)** built CALM as a benchmark for LLM-judge bias across a broader catalogue than the four above. We run the four most relevant to this setting.

### Stage 9 — Extrinsic-validity handoff (**Mahbub et al., 2026**'s stage 6; aggregation, not evaluation) (Post-processing)

- **What happens.** Stage 9 is aggregation, not a comparison. It joins per-bill-per-method rows — quadruplet count, Stage 4 coverage rate, Stage 5 novel count, per-type counts — into `output/evals/v1/results/stage9_extrinsic.csv`, with a companion `.md` documenting the columns.
- **The actual validity tests live downstream.** The analyses that test whether richer extraction predicts external outcomes (entity distribution, entity relations, temporal drift, focus-vs-outcome) run in the bill-analysis work described in `docs/mpsa_draft_outline.md` lines 264-272 and consume Stage 9's CSV as input.
- **Cited framework.** **(Mahbub et al., 2026)** operationalise external predictive validity as stage 6 of their pipeline: they demonstrate, in their setting, that extractor outputs predict a downstream variable of interest. Our downstream variables are the four named above; this stage produces the join key those analyses need, but the validity claim itself lives in those analyses, not here.

