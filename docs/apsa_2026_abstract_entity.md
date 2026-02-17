## Abstract

Unpacking AI Governance: LLM-Based Entity Extraction of Regulatory Targets from State Legislation

What do state AI bills actually regulate, and who or what do they place obligations on? Existing research often treats “AI bills” as a single unit and emphasizes coarse outcomes such as accounts, introduction, adoption or document-level labels. But bills are bundles of heterogeneous governance instruments, and collapsing them can mask real variation and yield weak or unstable relationships. We argue that a more granular approach is needed to understand the substantive content of AI governance. 

We propose a multi-turn conversational extraction pipeline with named entity recognition (NER) techniques that iteratively probes each bill for entities across two dimensions: regulatory targets and regulated entities. Unlike single-pass extraction, this approach refines, expands and updates entity lists and spans through follow-up prompts, enabling more thorough coverage of fragmented and varied legislative language. 

Applying this pipeline to 1,200+ state AI bills, we extract and categorize regulatory targets and entities, then analyze their distribution across states, time, and party composition.

This approach enables researchers to move beyond coarse indicators of AI governance to understanding the substantive content of AI governance, revealing which sectors dominate policy attention, whether states regulate private companies or government agencies differently, and how regulatory focus shifts over time.



# Offshoot Paper: Entity Extraction from AI Legislation

## Roadmap

---

The core problem (not a variable-mixing game)
Most state-AI-bill research commits a granularity error: it treats “AI bill” as a single, comparable unit and models outcomes like introduction/adoption. But bills are bundles of heterogeneous governance instruments. Two bills can both be “AI bills” while doing fundamentally different things (procurement rules vs private-sector liability vs biometrics bans vs sector-specific constraints). When you collapse them, you get:
Measurement error: the key independent variable (“AI bill”) is too coarse, so estimated relationships attenuate or flip.
Ecological fallacy across texts: adoption correlates might reflect which kinds of bills are introduced, not “AI regulation” broadly.
Missing mechanism: adoption studies rarely tell you what policy design moved (obligations, targets, enforcement, exemptions).
Your LLM extraction is the fix because it builds document-level policy-design variables that let you test mechanisms rather than existence.

What “granular document-level variables” should buy you
A strong framing is: coarse bill counts are not the outcome; policy design is. Adoption becomes one stage; the paper’s payoff is explaining variation in regulatory content.
Concretely, extraction should produce variables that are close to “policy design primitives,” e.g.:
Target (domain/technology), regulated entity (developer/deployer/agency), instrument (ban, mandate, disclosure, audit, procurement), enforcement (agency, penalties, private right of action), exemptions/safe harbors, definition scope, risk triggers.

### 1. General Idea and Research Question

**Core shift:** From "What is AI?" → "What does AI policy target?"

**Two extraction dimensions:**
1. **Regulatory Targets (Themes):** What domains/sectors/applications are being regulated?
2. **Regulated Entities:** Who/what is being regulated — companies, government agencies, technologies, infrastructure?

**RQ:** What regulatory targets and entities do state AI bills address, and how do these vary across states and over time?

**Motivation:**
- Definitions tell us *what AI is*; entity extraction tells us *what AI policy cares about*
- Identifies regulatory themes: healthcare, employment, law enforcement, facial recognition, etc.
- Identifies regulated entities: private companies, government agencies, infrastructure, technologies themselves
- Reveals policy priorities and potential gaps across states

---

### 2. Preferred Methods

**Primary Method:** Named Entity Recognition (NER) via LLM prompting

**Approach Options:**
| Approach | Description | References |
|----------|-------------|------------|
| Multi-agent zero-shot | Cooperative agents, no fine-tuning | #1, #2 |
| Pure prompt engineering | GPT-4/Gemini, zero-shot or few-shot | #3, #4, #5 |
| Two-stage decomposition | Entity locating → entity typing | #7 |

**Key methodological feature:** Multi-turn conversational prompting
- Iteratively update extraction prompt for each document
- Enables more thorough entity extraction than single-pass
- Each turn refines and expands the extracted entity list

**Recommended pipeline:**
1. Define dual taxonomy (regulatory targets + regulated entities)
2. Initial prompt: extract entities from bill text
3. Follow-up prompts: iteratively probe for missed entities, clarify ambiguous extractions
4. Aggregate and categorize extracted entities by both dimensions
5. Analyze distribution across states, time, party composition

---

### 3. Dataset

**Source:** Same as definition paper — 1,200+ AI bills from U.S. state legislatures (2025)

**Required fields:**
- `state`
- `bill_id`
- `bill_url`
- `text` (full legislation text)
- `year` / `date_introduced`
- `status` (introduced, passed, etc.)
- `party_composition` (if available)

---

### 4. Entity Taxonomy (Draft)

**Dimension 1: Regulatory Targets (Themes)**
| Category | Examples |
|----------|----------|
| **Sectors** | Healthcare, education, employment, law enforcement, finance, transportation |
| **Technologies** | Facial recognition, autonomous vehicles, chatbots, predictive analytics, generative AI |
| **Applications** | Hiring decisions, credit scoring, criminal sentencing, medical diagnosis, content moderation |

**Dimension 2: Regulated Entities (Who/What gets regulated)**
| Category | Examples |
|----------|----------|
| **Private sector** | Companies, developers, deployers, vendors |
| **Public sector** | Government agencies, public institutions, state departments |
| **Infrastructure** | Platforms, systems, networks, data centers |
| **Technology itself** | AI systems, algorithms, models, automated decision systems |
| **Individuals** | Consumers, employees, students, citizens |

---

### 5. References

See `references/references.md` — 10 papers on LLM/agent-based NER methods

**Most relevant:**
- #1, #2: Agent-based, multi-agent coordination
- #3, #4, #5: Pure prompting with commercial LLMs (GPT-4, Gemini)
- #7: Two-stage decomposition via prompting

---

### 6. Next Steps

- [ ] Finalize dual taxonomy (targets + entities) based on preliminary bill review
- [ ] Design multi-turn prompt template for iterative entity extraction
- [ ] Run pilot extraction on subset (~50 bills)
- [ ] Evaluate extraction quality on both dimensions
- [ ] Scale to full dataset
- [ ] Analyze entity distribution patterns
- [ ] Draft abstract

---

### 7. Potential Title Ideas (≤80 chars)

1. What AI Policy Targets: Entity Extraction from U.S. State Legislation
2. Mapping AI Regulation: NER-Based Analysis of State Legislative Priorities
3. Beyond Definitions: Extracting Regulatory Targets from State AI Bills
4. Who and What Gets Regulated: Entity Extraction from State AI Bills
