# QA App Severity Report

Ground-truth eval of 20 hand-curated questions against the live `QAService`
(vector backend, 123,787 chunks, `google/gemini-2.5-flash` answer model,
`google/gemini-2.5-flash` filter extractor). See `results.json` and `report.md`
for raw scores.

---

## 1. Headline numbers

| Difficulty | N | Pass rate | Mean keyword | Mean citation F1 | Mean count score |
|---|---:|---:|---:|---:|---:|
| easy | 5 | 40% | 0.40 | 0.23 | - |
| medium | 5 | 60% | 0.60 | 0.23 | - |
| hard | 6 | 33% | 0.64 | 0.60 | 0.62 |
| very_hard | 4 | 50% | 0.38 | 0.19 | 1.00 |
| **Overall** | **20** | **45%** | **0.52** | **0.31** | **0.54** |

**The 45% pass rate overstates quality.** Inspection of individual answers shows
multiple "passes" that are false positives (answered correctly by coincidence or
on keywords, while citing the wrong bills). Counting only answers that are
*actually* right about *actually-cited* evidence, the true pass rate is
**roughly 4-5 / 20 ≈ 20-25%**.

False positives identified by hand audit of `results.json`:

- **E03**: keyword "2025" matched, but the answer explicitly *rejects* 2025 as
  the enactment year ("the year 2025 is likely part of the bill ID format and
  not the enactment year itself"). Ground truth: TX HB 149 was enacted in 2025.
- **M01** (AZ SB 1295): answered "Class 5 felony" but cited only 2024 Arizona
  bills. AZ SB 1295 never appeared in retrieval. Correct answer by luck/model
  prior, not evidence.
- **M02** (MS SB 2426): passed on keyword "task force" because a *different*
  Mississippi bill (2024 MS S 2062, AI in Education Task Force) was retrieved.
  MS SB 2426 was never retrieved.
- **VH01** (headline "enacted 2025 + commonalities"): listed 3 bills of 192
  actual (recall 1.6%), yet counted "pass" because keyword threshold 0.5 was
  met. This is the very failure you flagged in chat.

---

## 2. How serious is the problem?

**Severe.** The app is doing three fundamentally different things well, badly,
and catastrophically depending on question shape:

### Works well (3 questions / 15%)

- **H06** (CA SB 53 large developers): perfect precision/recall/F1 when the
  question happens to match the semantic content of one specific bill *and*
  the filter extractor narrows to one state correctly.
- **VH04** (AI in employment 2025): correctly found NY SB 822 and IL SB 2394
  using `topics=["Effect on Labor/Employment"]`.
- **M05** (MT SB 25 regulates deepfakes in elections): correct.

### Works shakily (5 questions / 25%)

- **H01** (Delaware enacted 2025): answer said "the AI-related bill" (singular)
  and cited DE H 16 five times. Missed DE H 105 entirely. Count off by 50%.
- **H02** (election-topic 2025 enacted): found 2 of 6 ground-truth states
  (Kentucky, Montana). Missed Nevada, North Dakota, Rhode Island, South Dakota.
  Recall = 29%.
- **H05** (deepfake / synthetic media bills): found 3-4 of ≥12 ground-truth
  bills. Recall = 19%.
- **E05** (who passed "California AI Transparency Act"): correct (CA AB 853).
- **VH04**: see above.

### Fails catastrophically (12 questions / 60%)

- **E01 / E04 / M03 / E02 / M02 / M01**: single-bill fact lookups by ID
  ("Delaware HB 16", "Kentucky SB 4", "California SB 53"). Retrieval cannot
  find a specific bill by its number because the bill_id string appears in
  metadata, not in the bill's legal text. The semantic score ranks other
  bills in the same state higher.
  - Example, E01: asked for DE H 16's title. Retrieval returned DE H 233, DE
    H 105, DE SCR 18 (all Delaware, none DE H 16). Answer: "not supported by
    the retrieved bill text".
- **H03** (count enacted 2025): answered "approximately 4". Truth: 192.
  Count score = 0.026. Off by ~50x.
- **H04** (which state enacted the most in 2025): answer "not supported by the
  retrieved bill text". The app has no path to compute aggregates; it can only
  summarize the 5 retrieved chunks.
- **VH02** (compare TX HB 149 and CO SB 4): filter extractor locked
  `state="Texas"` only, so Colorado was never retrieved. Answer: cannot
  compare.
- **VH03** (common approach of RI HB 5872, MT SB 25, ND HB 1167): same failure
  mode, filter extractor locked `state="Rhode Island"`, MT and ND invisible.
- **VH01** (enacted 2025 + commonalities): listed 3 bills of 192. The
  "commonalities" section sounded plausible (transparency, ethics, risk) but
  was generalized from those 3 bills, not the actual population.

---

## 3. Root causes (ranked by how many eval failures they explain)

### RC1. Top-k=5 ceiling on list/aggregate questions (explains 7 failures)

H01, H02, H03, H04, H05, VH01, and half of VH04 all hit the same wall: the
retriever returns at most 5 chunks, chunks can repeat the same bill, and the
answer model can only reason about what is in those 5 chunks. "Enacted in 2025"
has 192 bills in the corpus; retrieval surfaces at most ~5.

Evidence from `results.json`:
- H01 (2 true bills in Delaware): returned DE H 16 five times, DE H 105 zero
  times.
- H03 (192 true): cited 4 distinct bills.
- VH01 (192 true): cited 3 distinct bills.

### RC2. Filter extractor over-constrains multi-entity questions (explains 2 failures)

VH02, VH03 both name 2-3 states, but `FilterExtractor` picked one:
- VH02 applied `state="Texas"` (dropped Colorado).
- VH03 applied `state="Rhode Island"` (dropped Montana, North Dakota).

Root cause: the tool schema treats `state` as a single string
(`"state": {"type": "string"}`), not a list. Same for `topic_hints` when the
question spans multiple topics.

### RC3. Filter extractor emits malformed topic strings (explains 1 failure)

M04 applied `topics=["Effect on Labor/Employment Government Use"]` — two
separate topic tokens concatenated into one string. This matches zero chunks
(`topics_list` is compared set-wise), so the retriever returned 0 chunks and
the answer model replied "I could not find relevant bill text".

### RC4. No bill-ID-aware retrieval (explains 4 failures)

E01, E02, E04, M03 all mention a specific bill by number. The bill_id
("DE H 16", "CA S 53") is stored as metadata but is not embedded into the
chunk text. Semantic search over the legal text cannot rank by numeric ID.
The user-facing symptom: asking "What does DE HB 16 do?" returns other
Delaware bills.

### RC5. No aggregate / count / "most" capability (explains 2 failures)

H03 ("how many 2025 enacted?") and H04 ("which state enacted the most?") are
not retrieval questions, they are database aggregation questions. The app has
no mechanism for these — it can only summarize what is in 5 retrieved chunks.
A single chunk cannot tell the model "there are 192 bills matching filter F".

### RC6. Ranking diversity problem (explains part of H01, H05)

When a filter narrows to a small set (e.g., Delaware has 4 bills), top-5
chunks can still all come from one bill. H01 is the cleanest example: 5
citations, all DE H 16. With no per-bill cap, the retriever dumped five
slightly-different chunks of the same bill.

---

## 4. Recommendation: where to spend effort

The evidence supports a tiered plan. I'm re-using the existing agent
infrastructure (`src/agent/loop.py`, `src/agent/tools.py`) so "build new
fixed flows" is actually *more* new code than going agentic here.

### Priority 1: quick-and-critical fixes (≈70-100 new lines, half a day)

These are bug-class, not architecture-class. They dominate the failure list.

1. **Fix `FilterExtractor` schema so `state` and `topics` accept lists.** Bug,
   not a feature trade-off. Resolves VH02, VH03, and probably also the
   "topics concatenated into one string" symptom in M04.
   - ~30 lines in `src/qa/filter_extractor.py` (change tool JSON schema,
     normalize list outputs). Then pass `list[str]` through to
     `QAService._normalize_filters` — which already accepts list of strings
     for topics.

2. **Add a bill-ID fast path.** Before the filter extractor runs, regex the
   question for bill-id patterns (`\b[A-Z]{2}\s*[HS]\w*\s*\d+\b`, plus common
   prose forms "House Bill 16", "SB 53"). If a pattern matches and the
   resolved bill_id exists in the index, bypass semantic retrieval and fetch
   that bill's chunks directly. Resolves E01, E02, E04, M03, and also
   silently fixes many medium-difficulty single-bill questions.
   - ~40 lines in `src/qa/service.py` plus small helper in `src/qa/retriever.py`
     that can return all chunks for a given bill_id.

3. **Per-bill citation diversity (Tier 0).** Cap chunks-per-bill in top-k
   (e.g., max 1 chunk per bill_id for list-style questions) and bump
   `retrieval_top_k` conditional on filters being present (e.g., 5 -> 15
   when a filter reduces the candidate set). Improves H01, H02, H05.
   - ~30 lines in `src/qa/retriever.py`.

**Estimated effect on this eval**: probably lifts 6-8 questions from fail to
pass, taking pass rate from 45% headline / ~25% real to roughly 60-70% real.

### Priority 2: structural fix for list/count/aggregate (≈150 new lines, ~1 day)

The RC1 (top-k ceiling) and RC5 (no aggregate) failures are *not* fixable by
tweaking parameters. They need the model to be able to *ask the corpus* about
its shape, not just read 5 chunks.

Two implementation options — both leverage existing agent infrastructure, so
new code is smaller than it looks:

**Option A: Agentic RAG using `run_tool_loop`** (recommended)
- Tools to register: `search(query, filters)`, `count(filters)`,
  `list_bills(filters, max=50)`, `get_bill(bill_id)`.
- System prompt tells the model: for count/list/aggregate, call `count` /
  `list_bills`; for content, call `search` / `get_bill`.
- Route `QAService.answer_question` through `run_tool_loop` when the filter
  extractor reports an "aggregate intent" (`count`, `most`, `how many`, `list
  all`, `compare`), otherwise keep today's fast path.
- New code: one `qa_agent.py` (~120 lines) + small plumbing in `service.py`
  (~30 lines). Handlers reuse existing retriever / metadata access.

**Option B: Fixed map-reduce query decomposition** (not recommended)
- Detect compound / count / compare intent in the filter extractor, emit
  sub-queries and an aggregation step, then run each.
- New code: ~200-260 lines (query classifier, per-subquery runner, reducer,
  new answer prompt). More new code than Option A and reimplements what
  `run_tool_loop` already does.

**Estimated effect on this eval**: lifts H03, H04, VH01 from fail to
substantive pass. Pass rate should exceed 80%.

### Priority 3: optional polish

- Bump `retrieval_top_k` from 5 to 10-15 as a default.
- Add a trace UI element showing inferred filters *and* the raw count of
  chunks considered (currently filters are shown; count is not).
- Add the eval harness to CI at a reduced question count (5 fastest) as a
  regression gate.

---

## 5. Explicit recommendation

Do Priority 1 first (filter-extractor bugs + bill-ID fast path + diversity).
They are cheap, the failures they fix are not "the hard compound question"
failures but *easy question* failures (single-bill lookups are Easy-tier and
half of them fail today). Without these, no amount of agentic cleverness on
top will fix the easy cases.

Then do Priority 2 Option A (agentic RAG using `run_tool_loop`). The existing
agent infrastructure means this is ~150 new lines, not a rewrite. It is the
only option that fixes "how many" and "which state has the most" at all.

Skip Priority 2 Option B entirely. Fixed map-reduce flows are strictly more
new code than the agentic path given the agent infra already exists.

---

## 6. How to reproduce

```
.\.venv\Scripts\python.exe -m src.qa.eval.runner
```

Optional flags:
- `--max-questions N` — smoke test on first N.
- `--answer-model <id>` — switch answer model.
- `--results PATH --report PATH` — write outputs elsewhere.

Output files:
- `src/qa/eval/results.json` — per-question raw answers, citations, scores.
- `src/qa/eval/report.md` — auto-generated tabular summary.
- `src/qa/eval/severity_report.md` — this file (human-authored analysis).
- `src/qa/eval/ground_truth.json` — the 20-question ground-truth dataset.
