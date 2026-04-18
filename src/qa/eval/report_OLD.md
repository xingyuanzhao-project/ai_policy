# QA Eval Report

- Ground truth: `C:/Users/meier/OneDrive/Documents/ai_policy/src/qa/eval/ground_truth.json`
- Corpus: `data/ncsl/us_ai_legislation_ncsl_text.jsonl`
- Retrieval backend: `vector`, chunks indexed: 123787
- Answer model: `google/gemini-2.5-flash`
- N questions: **20**
- Overall pass rate: **45.00%**

## Pass rate by difficulty

| Difficulty | N | Pass rate | Mean keyword | Mean citation F1 | Mean count score | Mean latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| easy | 5 | 40% | 0.40 | 0.23 | 0.00 | 1.85 |
| hard | 6 | 33% | 0.64 | 0.60 | 0.62 | 2.55 |
| medium | 5 | 60% | 0.60 | 0.23 | 0.00 | 1.78 |
| very_hard | 4 | 50% | 0.38 | 0.19 | 1.00 | 2.94 |

## Per-question detail

| ID | Difficulty | Pattern | Pass | KW | Cit P | Cit R | Cit F1 | Count | #Cit | Notes |
|---|---|---|:---:|---:|---:|---:|---:|---:|---:|---|
| E01 | easy | single_bill_fact | no | 0.00 | 0.00 | 0.00 | 0.00 | - | 5 | low citation recall (0.0) |
| E02 | easy | single_bill_fact | no | 0.00 | 0.00 | 0.00 | 0.00 | - | 5 | low citation recall (0.0) |
| E03 | easy | single_bill_fact | YES | 1.00 | 0.50 | 1.00 | 0.67 | - | 5 |  |
| E04 | easy | single_bill_fact | no | 0.00 | 0.00 | 0.00 | 0.00 | - | 5 | low citation recall (0.0) |
| E05 | easy | single_bill_fact | YES | 1.00 | 0.33 | 1.00 | 0.50 | - | 5 |  |
| M01 | medium | single_bill_detail | YES | 1.00 | 0.00 | 0.00 | 0.00 | - | 5 | low citation recall (0.0) |
| M02 | medium | single_bill_detail | YES | 1.00 | 0.00 | 0.00 | 0.00 | - | 5 | low citation recall (0.0) |
| M03 | medium | single_bill_detail | no | 0.00 | 0.50 | 1.00 | 0.67 | - | 5 |  |
| M04 | medium | single_bill_detail | no | 0.00 | 0.00 | 0.00 | 0.00 | - | 0 | citations 0 < min 1; low citation recall (0.0) |
| M05 | medium | single_bill_detail | YES | 1.00 | 0.33 | 1.00 | 0.50 | - | 5 |  |
| H01 | hard | filter_list | no | 0.50 | 1.00 | 0.50 | 0.67 | 1.00 | 5 |  |
| H02 | hard | filter_list | no | 0.33 | 1.00 | 0.29 | 0.44 | 0.83 | 5 |  |
| H03 | hard | count | no | 1.00 | - | - | - | 0.03 | 5 |  |
| H04 | hard | aggregate | no | 0.00 | - | - | - | - | 5 |  |
| H05 | hard | filter_list | YES | 1.00 | 0.75 | 0.19 | 0.30 | - | 5 | low citation recall (0.1875) |
| H06 | hard | filter_list | YES | 1.00 | 1.00 | 1.00 | 1.00 | - | 5 |  |
| VH01 | very_hard | list_and_compare | YES | 0.50 | 1.00 | 0.16 | 0.27 | 1.00 | 5 | low citation recall (0.15789473684210525) |
| VH02 | very_hard | compare | no | 0.00 | 0.00 | 0.00 | 0.00 | - | 5 | low citation recall (0.0) |
| VH03 | very_hard | compare | no | 0.00 | 0.00 | 0.00 | 0.00 | - | 5 | low citation recall (0.0) |
| VH04 | very_hard | filter_list_compare | YES | 1.00 | 0.50 | 0.50 | 0.50 | - | 5 |  |

## Per-question answers (first 1000 chars each)

### E01 -- easy / single_bill_fact

**Q:** What is the title of Delaware House Bill 16 from 2025?

**Citations (5):** `2025__DE H    233, 2025__DE H    105, 2025__DE SCR  18, 2025__DE SCR  18, 2025__DE H    233`

**Applied filters:** `{"state": "Delaware", "year": 2025}`

**A:**

> The answer is not supported by the retrieved bill text.

### E02 -- easy / single_bill_fact

**Q:** What is the status of Kentucky Senate Bill 4 from 2025?

**Citations (5):** `2025__KY H    498, 2025__KY H    672, 2025__KY H    672, 2025__KY SCR  142, 2025__KY H    498`

**Applied filters:** `{"state": "Kentucky", "year": 2025}`

**A:**

> The provided text does not contain information about the status of Kentucky Senate Bill 4 from 2025. The retrieved bills are House Bill 498, House Bill 672, and Senate Concurrent Resolution 142 [1, 2, 3, 4, 5].

### E03 -- easy / single_bill_fact

**Q:** In what year was Texas House Bill 149 (Regulation of the Use of Artificial Intelligence) enacted?

**Citations (5):** `2025__TX S    1964, 2025__TX S    1964, 2025__TX S    1964, 2025__TX S    1964, 2025__TX H    149`

**Applied filters:** `{"state": "Texas", "status_bucket": "Enacted"}`

**A:**

> The provided text does not explicitly state the enactment year of Texas House Bill 149 (Regulation of the Use of Artificial Intelligence). It does indicate "bill_id=2025__TX H 149" and "status=Enacted - Act No. 1174" [5], but the year 2025 is likely part of the bill ID format and not the enactment year itself.

### E04 -- easy / single_bill_fact

**Q:** What topics does California Senate Bill 53 (Artificial Intelligence Models: Large Developers) address?

**Citations (5):** `2025__CA S    430, 2023__CA SCR   17, 2025__CA A    1337, 2025__CA A    887, 2024__CA S    108`

**Applied filters:** `{"state": "California"}`

**A:**

> The answer is not supported by the retrieved bill text.

### E05 -- easy / single_bill_fact

**Q:** Which California bill passed in 2025 is known as the California AI Transparency Act?

**Citations (5):** `2025__CA A    979, 2025__CA A    979, 2025__CA S    53, 2025__CA A    853, 2025__CA A    979`

**Applied filters:** `{"state": "California", "status_bucket": "Enacted", "year": 2025}`

**A:**

> The California bill passed in 2025 known as the California AI Transparency Act is bill number 2025__CA A 853 [4].

### M01 -- medium / single_bill_detail

**Q:** Under Arizona Senate Bill 1295, what class of felony is criminal impersonation using computer-generated voice or video?

**Citations (5):** `2024__AZ S   1078, 2024__AZ S   1078, 2024__AZ H   2727, 2024__AZ S   1599, 2024__AZ S   1599`

**Applied filters:** `{"state": "Arizona", "topics": ["Criminal Use"]}`

**A:**

> Under Arizona Senate Bill 1295, criminal impersonation using a computer-generated voice recording, image, or video with intent to defraud or harass is a Class 5 felony [2].

### M02 -- medium / single_bill_detail

**Q:** What does Mississippi Senate Bill 2426 establish, and what is its charge?

**Citations (5):** `2023__MS S   3000, 2025__MS S    3002, 2025__MS H    1489, 2025__MS S    2834, 2024__MS S   2062`

**Applied filters:** `{"state": "Mississippi"}`

**A:**

> The provided text does not contain information about Mississippi Senate Bill 2426. Therefore, its establishment and charge cannot be determined from the given context. Instead, the retrieved text mentions Mississippi Senate Bill 2062, which establishes the Artificial Intelligence in Education Task Force Act [5].

### M03 -- medium / single_bill_detail

**Q:** What change does Delaware House Bill 16 make to the AI Commission's membership?

**Citations (5):** `2025__DE H    16, 2025__DE H    16, 2024__DE H    333, 2024__DE H    333, 2024__DE H    333`

**Applied filters:** `{"state": "Delaware"}`

**A:**

> The retrieved bill context does not describe changes made by Delaware House Bill 16 to the AI Commission's membership, but rather defines its current composition [1, 2]. Therefore, the answer is not supported by the retrieved bill text.

### M04 -- medium / single_bill_detail

**Q:** Under New York Senate Bill 822, what must state agencies do regarding automated employment decision making tools?

**Citations (0):** `(none)`

**Applied filters:** `{"state": "New York", "topics": ["Effect on Labor/Employment Government Use"]}`

**A:**

> I could not find relevant bill text to answer that question.

### M05 -- medium / single_bill_detail

**Q:** What does Montana Senate Bill 25 regulate?

**Citations (5):** `2025__MT S    413, 2025__MT H    82, 2025__MT H    82, 2025__MT S    25, 2025__MT H    82`

**Applied filters:** `{"state": "Montana"}`

**A:**

> Montana Senate Bill 25 regulates the use of deepfakes in election communications [4]. It outlines penalties for violations, which can include civil penalties, misdemeanor prosecution for a second offense, and felony prosecution for a third offense [4].

### H01 -- hard / filter_list

**Q:** List the AI-related bills that Delaware enacted in 2025.

**Citations (5):** `2025__DE H    16, 2025__DE H    16, 2025__DE H    16, 2025__DE H    16, 2025__DE H    16`

**Applied filters:** `{"state": "Delaware", "status_bucket": "Enacted", "year": 2025}`

**A:**

> The AI-related bill that Delaware enacted in 2025 is titled "Artificial Intelligence Commission" (bill_id=2025__DE H 16) [1, 2, 3, 4, 5].

### H02 -- hard / filter_list

**Q:** Which US states enacted AI-related bills that specifically address elections in 2025?

**Citations (5):** `2025__KY S    4, 2025__MT S    25, 2025__KY S    4, 2025__KY S    4, 2025__MT S    25`

**Applied filters:** `{"status_bucket": "Enacted", "topics": ["Elections"], "year": 2025}`

**A:**

> Kentucky and Montana enacted AI-related bills that specifically address elections in 2025.
> 
> In Kentucky, a bill addresses the "critical impact artificial intelligence can have on the security of data and information in the Commonwealth" and states the importance "to protect candidates and election officers from fraudulent misrepresentations of themselves and their issues" [1].
> 
> In Montana, a bill concerns the "use of deepfakes in election communications" [2]. It states that "revolutionary innovations in generative artificial intelligence systems... pose a threat to free and fair elections" and that "AI-generated content may be used to create deepfakes that falsely depict a candidate's speech or action in order to spread misinformation and disinformation" [2]. The bill also prohibits the distribution of deepfakes of candidates or political parties in election communications unless a disclosure is included [5].

### H03 -- hard / count

**Q:** Approximately how many AI-related bills were enacted across US states and territories in 2025?

**Citations (5):** `2025__KY S    4, 2025__MS S    2426, 2025__TN H    1409, 2025__MD H    956, 2025__KY S    4`

**Applied filters:** `{"status_bucket": "Enacted", "year": 2025}`

**A:**

> The retrieved bill text indicates that at least four AI-related bills were enacted in US states with enactment dates in 2025: Kentucky S 4 [1, 5], Mississippi S 2426 [2], Maryland H 956 [4]. Another bill, Tennessee H 1409, mentions appropriations for implementing other bills, but its direct relation to AI regulation is not evident from the provided excerpt [3]. Therefore, approximately **4** AI-related bills were enacted across US states and territories in 2025, based on the provided context.

### H04 -- hard / aggregate

**Q:** Which US state had the most AI-related bills enacted in 2025?

**Citations (5):** `2025__KY S    4, 2025__MS S    2426, 2025__TN H    1409, 2025__MD H    956, 2025__KY S    4`

**Applied filters:** `{"status_bucket": "Enacted", "year": 2025}`

**A:**

> The answer is not supported by the retrieved bill text.

### H05 -- hard / filter_list

**Q:** Name several 2025 enacted US state bills that specifically target deepfakes or synthetic media.

**Citations (5):** `2025__MT S    25, 2025__SD S    164, 2025__SD S    164, 2025__TX S    441, 2025__MT S    413`

**Applied filters:** `{"status_bucket": "Enacted", "topics": ["Criminal Use"], "year": 2025}`

**A:**

> Several 2025 enacted US state bills specifically target deepfakes or synthetic media:
> *   Montana S 25, "Use Of Deepfakes In Election Communications" [1].
> *   South Dakota S 164, "Deepfake to Influence an Election" [2, 3].
> *   Texas S 441, "Artificial Intimate Visual Material" [4].
> *   Montana S 413, "Crime of Disclosing Explicit Synthetic Media" [5].

### H06 -- hard / filter_list

**Q:** What did California enact in 2025 concerning large AI model developers?

**Citations (5):** `2025__CA S    53, 2025__CA S    53, 2025__CA S    53, 2025__CA S    53, 2025__CA S    53`

**Applied filters:** `{"state": "California", "status_bucket": "Enacted", "year": 2025}`

**A:**

> In 2025, California enacted Act No. 2025-138 regarding Artificial Intelligence Models: Large Developers. This act mandates that a "large frontier developer" write, implement, comply with, and publish a "frontier AI framework" on its internet website. This framework must describe how the developer addresses:
> 
> *   Incorporating national and international standards, and industry-consensus best practices [2, 4].
> *   Defining and assessing thresholds to identify and evaluate if a frontier model has capabilities that could pose a catastrophic risk [4].
> *   Applying mitigations to address potential catastrophic risks based on assessments [4].
> *   Reviewing assessments and the adequacy of mitigations before deploying a frontier model or using it extensively internally [4].
> *   Using third parties to assess potential catastrophic risks and the effectiveness of mitigations [4].
> *   Revisiting and updating the frontier AI framework, including criteria for updates and determining when substantial ...

### VH01 -- very_hard / list_and_compare

**Q:** What are the bills that are about AI that are passed or enacted in 2025? What do they have in common?

**Citations (5):** `2025__KY S    4, 2025__CA S    53, 2025__KY S    4, 2025__CA S    53, 2025__MS S    2426`

**Applied filters:** `{"status_bucket": "Enacted", "year": 2025}`

**A:**

> There are three enacted AI-related bills in 2025:
> 1.  **Kentucky S 4**: "Protection of Information" [1, 3]
> 2.  **California S 53**: "Artificial Intelligence Models: Large Developers" [2, 4]
> 3.  **Mississippi S 2426**: "Artificial Intelligence Regulation Task Force" [5]
> 
> The common themes among these bills are:
> *   **Risk and Benefit Assessment**: All bills acknowledge the need to consider both the potential benefits and risks of AI. For example, Kentucky S 4 mentions internationally recognized risk management frameworks [1], California S 53 highlights the potential for both innovation and "catastrophic risks" [2], and Mississippi S 2426 includes "Assessment of risk and benefits" as a review focus [5].
> *   **Transparency and Disclosure**: Promoting transparency in AI systems is a common element. Kentucky S 4 requires disclosure of how AI is used in decision-making and the extent of human involvement [3]. California S 53 mentions "key instruments to increase transparency" [2], and Missis...

### VH02 -- very_hard / compare

**Q:** Compare how Texas House Bill 149 and Colorado Senate Bill 4 (both 2025) approach AI regulation.

**Citations (5):** `2025__TX S    1964, 2025__TX H    1709, 2025__TX H    1709, 2025__TX H    1709, 2025__TX H    1709`

**Applied filters:** `{"state": "Texas", "year": 2025}`

**A:**

> The provided texts do not contain any information about Colorado Senate Bill 4 (2025). Therefore, a comparison of how Texas House Bill 149 and Colorado Senate Bill 4 (2025) approach AI regulation cannot be made from the given context. The answer is not supported by the retrieved bill text.

### VH03 -- very_hard / compare

**Q:** What common approach do Rhode Island HB 5872, Montana SB 25, and North Dakota HB 1167 share?

**Citations (5):** `2023__RI H   6286, 2024__RI H   6286, 2023__RI H   6423, 2025__RI S    627, 2025__RI H    5172`

**Applied filters:** `{"state": "Rhode Island"}`

**A:**

> The retrieved bill text does not contain information on Montana SB 25 or North Dakota HB 1167. Therefore, I cannot determine a common approach shared by Rhode Island HB 5872, Montana SB 25, and North Dakota HB 1167 from the provided context. [1, 2, 3, 4, 5]

### VH04 -- very_hard / filter_list_compare

**Q:** List 2025 enacted bills that address the use of AI in employment, and what do they require of employers or state agencies?

**Citations (5):** `2025__IL S    2394, 2025__IL S    2394, 2025__NY S    822, 2025__NY S    822, 2025__NY S    822`

**Applied filters:** `{"status_bucket": "Enacted", "topics": ["Effect on Labor/Employment"], "year": 2025}`

**A:**

> The following 2025 enacted bills address the use of AI in employment:
> 
> *   **Illinois S 2394**:
>     *   Prohibits employers from using AI that discriminates against protected classes or uses zip codes as a proxy for protected classes in recruitment, hiring, promotion, renewal of employment, selection for training or apprenticeship, discharge, discipline, tenure, or terms/privileges/conditions of employment [1].
>     *   Requires employers to provide notice to an employee if the employer is using AI for the purposes described above [1, 2]. The Department shall adopt rules for the implementation and enforcement of this subdivision, including requirements for notice [2].
> 
> *   **New York S 822**:
>     *   Prohibits the use of AI systems from resulting in the discharge, displacement, loss of position, reduction in non-overtime work hours, wages, or employment benefits, or impairment of existing collective bargaining agreements for employees of the state or any agency or public authority [4].
> ...