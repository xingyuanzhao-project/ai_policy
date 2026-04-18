# QA Eval Report

- Ground truth: `C:/Users/meier/OneDrive/Documents/ai_policy/src/qa/eval/ground_truth.json`
- Corpus: `data/ncsl/us_ai_legislation_ncsl_text.jsonl`
- Retrieval backend: `vector`, chunks indexed: 123787
- Answer model: `google/gemini-2.5-flash`
- N questions: **20**
- Overall pass rate: **75.00%**

## Pass rate by difficulty

| Difficulty | N | Pass rate | Mean keyword | Mean citation F1 | Mean count score | Mean latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| easy | 5 | 80% | 0.80 | 0.23 | 0.00 | 3.34 |
| hard | 6 | 83% | 1.00 | 0.65 | 0.64 | 3.76 |
| medium | 5 | 60% | 0.60 | 0.12 | 0.00 | 5.88 |
| very_hard | 4 | 75% | 0.75 | 0.21 | 1.00 | 13.41 |

## Per-question detail

| ID | Difficulty | Pattern | Pass | KW | Cit P | Cit R | Cit F1 | Count | #Cit | Notes |
|---|---|---|:---:|---:|---:|---:|---:|---:|---:|---|
| E01 | easy | single_bill_fact | YES | 1.00 | 0.25 | 1.00 | 0.40 | - | 8 |  |
| E02 | easy | single_bill_fact | YES | 1.00 | 0.17 | 1.00 | 0.29 | - | 10 |  |
| E03 | easy | single_bill_fact | YES | 1.00 | 0.17 | 1.00 | 0.29 | - | 10 |  |
| E04 | easy | single_bill_fact | YES | 1.00 | 0.10 | 1.00 | 0.18 | - | 10 |  |
| E05 | easy | single_bill_fact | no | 0.00 | 0.00 | 0.00 | 0.00 | - | 0 | citations 0 < min 1; low citation recall (0.0) |
| M01 | medium | single_bill_detail | YES | 1.00 | 0.00 | 0.00 | 0.00 | - | 10 | low citation recall (0.0) |
| M02 | medium | single_bill_detail | no | 0.00 | 0.00 | 0.00 | 0.00 | - | 6 | low citation recall (0.0) |
| M03 | medium | single_bill_detail | YES | 1.00 | 0.20 | 1.00 | 0.33 | - | 10 |  |
| M04 | medium | single_bill_detail | no | 0.00 | 0.00 | 0.00 | 0.00 | - | 0 | citations 0 < min 1; low citation recall (0.0) |
| M05 | medium | single_bill_detail | YES | 1.00 | 0.14 | 1.00 | 0.25 | - | 10 |  |
| H01 | hard | filter_list | YES | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 4 |  |
| H02 | hard | filter_list | YES | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 10 |  |
| H03 | hard | count | no | 1.00 | - | - | - | 0.26 | 10 |  |
| H04 | hard | aggregate | YES | 1.00 | - | - | - | - | 10 |  |
| H05 | hard | filter_list | YES | 1.00 | 0.67 | 0.25 | 0.36 | - | 10 |  |
| H06 | hard | filter_list | YES | 1.00 | 0.14 | 1.00 | 0.25 | - | 10 |  |
| VH01 | very_hard | list_and_compare | YES | 1.00 | 0.83 | 0.26 | 0.40 | 1.00 | 10 |  |
| VH02 | very_hard | compare | YES | 1.00 | 0.14 | 0.50 | 0.22 | - | 10 |  |
| VH03 | very_hard | compare | no | 0.00 | 0.00 | 0.00 | 0.00 | - | 10 | low citation recall (0.0) |
| VH04 | very_hard | filter_list_compare | YES | 1.00 | 0.14 | 0.50 | 0.22 | - | 10 |  |

## Per-question answers (first 1000 chars each)

### E01 -- easy / single_bill_fact

**Q:** What is the title of Delaware House Bill 16 from 2025?

**Citations (8):** `2025__DE H    105, 2025__DE H    233, 2025__DE H    233, 2025__DE H    16, 2025__DE SCR  18, 2025__DE H    105, 2025__DE SCR  18, 2025__DE H    16`

**Applied filters:** `{"routing_path": "agent", "state": "Delaware", "year": 2025}`

**A:**

> The title of Delaware House Bill 16 from 2025 is "Artificial Intelligence Commission".

### E02 -- easy / single_bill_fact

**Q:** What is the status of Kentucky Senate Bill 4 from 2025?

**Citations (10):** `2025__KY H    498, 2025__KY H    20, 2025__KY S    4, 2025__KY H    672, 2025__KY H    20, 2025__KY H    672, 2025__KY H    498, 2025__KY H    21, 2025__KY SCR  142, 2025__KY SCR  142`

**Applied filters:** `{"routing_path": "agent", "state": "Kentucky", "year": 2025}`

**A:**

> Kentucky Senate Bill 4 (2025__KY S 4) from 2025 has been Enacted as Act No. 66.

### E03 -- easy / single_bill_fact

**Q:** In what year was Texas House Bill 149 (Regulation of the Use of Artificial Intelligence) enacted?

**Citations (10):** `2025__TX S    1964, 2025__TX S    1964, 2025__TX H    149, 2025__TX H    149, 2023__TX H   2060, 2025__TX S    1188, 2025__TX H    2818, 2023__TX H   2060, 2025__TX H    2818, 2025__TX H    3512`

**Applied filters:** `{"routing_path": "agent", "state": "Texas", "status_bucket": "Enacted"}`

**A:**

> Texas House Bill 149 (Regulation of the Use of Artificial Intelligence) was enacted in 2025.

### E04 -- easy / single_bill_fact

**Q:** What topics does California Senate Bill 53 (Artificial Intelligence Models: Large Developers) address?

**Citations (10):** `2024__CA A   2013, 2024__CA S   1047, 2025__CA S    833, 2025__CA S    503, 2025__CA S    53, 2023__CA S    313, 2024__CA S    313, 2025__CA S    468, 2025__CA S    420, 2025__CA S    11`

**Applied filters:** `{"routing_path": "agent", "state": "California", "topics": ["Oversight/Governance", "Impact Assessment", "Private Sector Use"]}`

**A:**

> California Senate Bill 53 (2025__CA S 53) addresses several topics related to artificial intelligence models and large developers. These include:
> 
> *   **Cybersecurity**: The bill touches upon the security aspects of AI models.
> *   **Government Use**: It considers the use of AI by governmental entities.
> *   **Impact Assessment**: The legislation focuses on evaluating the potential effects and risks of AI.
> *   **Private Sector Use**: It specifically addresses the application of AI within the private sector.
> *   **Studies**: The bill also involves the undertaking of studies related to AI.

### E05 -- easy / single_bill_fact

**Q:** Which California bill passed in 2025 is known as the California AI Transparency Act?

**Citations (0):** `(none)`

**Applied filters:** `{"routing_path": "agent", "state": "California", "status_bucket": "Enacted", "year": 2025}`

**A:**

> There are no California bills enacted in 2025 known as the California AI Transparency Act.

### M01 -- medium / single_bill_detail

**Q:** Under Arizona Senate Bill 1295, what class of felony is criminal impersonation using computer-generated voice or video?

**Citations (10):** `2024__AZ S   1078, 2024__AZ S   1078, 2024__AZ H   2727, 2024__AZ S   1599, 2024__AZ S   1599, 2024__AZ H   2307, 2024__AZ H   2307, 2025__AZ H    2678, 2025__AZ H    2678, 2025__AZ H    2712`

**Applied filters:** `{"routing_path": "agent", "state": "Arizona", "topics": ["Criminal Use"]}`

**A:**

> Arizona Senate Bill 1295 was not found. However, Arizona Senate Bill 1078 (2024) addresses criminal impersonation using computer-generated voice or video. Under this bill, criminal impersonation using a computer-generated voice recording, image, or video with intent to defraud or harass is a Class 5 felony. Other forms of criminal impersonation are a Class 1 misdemeanor.

### M02 -- medium / single_bill_detail

**Q:** What does Mississippi Senate Bill 2426 establish, and what is its charge?

**Citations (6):** `2024__MS S   2062, 2024__MS S   2423, 2024__MS S   2423, 2024__MS S   2062, 2024__MS H   1267, 2024__MS H   1267`

**Applied filters:** `{"routing_path": "agent", "state": "Mississippi"}`

**A:**

> Mississippi Senate Bill 2426 was not found. There are no bills matching "Senate Bill 2426" in Mississippi for the year 2024.

### M03 -- medium / single_bill_detail

**Q:** What change does Delaware House Bill 16 make to the AI Commission's membership?

**Citations (10):** `2025__DE H    16, 2024__DE H    333, 2024__DE H    333, 2025__DE H    16, 2025__DE SCR  18, 2025__DE SCR  18, 2025__DE H    105, 2025__DE H    105, 2025__DE H    233, 2025__DE H    233`

**Applied filters:** `{"routing_path": "agent", "state": "Delaware"}`

**A:**

> Delaware House Bill 16 (2025__DE H 16) establishes the Delaware AI Commission with 23 voting members. The membership includes 12 state government officials, 4 members appointed by the President Pro Tempore of the Senate (including AI specialists and Senate members), 4 members appointed by the Speaker of the House (including AI specialists and House members), and 3 members appointed by the Governor (including an AI specialist, a lawyer, and a private sector AI expert). Additionally, there will be an annual non-voting student member appointed by the Chair. Appointed members serve up to 3-year terms and can be reappointed.

### M04 -- medium / single_bill_detail

**Q:** Under New York Senate Bill 822, what must state agencies do regarding automated employment decision making tools?

**Citations (0):** `(none)`

**Applied filters:** `{"routing_path": "agent", "state": "New York", "topics": ["Effect on Labor/Employment Government Use"]}`

**A:**

> I couldn't find any bills matching "New York Senate Bill 822" with the specified filters. Therefore, I cannot answer what state agencies must do regarding automated employment decision-making tools under that bill.

### M05 -- medium / single_bill_detail

**Q:** What does Montana Senate Bill 25 regulate?

**Citations (10):** `2025__MT HJR  4, 2025__MT HJR  4, 2025__MT H    82, 2025__MT H    514, 2025__MT H    82, 2025__MT S    212, 2025__MT S    25, 2025__MT S    25, 2025__MT S    413, 2025__MT H    10`

**Applied filters:** `{"routing_path": "agent", "state": "Montana"}`

**A:**

> Montana Senate Bill 25 (2025__MT S 25) regulates the use of deepfakes in election communications. Its purpose is to prevent misleading voters and harming candidates' reputations through AI-generated deepfakes. The bill applies to election and electioneering communications involving deepfakes of candidates or political parties, covering any person, corporation, committee, or political party involved in an official election capacity.
> 
> Key obligations include prohibiting the distribution or sponsorship of deepfakes of candidates or political parties within 60 days of an election's voting initiation. This prohibition does not apply if the communication includes a clear disclosure stating it has been "significantly edited by artificial intelligence and depicts speech or conduct that falsely appears to be authentic or truthful." The bill specifies format requirements for disclosures across various media. Exceptions are made for bona fide newscasts, news interviews, news documentaries, or on-...

### H01 -- hard / filter_list

**Q:** List the AI-related bills that Delaware enacted in 2025.

**Citations (4):** `2025__DE H    16, 2025__DE H    16, 2025__DE H    105, 2025__DE H    105`

**Applied filters:** `{"routing_path": "agent", "state": "Delaware", "status_bucket": "Enacted", "year": 2025}`

**A:**

> Delaware enacted 2 AI-related bills in 2025:
> 
> *   **DE H 16**: "Artificial Intelligence Commission" (topics: Education Use, Studies)
> *   **DE H 105**: "Employment Practices" (topics: none listed)

### H02 -- hard / filter_list

**Q:** Which US states enacted AI-related bills that specifically address elections in 2025?

**Citations (10):** `2025__KY S    4, 2025__MT S    25, 2025__KY S    4, 2025__MT S    25, 2025__ND H    1167, 2025__SD S    164, 2025__NV A    73, 2025__RI S    816, 2025__RI H    5872, 2025__SD S    164`

**Applied filters:** `{"routing_path": "agent", "status_bucket": "Enacted", "topics": ["Elections"], "year": 2025}`

**A:**

> Kentucky, Montana, North Dakota, South Dakota, Nevada, and Rhode Island have enacted AI-related bills that specifically address elections in 2025.
> 
> The bills are:
> *   **Kentucky:** KY S 4, "Protection of Information"
> *   **Montana:** MT S 25, "Use Of Deepfakes In Election Communications"
> *   **North Dakota:** ND H 1167, "Artificial Intelligence Disclosure Statements"
> *   **South Dakota:** SD S 164, "Deepfake to Influence an Election"
> *   **Nevada:** NV A 73, "Campaign Practices"
> *   **Rhode Island:** RI S 816, "Deceptive and Fraudulent Synthetic Media in Elections" and RI H 5872, "Regulate the Use of Synthetic Media in Elections"

### H03 -- hard / count

**Q:** Approximately how many AI-related bills were enacted across US states and territories in 2025?

**Citations (10):** `2025__KY S    4, 2025__MS S    2426, 2025__TN H    1409, 2025__MD H    956, 2025__KY S    4, 2025__TX H    149, 2025__CA S    53, 2025__TX H    149, 2025__UT S    180, 2025__NY S    822`

**Applied filters:** `{"routing_path": "agent", "status_bucket": "Enacted", "year": 2025}`

**A:**

> In 2025, 50 AI-related bills were enacted across US states and territories.

### H04 -- hard / aggregate

**Q:** Which US state had the most AI-related bills enacted in 2025?

**Citations (10):** `2025__KY S    4, 2025__MS S    2426, 2025__TN H    1409, 2025__MD H    956, 2025__KY S    4, 2025__TX H    149, 2025__CA S    53, 2025__TX H    149, 2025__UT S    180, 2025__NY S    822`

**Applied filters:** `{"routing_path": "agent", "status_bucket": "Enacted", "year": 2025}`

**A:**

> California had the most AI-related bills enacted in 2025, with 8 bills. Texas followed with 6 enacted bills.

### H05 -- hard / filter_list

**Q:** Name several 2025 enacted US state bills that specifically target deepfakes or synthetic media.

**Citations (10):** `2025__MT S    25, 2025__SD S    164, 2025__SD S    164, 2025__TX S    441, 2025__MT S    413, 2025__MT H    513, 2025__TX S    441, 2025__TN S    1346, 2025__TN S    1346, 2025__MT S    25`

**Applied filters:** `{"routing_path": "agent", "status_bucket": "Enacted", "topics": ["Criminal Use"], "year": 2025}`

**A:**

> Several 2025 enacted US state bills specifically target deepfakes or synthetic media:
> 
> *   **Montana:**
>     *   MT S 25: "Use Of Deepfakes In Election Communications"
>     *   MT S 413: "Crime of Disclosing Explicit Synthetic Media"
>     *   MT H 513: "Name, Voice, and Likeness Property Rights"
> *   **South Dakota:**
>     *   SD S 164: "Deepfake to Influence an Election"
> *   **Texas:**
>     *   TX S 441: "Artificial Intimate Visual Material"
> *   **Tennessee:**
>     *   TN S 1346: "Preventing Deepfake Images Act"
> *   **Washington:**
>     *   WA H 1205: "Knowing Distribution of a Forged Digital Likeness"
> *   **Colorado:**
>     *   CO S 288: "Intimate Digital Depictions Criminal Civil Actions"
> *   **Utah:**
>     *   UT S 271: "Unauthorized Artificial Intelligence Impersonation"

### H06 -- hard / filter_list

**Q:** What did California enact in 2025 concerning large AI model developers?

**Citations (10):** `2025__CA S    53, 2025__CA S    53, 2025__CA A    1170, 2025__CA S    524, 2025__CA A    1170, 2025__CA S    361, 2025__CA A    102, 2025__CA A    316, 2025__CA A    979, 2025__CA S    524`

**Applied filters:** `{"routing_path": "agent", "state": "California", "status_bucket": "Enacted", "year": 2025}`

**A:**

> In 2025, California enacted CA S 53, titled "Artificial Intelligence Models: Large Developers." This bill focuses on establishing transparency and accountability for large AI model developers to mitigate catastrophic risks and ensure public safety. It applies to "frontier models" and "frontier developers," defined by their potential to cause catastrophic risks, and explicitly avoids burdening smaller companies. The bill mandates standardized and objective reporting by frontier developers to the government and public regarding their AI models' capabilities and safety, and requires timely reporting of critical safety incidents.

### VH01 -- very_hard / list_and_compare

**Q:** What are the bills that are about AI that are passed or enacted in 2025? What do they have in common?

**Citations (10):** `2025__KY S    4, 2025__CA S    53, 2025__KY S    4, 2025__CA S    53, 2025__MS S    2426, 2025__MS S    2426, 2025__MD H    956, 2025__NY S    822, 2025__DE H    16, 2025__NY S    822`

**Applied filters:** `{"routing_path": "agent", "status_bucket": "Enacted", "year": 2025}`

**A:**

> There are 49 bills about AI that were enacted in 2025. These bills share several common themes:
> 
> *   **Defining AI:** Many bills include definitions of "Artificial Intelligence" or "AI," generally describing it as a machine-based system that infers from inputs to generate outputs that can influence environments.
> *   **Generative AI:** Several bills specifically define "Generative Artificial Intelligence" (GenAI) as AI capable of producing synthetic content like audio, images, text, and video.
> *   **Government Use and Oversight:** A recurring theme is the establishment of task forces, workgroups, or divisions to study, recommend, and oversee the use of AI within state government agencies, including developing policies and best practices.
> *   **Transparency and Disclosure:** Multiple bills emphasize the need for transparency regarding AI use, particularly when it interacts with the public or generates content, often requiring disclaimers or disclosures.
> *   **Focus on Risks and Harms:** ...

### VH02 -- very_hard / compare

**Q:** Compare how Texas House Bill 149 and Colorado Senate Bill 4 (both 2025) approach AI regulation.

**Citations (10):** `2025__TX H    1709, 2025__TX H    1709, 2025__TX S    1964, 2025__TX S    1822, 2025__TX H    1265, 2025__TX S    1964, 2025__TX H    149, 2025__TX S    1411, 2025__TX H    4018, 2025__TX S    1822`

**Applied filters:** `{"routing_path": "agent", "state": "Texas", "year": 2025}`

**A:**

> Texas House Bill 149 and Colorado Senate Bill 4 (both 2025) approach AI regulation with distinct focuses.
> 
> Texas House Bill 149 (2025__TX H 149) adopts a data-centric approach, primarily regulating the use of biometric identifiers in AI. It defines "biometric identifier" and sets conditions for the capture, storage, and commercial use of such data, especially when used for training AI systems. The bill requires consent for biometric data use and outlines penalties for violations related to the possession and destruction of biometric identifiers. It also clarifies responsibilities for data processors in assisting consumers with rights requests.
> 
> Colorado Senate Bill 4 (2025__CO S 4) takes a system-centric and impact-focused approach, aiming to increase transparency for algorithmic systems and prevent "algorithmic discrimination," particularly from "high-risk artificial intelligence systems." This bill targets developers of high-risk AI systems, imposing duties related to reasonable care...

### VH03 -- very_hard / compare

**Q:** What common approach do Rhode Island HB 5872, Montana SB 25, and North Dakota HB 1167 share?

**Citations (10):** `2023__RI H   5866, 2023__RI S    146, 2023__RI H   6423, 2023__RI H   6423, 2023__RI S    146, 2023__RI H   5734, 2023__RI S    117, 2023__RI H   5866, 2023__RI H   5734, 2023__RI H   6286`

**Applied filters:** `{"routing_path": "agent", "state": "Rhode Island"}`

**A:**

> The bills "Rhode Island HB 5872", "Montana SB 25", and "North Dakota HB 1167" could not be found. Please ensure the bill IDs are correct and from the 2023 legislative session.

### VH04 -- very_hard / filter_list_compare

**Q:** List 2025 enacted bills that address the use of AI in employment, and what do they require of employers or state agencies?

**Citations (10):** `2025__IL S    2394, 2025__NY S    822, 2025__IL S    2394, 2025__NY S    822, 2025__WV H    3187, 2025__WV H    3187, 2025__CA S    47, 2025__NM H    20, 2025__ND H    1448, 2025__CA S    253`

**Applied filters:** `{"routing_path": "agent", "status_bucket": "Enacted", "topics": ["Effect on Labor/Employment", "Effect on Labor/Employment Government Use"], "year": 2025}`

**A:**

> In 2025, five enacted bills address the use of AI in employment:
> 
> *   **NY S 822** (New York): This bill regulates the use of automated employment decision-making tools by state agencies. It requires state agencies to ensure these tools are subject to "continued and operational meaningful human review" and cannot procure or acquire services utilizing such tools unless they include this review. The use of these tools cannot affect existing employee rights under collective bargaining agreements or result in employee discharge, displacement, or loss of position. "Meaningful human review" is defined as review, oversight, and control by individuals who understand the tool's risks, limitations, and functionality, are trained to use it, and have the authority to intervene or alter decisions.
> 
> *   **WV H 3187** (West Virginia): This bill establishes a Task Force on Artificial Intelligence to study and make recommendations regarding AI, including its use in the workforce and its effect on emplo...