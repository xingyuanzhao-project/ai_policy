Section structure organises the theory-side papers cited in `docs/lit_rev_theory.docx`, which supplies the framing for `docs/mpsa_draft_outline.md` lines 67-73 ("many states are introducing AI bills, but the focus are all over the place", "no way to research unless clearly define and draw the scope", "treated like buzzwords, not solid concepts"). Format mirrors `docs/lit_rev_eval.md` and `docs/lit_rev_ner.md`, with an added `main takeaway` field per current instructions.

---

# Landscape and cross-jurisdiction qualitative overviews of US AI legislation

**(Yoo & Lai, 2020)**
### Regulation of Algorithmic Tools in the United States
- https://scholarship.law.upenn.edu/faculty_scholarship/2246/  (Journal of Law & Economic Regulation 13: 7-22; also U Penn Law Public Law Research Paper No. 21-04)
- method: qualitative legal and policy survey. Catalogues US federal initiatives (presidential directives, responding-agency guidance), early Biden-administration signals, sector-specific federal efforts, pending federal legislative proposals, and state and local actions. No quantitative data.
- required data: primary US federal documents (executive orders, agency guidance, pending bills); state and local regulatory documents; industry and technical-society standards.
- main takeaway: as of 2020 US AI policy is only nascent and fragmented: momentum is built from research-funding calls, piecemeal guidance, proposals, and state/local initiatives rather than any comprehensive federal statute. The authors frame the US regulatory trajectory as an ongoing search for balance between public safety / transparency and promoting innovation and global competitiveness.

**(Kuteynikov, Izhaev, Zenin & Lebedev, 2022)**
### Key Approaches to the Legal Regulation of Artificial Intelligence
- https://doi.org/10.21684/2411-7897-2022-8-1-209-232  (Vestnik Tyumenskogo Gos. Universiteta / Tyumen State University Herald 8(1): 209-232)
- method: cross-jurisdictional qualitative review. Authors compile and contrast regulatory legal acts on AI adopted or under consideration in four jurisdictions (EU, US, PRC, Russian Federation) and derive the dominant approach used in each. Applies methods of objectivity, specificity, and pluralism; no quantitative data.
- required data: official regulatory and legal acts / draft legislation from the EU, US, China, and Russia; Council of Europe reports; EU AI Act draft.
- main takeaway: the four jurisdictions cluster into four distinct regulatory postures. The EU uses a risk-oriented approach with ex-ante requirements; the US regulates AI inside its existing legal framework, making it the most flexible of the four; China builds ambitious, centrally-coordinated control of the AI lifecycle; Russia identifies cross-cutting issues and attempts sector-by-sector regulation. The US approach is comparatively minimal and reactive rather than comprehensive.

**(Sheikin, 2024)**
### Principles of Legislative Regulation of Artificial Intelligence in the USA and Its Impact on the Development of the Technology Sector
- [In Russian]  (no DOI located; cited only within `lit_rev_theory.docx` and commonly indexed in Russian legal-studies repositories; primary citation is the `lit_rev_theory.docx` entry as of publication)
- method: qualitative legal analysis from an outside (Russian-scholar) vantage point; reviews US federal and state regulatory approaches to AI and their effect on the tech sector.
- required data: US legal and regulatory documents governing AI (federal guidance, state-level bills); industry and sector statistics where available.
- main takeaway: US AI legislation is comparatively flexible and favours accommodating existing frameworks over creating a comprehensive AI-specific statute. This flexibility has implications (both enabling and uncertain) for technology-sector development. Corroborates the "US = flexible / minimal" reading also found in Kuteynikov et al. (2022) and Yoo & Lai (2020).

**(Wang & Haak, 2024)**
### Regulating Artificial Intelligence in the European Union and the United States
- https://doi.org/10.47611/jsrhs.v13i2.6798  (Journal of Student Research 13(2); St. John's School + Mentor High School; high-school research article)
- method: qualitative comparative literature review. Describes the EU's legislative development process (High-Level Expert Group, 2020 White Paper, April-2021 AI Regulatory Act proposal, risk-based tiers, EAIB enforcement) and compares with US federal and state-level efforts (AI Bill of Rights blueprint, FTC investigations, 2023 EO, AIRIA proposal). Draws on secondary sources only (Google Scholar and government websites).
- required data: EU primary documents (AI Act proposal, White Paper, HLEG outputs); US primary documents (AI Bill of Rights, 2023 EO, AIRIA text, state-level snapshots); secondary commentary.
- main takeaway: the US "has not acted as swiftly and lacks major, tangible pieces of legislation for the regulation of AI" compared with the EU. Only ~1/4 of US states had enacted AI legislation by 2023, and federal efforts remain at the blueprint / executive-order stage; without federal leadership the US risks a disorganised, dispute-prone regulatory patchwork. The paper recommends that the US adopt EU-style risk-tiered regulation (reflected already in the AIRIA proposal).

---

# Drivers of US state AI legislation (quantitative political economy)

**(Parinandi, Crosson, Peterson & Nadarevic, 2024)**
### Investigating the Politics and Content of US State Artificial Intelligence Legislation
- https://doi.org/10.1017/bap.2023.40  (Business and Politics 26(2): 240-262)
- method: quantitative political-economy analysis with two layers. (1) State-level event-history logistic regression of AI policy adoption (2017-2022), where the dependent variable is a state's first adoption (Model 1) or any adoption (Model 2) of an AI bill; independent variables include per-capita income, unemployment, state economic activity, annual inflation, life expectancy, % BA+, unified Democratic government, governor party, Shor-McCarty average legislator ideology, legislative professionalism, neighbour and ideological-neighbour AI adoption, Trump vote share, linear year, and prior AI bills. Robustness: bill-introduction TSCS model and ordered-progression model. (2) Legislator-level roll-call analysis: all AI-relevant final-passage roll calls from LegiScan are hand-coded as consumer-protection vs. economic-development (two coders, inconclusive cases resolved jointly) and individual "yes" votes are modelled on party and Shor-McCarty ideology plus legislator / state controls.
- required data: NCSL Artificial Intelligence state legislation database (2018-2022, last updated 26 Aug 2022), LegiScan bill text and roll-call data (~1,700 votes across 32 final-passage roll calls in 19 states, 2019-2022), Shor-McCarty state-legislator ideal points, BEA / BLS economic indicators, presidential vote shares.
- main takeaway: at the state level, economic stress (rising unemployment and inflation) significantly reduces AI-adoption probability, and unified Democratic government marginally increases it, while ideology and neighbour adoption do not predict adoption. At the legislator level, liberals and Democrats are significantly more likely to support consumer-protection AI bills, but economic-development AI bills lack clear ideological / partisan structure. Party unity on AI roll calls has risen over time. Overall: economic concerns dominate state-level adoption, while traditional partisan fault lines are emerging specifically around consumer-protection content - not around AI regulation in general.

**(DeFranco & Biersmith, 2024)**
### Assessing the State of AI Policy
- https://doi.org/10.48550/arXiv.2407.21717  (arXiv:2407.21717; submitted 31 Jul 2024)
- method: qualitative landscape review + overlap and gap analysis. Authors catalogue AI legislation and directives at international, US federal, US state, and US city levels, plus business standards and technical-society initiatives, and cross-tabulate coverage to identify overlaps and unaddressed risk domains. Output is a "reference guide" with recommendations.
- required data: international regulatory documents (EU AI Act, OECD, ISO), US federal directives (EOs, agency guidance), NCSL state-legislation database, municipal AI directives, industry standards (NIST AI RMF, business association codes), technical-society initiatives (IEEE, ACM).
- main takeaway: policymakers lack the technical expertise to judge emerging AI technologies on their own and therefore depend on expert opinion; they are better served when that is paired with a structured map of existing guidelines at every jurisdictional level. The paper provides such a map and flags risks (physical injury, bias, unfair outcomes) that remain unevenly covered. Directly motivates this project's framing that definitional and scope confusion is pervasive at the state level because policymakers inherit a scattered landscape rather than a single coherent one.

---

# Content and effects of US AI legislation

**(Oduro, Moss & Metcalf, 2022)**
### Obligations to Assess: Recent Trends in AI Accountability Regulations
- https://doi.org/10.1016/j.patter.2022.100608  (Patterns 3(11): 100608; Data & Society + Intel Labs)
- method: qualitative doctrinal analysis of four illustrative impact-assessment instruments. Authors examine each bill on three axes: (i) identifying and documenting harms, (ii) public transparency, and (iii) anti-discrimination and disparate impact. The four instruments are the US Algorithmic Accountability Act of 2022 (AAA), NYC Local Law / Int. 1894 on bias audits of employment decision tools, California AB 13 (Automated Decision Systems Accountability Act of 2021), and the EU AI Act proposal (April 2021). Also notes the ADPPA as a near-identical successor vehicle.
- required data: bill texts of AAA 2022, NYC Int. 1894, CA AB 13, and the EU AI Act proposal; supplementary FTC / EEOC guidance; prior algorithmic-accountability scholarship.
- main takeaway: proposed AI regulations are converging on "obligation to assess" as the governance model: they require developers to produce accountability documentation on harms, transparency, and disparate impact that goes beyond traditional technical audits. This shift reshapes how developers must build algorithmic systems, because it demands measuring effects on individuals, communities, and society and not just on technical performance. For AI governance to deliver, developer compliance practices and public-interest consultation infrastructure have to catch up.

**(DePaula, Gao, Mellouli, Luna-Reyes & Harrison, 2024)**
### Regulating the Machine: An Exploratory Study of US State Legislations Addressing Artificial Intelligence, 2019-2023
- https://doi.org/10.1145/3657054.3657148  (Proceedings of the 25th Annual International Conference on Digital Government Research, pp. 815-826)
- method: qualitative content analysis of 68 enacted US state AI legislations passed 2019-2023. Authors compiled the corpus from the NCSL AI legislation database, iteratively developed coding categories (domain, public-sector vs. industry target, novel issues addressed), then distributed bills so that each was independently coded by at least two authors with disagreements resolved jointly.
- required data: full text of 68 enacted state AI laws (2019-2023) via NCSL; authors' hand-coded categories covering domain (health, education, labour, advisory group, etc.), sector of regulation, and novel issues.
- main takeaway: US state AI legislation has been evolving and becoming more targeted - 36% of 2019-2023 bills deal with health (e.g. cancer detection, drug development) or education, ~21% create advisory groups to inventory state AI use, and ~30% regulate or incentivise private industry. Despite this growth, no federal law addresses AI comprehensively, so the states are building a disjointed patchwork that creates simultaneous challenges and opportunities for government agencies. Provides the empirical baseline that the 2025 follow-up below extends.

**(DePaula, Gao, Mellouli, Luna-Reyes & Harrison, 2025)**
### The Evolving AI Regulation Space: A Preliminary Analysis of US State Legislations Addressing AI, 2024
- https://doi.org/10.59490/dgo.2025.937  (Proceedings of the 26th Annual International Conference on Digital Government Research)
- method: extends the DePaula et al. (2024) protocol to 2024. Authors compile all 79 AI-related bills passed/enacted across the 50 US states in 2024 from NCSL's AI 2024 legislation page, iteratively refine coding categories, and have each bill independently coded by at least two authors, with commonalities / discrepancies discussed to reach agreement. Output is a preliminary poster-length analysis with a 2019-2024 time series and a 2024 state-level map.
- required data: 79 AI-related passed/enacted 2024 state bills (NCSL AI 2024 Legislation database); coding protocol from the 2024 paper.
- main takeaway: 2024 alone produced more state AI legislation than 2019-2023 combined. Bills continue to address healthcare and education at 2019-2023 rates; 2024 newly and substantially addresses generative AI and AI-generated content (deepfakes, synthetic election media, CSAM, litigation cause of action), and several states (California, Florida, Maryland, Utah) passed comprehensive governance frameworks. However, private-sector regulation dropped to ~11% of 2024 bills (from ~30%), AI bias and workforce development are addressed very unevenly, and AI definitions are inconsistent or absent - producing exactly the heterogeneous, buzzword-prone landscape framed in the draft outline's theory section.

---

# Navigating overlapping US AI regulatory frameworks (compliance and ethics)

**(Agbadamasi, Opoku, Adukpo & Mensah, 2025)**
### Navigating the Intersection of U.S. Regulatory Frameworks and Artificial Intelligence: Strategies for Ethical Compliance
- https://doi.org/10.30574/wjarr.2025.25.3.0814  (World Journal of Advanced Research and Reviews 25(3): 969-979)
- method: qualitative review article drawing on interdisciplinary literature. Authors (i) catalogue the current US AI-ethics and regulation landscape, (ii) doctrinally review three anchor statutes - Federal Trade Commission Act, Algorithmic Accountability Act, HIPAA - against contemporary AI-risk categories (algorithmic bias, privacy, transparency), and (iii) derive actionable compliance measures. No quantitative data.
- required data: US statutory texts (FTC Act, AAA, HIPAA and sectoral laws); interdisciplinary ethics and governance scholarship; empirical studies on bias in AI hiring / lending / law-enforcement and on AI-driven surveillance.
- main takeaway: existing US statutes provide foundational oversight but are not structured to handle the scale and complexity of contemporary AI systems - bias in hiring, lending, and policing persists and privacy protections are increasingly defeated by AI-driven surveillance. The authors argue for a unified, comprehensive US regulatory approach that (i) embeds ethical principles directly into legislation, (ii) fosters inter-agency coordination, and (iii) promotes international harmonisation; i.e. proactive rather than reactive governance. Reinforces the "overlapping and incomplete" theme that motivates mechanism-level rather than label-level analysis of AI bills.
