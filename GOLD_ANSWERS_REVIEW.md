# Gold answers for verification

Extracted directly from the filing PDFs with pdfplumber — independent of the RAG system they grade.
All segment/category subtotals reconcile to their reported totals.

65 queries across 13 sessions. Edit `data/gold_answers.json` to correct anything, then run the migration.

Paraphrase probes inherit their original's gold answer by construction — a rephrasing must resolve to the same fact.

---

## apple_fy2022_basics — Apple (FY2022)

**q_001** · _tier_a_factoid_
> What was Apple's total net sales revenue in FY2022?

$394,328 million (approximately $394.3 billion) in total net sales for fiscal 2022.

<sub>Source: APPLE_2022_10K p.29 (Consolidated Statements of Operations)</sub>

---

**q_002** · _tier_a_factoid_
> How many operating segments did Apple report in FY2022?

Five reportable segments: Americas, Europe, Greater China, Japan, and Rest of Asia Pacific.

⚠️ Apple's reportable segments are geographic, not product lines.

<sub>Source: APPLE_2022_10K p.25 (Segment Operating Performance)</sub>

---

**q_003** · _terminology_mismatch_probe_
> What was Apple's gross profit in fiscal year 2022?

$170,782 million. Apple reports this line as "gross margin" rather than "gross profit" (total net sales $394,328M less total cost of sales $223,546M).

⚠️ Terminology probe: the filing never uses the phrase 'gross profit'.

<sub>Source: APPLE_2022_10K p.29</sub>

---

**q_004** · _multi_hop_
> Compare Apple's revenue in 2022 vs 2021.

Net sales rose from $365,817 million in 2021 to $394,328 million in 2022 — an increase of $28,511 million, or 7.8%.

<sub>Source: APPLE_2022_10K p.29</sub>

---

**q_053** · _paraphrase_probe_ · paraphrase of **q_001**
> For fiscal 2022, how much did Apple take in from sales overall?

$394,328 million (approximately $394.3 billion) in total net sales for fiscal 2022.

⚠️ Paraphrase of q_001; same gold answer by construction.

<sub>Source: APPLE_2022_10K p.29 (Consolidated Statements of Operations)</sub>

---

## apple_products_2020_2022 — Apple (FY2020-2022)

**q_009** · _tier_a_factoid_
> What were Apple's main product categories by revenue in 2022?

By net sales in fiscal 2022: iPhone $205,489M; Services $78,129M; Wearables, Home and Accessories $41,241M; Mac $40,177M; iPad $29,292M.

<sub>Source: APPLE_2022_10K p.24 (Products and Services Performance)</sub>

---

**q_010** · _tier_a_factoid_
> Which product line had the highest revenue for Apple in 2022?

iPhone, at $205,489 million in fiscal 2022.

<sub>Source: APPLE_2022_10K p.24</sub>

---

**q_011** · _multi_hop_
> Compare Services revenue between 2020 and 2022.

Services net sales grew from $53,768 million in 2020 to $78,129 million in 2022 — up $24,361 million, or 45.3%.

<sub>Source: APPLE_2022_10K p.24</sub>

---

**q_012** · _arithmetic_probe_
> What was the growth rate for Apple's Wearables category from 2021 to 2022?

Wearables, Home and Accessories grew from $38,367M in 2021 to $41,241M in 2022 — 7.5% (the filing rounds to 7%).

<sub>Source: APPLE_2022_10K p.24</sub>

---

**q_054** · _paraphrase_probe_ · paraphrase of **q_010**
> In 2022, which of Apple's product lines brought in the most money?

iPhone, at $205,489 million in fiscal 2022.

⚠️ Paraphrase of q_010; same gold answer by construction.

<sub>Source: APPLE_2022_10K p.24</sub>

---

## apple_profitability_trend — Apple (FY2020-2022)

**q_013** · _tier_a_factoid_
> What was Apple's operating income in 2022?

$119,437 million in fiscal 2022.

<sub>Source: APPLE_2022_10K p.29</sub>

---

**q_014** · _multi_hop_
> Did Apple's net income increase from 2021 to 2022?

Yes. Net income rose from $94,680 million in 2021 to $99,803 million in 2022 — up $5,123 million, or 5.4%.

<sub>Source: APPLE_2022_10K p.29</sub>

---

**q_015** · _arithmetic_probe_
> Calculate Apple's gross profit margin for 2022.

43.3% (gross margin $170,782M ÷ total net sales $394,328M).

<sub>Source: APPLE_2022_10K p.29</sub>

---

**q_016** · _multi_hop_
> How did Apple's operating expenses trend from 2020 to 2022?

Total operating expenses rose each year: $38,668M (2020), $43,887M (2021, +13.5%), $51,345M (2022, +17.0%).

<sub>Source: APPLE_2022_10K p.29</sub>

---

**q_055** · _paraphrase_probe_ · paraphrase of **q_013**
> How much operating profit did Apple earn in fiscal 2022?

$119,437 million in fiscal 2022.

⚠️ Paraphrase of q_013; same gold answer by construction.

<sub>Source: APPLE_2022_10K p.29</sub>

---

## apple_segments_multi_year — Apple (FY2020-2022)

**q_005** · _tier_a_factoid_
> List Apple's operating segments as reported in 2022.

Americas, Europe, Greater China, Japan, and Rest of Asia Pacific.

<sub>Source: APPLE_2022_10K p.25</sub>

---

**q_006** · _tier_a_factoid_
> Which segment generated the most revenue for Apple in 2022?

Americas, with net sales of $169,658 million in 2022.

<sub>Source: APPLE_2022_10K p.25</sub>

---

**q_007** · _arithmetic_probe_
> What percentage of Apple's revenue came from Americas in 2022?

Approximately 43% ($169,658M of $394,328M total net sales).

<sub>Source: APPLE_2022_10K p.25</sub>

---

**q_008** · _multi_hop_
> How did Apple's China segment revenue change from 2021 to 2022?

Greater China net sales rose from $68,366 million in 2021 to $74,200 million in 2022 — up $5,834 million, or 8.5% (the filing rounds this to 9%).

<sub>Source: APPLE_2022_10K p.25</sub>

---

**q_056** · _paraphrase_probe_ · paraphrase of **q_006**
> Among Apple's reportable segments, which was largest by sales in 2022?

Americas, with net sales of $169,658 million in 2022.

⚠️ Paraphrase of q_006; same gold answer by construction.

<sub>Source: APPLE_2022_10K p.25</sub>

---

## jnj_fy2022_overview — Johnson & Johnson (FY2022)

**q_033** · _tier_a_factoid_
> What was Johnson & Johnson's total revenue in 2022?

$94,943 million in sales to customers for 2022.

<sub>Source: JOHNSON_JOHNSON_2022_10K p.47 (Consolidated Statements of Earnings)</sub>

---

**q_034** · _tier_a_factoid_
> How many business segments does Johnson & Johnson operate?

Three: Consumer Health, Pharmaceutical, and MedTech.

<sub>Source: JOHNSON_JOHNSON_2022_10K p.7</sub>

---

**q_035** · _terminology_mismatch_probe_
> What was Johnson & Johnson's net earnings in 2022?

$17,941 million in 2022 (down from $20,878 million in 2021).

<sub>Source: JOHNSON_JOHNSON_2022_10K p.47</sub>

---

**q_036** · _multi_hop_
> Compare Johnson & Johnson's revenue in 2022 vs 2021.

Sales rose from $93,775 million in 2021 to $94,943 million in 2022 — up $1,168 million, or 1.2%.

<sub>Source: JOHNSON_JOHNSON_2022_10K p.47</sub>

---

**q_057** · _paraphrase_probe_ · paraphrase of **q_033**
> How much did Johnson & Johnson sell to customers during 2022?

$94,943 million in sales to customers for 2022.

⚠️ Paraphrase of q_033; same gold answer by construction.

<sub>Source: JOHNSON_JOHNSON_2022_10K p.47 (Consolidated Statements of Earnings)</sub>

---

## jnj_healthcare_focus — Johnson & Johnson (FY2020-2022)

**q_041** · _tier_a_factoid_
> What were the main product categories within Johnson & Johnson's Pharmaceutical segment?

The Pharmaceutical segment's therapeutic areas in 2022 were Immunology $16,935M, Oncology $15,983M, Neuroscience $6,893M, Infectious Diseases $5,449M, Cardiovascular/Metabolism/Other $3,887M, and Pulmonary Hypertension $3,417M.

<sub>Source: JOHNSON_JOHNSON_2022_10K p.30</sub>

---

**q_042** · _multi_hop_
> Compare Johnson & Johnson's Consumer Health revenue between 2020 and 2022.

Consumer Health sales rose from $14,053 million in 2020 to $14,953 million in 2022 — up $900 million, or 6.4%.

<sub>Source: JOHNSON_JOHNSON_2020_10K p.34; JOHNSON_JOHNSON_2022_10K p.36</sub>

---

**q_043** · _arithmetic_probe_
> Calculate the combined revenue from Pharmaceutical and MedTech for 2022.

$79,990 million (Pharmaceutical $52,563M + MedTech $27,427M).

<sub>Source: JOHNSON_JOHNSON_2022_10K p.36</sub>

---

**q_044** · _terminology_mismatch_probe_
> What was Johnson & Johnson's operating profit in 2022?

J&J does not report a line called "operating profit". Segment earnings before tax were $23,438 million in 2022; worldwide income before tax was $21,725 million after unallocated expenses and Consumer Health separation costs.

⚠️ Terminology probe: 'operating profit' is not a line J&J reports; it maps to segment income before tax.

<sub>Source: JOHNSON_JOHNSON_2022_10K p.36/47</sub>

---

**q_058** · _paraphrase_probe_ · paraphrase of **q_042**
> Between 2020 and 2022, what happened to Consumer Health sales at J&J?

Consumer Health sales rose from $14,053 million in 2020 to $14,953 million in 2022 — up $900 million, or 6.4%.

⚠️ Paraphrase of q_042; same gold answer by construction.

<sub>Source: JOHNSON_JOHNSON_2020_10K p.34; JOHNSON_JOHNSON_2022_10K p.36</sub>

---

## jnj_segments_2020_2022 — Johnson & Johnson (FY2020-2022)

**q_037** · _tier_a_factoid_
> What are Johnson & Johnson's three main business segments?

Consumer Health, Pharmaceutical, and MedTech (MedTech was previously reported as Medical Devices).

<sub>Source: JOHNSON_JOHNSON_2022_10K p.7</sub>

---

**q_038** · _tier_a_factoid_
> Which segment had the highest revenue for Johnson & Johnson in 2022?

Pharmaceutical, at $52,563 million in 2022.

<sub>Source: JOHNSON_JOHNSON_2022_10K p.30/36</sub>

---

**q_039** · _arithmetic_probe_
> What percentage of J&J's revenue came from the Pharmaceutical segment in 2022?

Approximately 55.4% ($52,563M of $94,943M).

<sub>Source: JOHNSON_JOHNSON_2022_10K p.30/36</sub>

---

**q_040** · _multi_hop_
> How did the MedTech segment revenue change from 2021 to 2022?

MedTech revenue rose from $27,060 million in 2021 to $27,427 million in 2022 — up $367 million, or 1.4%.

<sub>Source: JOHNSON_JOHNSON_2022_10K p.36</sub>

---

**q_059** · _paraphrase_probe_ · paraphrase of **q_038**
> In 2022, which J&J business segment was the biggest by sales?

Pharmaceutical, at $52,563 million in 2022.

⚠️ Paraphrase of q_038; same gold answer by construction.

<sub>Source: JOHNSON_JOHNSON_2022_10K p.30/36</sub>

---

## microsoft_cloud_revenue — Microsoft (FY2020-2022)

**q_025** · _tier_a_factoid_
> What were the main revenue sources within Microsoft's Intelligent Cloud segment in 2022?

Two: server products and cloud services (Azure and other cloud services; SQL Server, Windows Server, Visual Studio, System Center and related CALs; Nuance and GitHub), and Enterprise Services (Enterprise Support Services, Microsoft Consulting Services, Nuance professional services).

<sub>Source: MICROSOFT_2022_10K (Intelligent Cloud segment description)</sub>

---

**q_026** · _multi_hop_
> Compare Azure revenue growth between 2021 and 2022.

Azure and other cloud services revenue grew 50% in FY2021 and 45% in FY2022 — still rapid, but decelerating by about 5 percentage points.

<sub>Source: MICROSOFT_2022_10K p.40/45; MICROSOFT_2021_10K p.39</sub>

---

**q_027** · _arithmetic_probe_
> Calculate the combined revenue from Productivity and Cloud segments for 2022.

$138,615 million (Productivity and Business Processes $63,364M + Intelligent Cloud $75,251M).

<sub>Source: MICROSOFT_2022_10K p.94</sub>

---

**q_028** · _multi_hop_
> What was the trend in Microsoft's total employees from 2020 to 2022?

Headcount rose from approximately 163,000 (June 2020) to 181,000 (June 2021) to 221,000 (June 2022) — an increase of about 36% over the two years.

<sub>Source: MICROSOFT 2020/2021/2022 10-K, Human Capital sections</sub>

---

**q_060** · _paraphrase_probe_ · paraphrase of **q_027**
> Added together, what did Microsoft's Productivity and Intelligent Cloud segments bring in for FY2022?

$138,615 million (Productivity and Business Processes $63,364M + Intelligent Cloud $75,251M).

⚠️ Paraphrase of q_027; same gold answer by construction.

<sub>Source: MICROSOFT_2022_10K p.94</sub>

---

## microsoft_fy2022_overview — Microsoft (FY2022)

**q_017** · _tier_a_factoid_
> What was Microsoft's total revenue in fiscal year 2022?

$198,270 million in fiscal year 2022 (year ended 30 June 2022).

<sub>Source: MICROSOFT_2022_10K p.57 (Income Statements)</sub>

---

**q_018** · _tier_a_factoid_
> How many business segments does Microsoft operate?

Three: Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.

<sub>Source: MICROSOFT_2022_10K p.94 (Segment Information)</sub>

---

**q_019** · _terminology_mismatch_probe_
> What was Microsoft's net income in FY2022?

$72,738 million in fiscal 2022.

<sub>Source: MICROSOFT_2022_10K p.57</sub>

---

**q_020** · _multi_hop_
> Compare Microsoft's FY2022 revenue to FY2021.

Revenue rose from $168,088 million in FY2021 to $198,270 million in FY2022 — up $30,182 million, or 18.0%.

<sub>Source: MICROSOFT_2022_10K p.57</sub>

---

**q_061** · _paraphrase_probe_ · paraphrase of **q_017**
> How much revenue in total did Microsoft report for FY2022?

$198,270 million in fiscal year 2022 (year ended 30 June 2022).

⚠️ Paraphrase of q_017; same gold answer by construction.

<sub>Source: MICROSOFT_2022_10K p.57 (Income Statements)</sub>

---

## microsoft_profitability — Microsoft (FY2020-2022)

**q_029** · _tier_a_factoid_
> What was Microsoft's operating income in 2022?

$83,383 million in fiscal 2022.

<sub>Source: MICROSOFT_2022_10K p.57</sub>

---

**q_030** · _arithmetic_probe_
> Did Microsoft's gross margin percentage improve from 2021 to 2022?

No — gross margin percentage slipped from 68.9% in FY2021 (115,856/168,088) to 68.4% in FY2022 (135,620/198,270), even though gross margin in dollars rose from $115,856M to $135,620M.

<sub>Source: MICROSOFT_2022_10K p.57</sub>

---

**q_031** · _arithmetic_probe_
> What percentage growth occurred in Microsoft's revenue from 2020 to 2022?

38.6% (from $143,015 million in FY2020 to $198,270 million in FY2022).

<sub>Source: MICROSOFT_2022_10K p.57</sub>

---

**q_032** · _multi_hop_
> How did Microsoft's capital expenditures change from 2021 to 2022?

Additions to property and equipment rose from $20,622 million in FY2021 to $23,886 million in FY2022 — up $3,264 million, or 15.8%.

<sub>Source: MICROSOFT_2022_10K p.60 (Cash Flows Statements)</sub>

---

**q_062** · _paraphrase_probe_ · paraphrase of **q_029**
> In fiscal 2022, how much operating income did Microsoft make?

$83,383 million in fiscal 2022.

⚠️ Paraphrase of q_029; same gold answer by construction.

<sub>Source: MICROSOFT_2022_10K p.57</sub>

---

## microsoft_segments_2020_2022 — Microsoft (FY2020-2022)

**q_021** · _tier_a_factoid_
> What are Microsoft's three main business segments as of 2022?

Productivity and Business Processes; Intelligent Cloud; More Personal Computing.

<sub>Source: MICROSOFT_2022_10K p.94</sub>

---

**q_022** · _tier_a_factoid_
> Which segment generated the most revenue for Microsoft in 2022?

Intelligent Cloud, at $75,251 million in FY2022.

<sub>Source: MICROSOFT_2022_10K p.94</sub>

---

**q_023** · _arithmetic_probe_
> What percentage of Microsoft's revenue came from Intelligent Cloud in 2022?

Approximately 38% ($75,251M of $198,270M total revenue).

<sub>Source: MICROSOFT_2022_10K p.94</sub>

---

**q_024** · _multi_hop_
> How did the Intelligent Cloud segment revenue grow from 2021 to 2022?

Intelligent Cloud revenue grew from $60,080 million in FY2021 to $75,251 million in FY2022 — up $15,171 million, or 25.3% (reported as 25%).

<sub>Source: MICROSOFT_2022_10K p.94</sub>

---

**q_063** · _paraphrase_probe_ · paraphrase of **q_022**
> Of Microsoft's three segments, which was largest by FY2022 revenue?

Intelligent Cloud, at $75,251 million in FY2022.

⚠️ Paraphrase of q_022; same gold answer by construction.

<sub>Source: MICROSOFT_2022_10K p.94</sub>

---

## pepsico_fy2022_financials — PepsiCo (FY2022)

**q_045** · _tier_a_factoid_
> What was PepsiCo's total net revenue in 2022?

$86,392 million in net revenue for 2022.

<sub>Source: PEPSICO_2022_10K p.62 (Consolidated Statement of Income)</sub>

---

**q_046** · _tier_a_factoid_
> How many business segments does PepsiCo operate?

Seven reportable segments (divisions): FLNA, QFNA, PBNA, LatAm, Europe, AMESA and APAC.

<sub>Source: PEPSICO_2022_10K p.69</sub>

---

**q_047** · _terminology_mismatch_probe_
> What was PepsiCo's net income in 2022?

$8,910 million attributable to PepsiCo in 2022 (total net income including non-controlling interests was $8,978 million).

<sub>Source: PEPSICO_2022_10K p.62</sub>

---

**q_048** · _multi_hop_
> Compare PepsiCo's revenue in 2022 vs 2021.

Net revenue rose from $79,474 million in 2021 to $86,392 million in 2022 — up $6,918 million, or 8.7%.

<sub>Source: PEPSICO_2022_10K p.62</sub>

---

**q_064** · _paraphrase_probe_ · paraphrase of **q_045**
> How much net revenue did PepsiCo report for 2022?

$86,392 million in net revenue for 2022.

⚠️ Paraphrase of q_045; same gold answer by construction.

<sub>Source: PEPSICO_2022_10K p.62 (Consolidated Statement of Income)</sub>

---

## pepsico_segments_2020_2022 — PepsiCo (FY2020-2022)

**q_049** · _tier_a_factoid_
> What are PepsiCo's main business segments as of 2022?

Frito-Lay North America (FLNA), Quaker Foods North America (QFNA), PepsiCo Beverages North America (PBNA), Latin America (LatAm), Europe, Africa/Middle East/South Asia (AMESA), and Asia Pacific/Australia-New Zealand/China (APAC).

<sub>Source: PEPSICO_2022_10K p.69</sub>

---

**q_050** · _tier_a_factoid_
> Which segment generated the most revenue for PepsiCo in 2022?

PBNA (PepsiCo Beverages North America), at $26,213 million in 2022.

<sub>Source: PEPSICO_2022_10K p.70</sub>

---

**q_051** · _arithmetic_probe_
> Calculate the revenue contribution percentage from Frito-Lay (FLNA) in 2022.

Approximately 27% ($23,291M of $86,392M net revenue).

<sub>Source: PEPSICO_2022_10K p.70</sub>

---

**q_052** · _multi_hop_
> How did PepsiCo's geographic segment revenues change from 2021 to 2022?

2021 to 2022: LatAm $8,108M to $9,779M (+20.6%); Europe $13,038M to $12,724M (-2.4%); AMESA $6,078M to $6,438M (+5.9%); APAC $4,615M to $4,787M (+3.7%).

<sub>Source: PEPSICO_2022_10K p.70</sub>

---

**q_065** · _paraphrase_probe_ · paraphrase of **q_050**
> Which PepsiCo division had the largest net revenue in 2022?

PBNA (PepsiCo Beverages North America), at $26,213 million in 2022.

⚠️ Paraphrase of q_050; same gold answer by construction.

<sub>Source: PEPSICO_2022_10K p.70</sub>

---
