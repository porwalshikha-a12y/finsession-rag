# Section 5: RESULTS - Extracted Data

Generated: 2026-09-15 15:56:34


## 5.1 Primary Findings: Cost Reduction and Answer Correctness


### Table 5.1: Accuracy, Efficiency, and Memory Utilization by Arm


| Arm | Accuracy | Tokens/Q | Total Tokens | Hit Rate | Avg Hits |

|-----|----------|----------|--------------|----------|----------|

| A1_unbounded | 88.5% | 7479 | 388,908 | 34.6% | 0.35 |

| A4_redundancy | 88.5% | 7463 | 388,063 | 34.6% | 0.37 |

| A6_hybrid | 86.5% | 7326 | 380,926 | 32.7% | 0.33 |

| A5_cost_aware | 86.5% | 7136 | 371,072 | 30.8% | 0.35 |

| A3_lfu | 86.5% | 7133 | 370,928 | 32.7% | 0.40 |

| A2_lru | 84.6% | 7409 | 385,251 | 30.8% | 0.33 |

| A0_no_memory | 84.6% | 7923 | 412,004 | 0.0% | 0.00 |



### Table 5.2: Cost Efficiency - Tokens per Correct Answer


| Arm | Total Q | Correct | Total Tokens | Tokens/Correct |

|-----|---------|---------|--------------|----------------|

| A3_lfu | 52 | 45 | 370,928 | 8243 |

| A5_cost_aware | 52 | 45 | 371,072 | 8246 |

| A4_redundancy | 52 | 46 | 388,063 | 8436 |

| A1_unbounded | 52 | 46 | 388,908 | 8455 |

| A6_hybrid | 52 | 45 | 380,926 | 8465 |

| A2_lru | 52 | 44 | 385,251 | 8756 |

| A0_no_memory | 52 | 44 | 412,004 | 9364 |



## 5.2 Memory Hit Rate and Consistency


### Table 5.3: Memory Performance Metrics


| Arm | Total Lookups | Total Hits | Hit Rate | Avg Hits/Q |

|-----|---------------|------------|----------|------------|

| A3_lfu | 52 | 21 | 40.4% | 0.40 |

| A4_redundancy | 52 | 19 | 36.5% | 0.37 |

| A1_unbounded | 52 | 18 | 34.6% | 0.35 |

| A5_cost_aware | 52 | 18 | 34.6% | 0.35 |

| A2_lru | 52 | 17 | 32.7% | 0.33 |

| A6_hybrid | 52 | 17 | 32.7% | 0.33 |

| A0_no_memory | 52 | 0 | 0.0% | 0.00 |



## 5.3 Grounding Fidelity: Did We Block Poison?


### Table 5.4: Grounding Gate Performance


| Arm | Total Q | Grounded | Rejected | Rejection Rate |

|-----|---------|----------|----------|----------------|

| A0_no_memory | 52 | 52 | 0 | 0.0% |

| A1_unbounded | 52 | 52 | 0 | 0.0% |

| A2_lru | 52 | 52 | 0 | 0.0% |

| A3_lfu | 52 | 52 | 0 | 0.0% |

| A4_redundancy | 52 | 52 | 0 | 0.0% |

| A5_cost_aware | 52 | 52 | 0 | 0.0% |

| A6_hybrid | 52 | 52 | 0 | 0.0% |



## 5.4 Robustness Across Companies and Query Types


### Table 5.5: Accuracy by Company and Arm


| Company | A0_no_memory | A1_unbounded | A2_lru | A3_lfu | A4_redundancy | A5_cost_aware | A6_hybrid |

|---------|---|---|---|---|---|---|---|

| Apple | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

| Johnson & Johnson | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

| Microsoft | 100.0% | 100.0% | 75.0% | 100.0% | 100.0% | 100.0% | 100.0% |

| PepsiCo | 75.0% | 75.0% | 75.0% | 75.0% | 75.0% | 75.0% | 75.0% |



### Table 5.6: Verdict Distribution by Arm


| Arm | Verdict | Count | % of Arm |

|-----|---------|-------|----------|

| A0_no_memory | SUPPORTED | 44 | 84.6% |

| A0_no_memory | PARTIAL | 8 | 15.4% |

| A1_unbounded | SUPPORTED | 46 | 88.5% |

| A1_unbounded | PARTIAL | 6 | 11.5% |

| A2_lru | SUPPORTED | 44 | 84.6% |

| A2_lru | PARTIAL | 8 | 15.4% |

| A3_lfu | SUPPORTED | 45 | 86.5% |

| A3_lfu | PARTIAL | 7 | 13.5% |

| A4_redundancy | SUPPORTED | 46 | 88.5% |

| A4_redundancy | PARTIAL | 6 | 11.5% |

| A5_cost_aware | SUPPORTED | 45 | 86.5% |

| A5_cost_aware | PARTIAL | 7 | 13.5% |

| A6_hybrid | SUPPORTED | 45 | 86.5% |

| A6_hybrid | PARTIAL | 7 | 13.5% |



### Table 5.7: Error Classification


| Error Type | Count | Arms Affected |

|-----------|-------|---------------|

| No error | 364 | A0_no_memory, A1_unbounded, A2_lru, A3_lfu, A4_redundancy, A5_cost_aware, A6_hybrid |


