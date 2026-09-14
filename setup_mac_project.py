#!/usr/bin/env python3
"""
Run on your Mac to set up all session files and new Python modules.
Copy this script to ~/PycharmProjects/finsession-rag/ and run it.

Usage:
    python3 setup_mac_project.py
"""

import json
from pathlib import Path

# All 13 sessions defined here
SESSIONS = {
    "apple_fy2022_basics": {
        "session_id": "apple_fy2022_basics",
        "company": "Apple",
        "period": "FY2022",
        "queries": [
            {"query_id": "q_001", "question": "What was Apple's total net sales revenue in FY2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "revenue"},
            {"query_id": "q_002", "question": "How many operating segments did Apple report in FY2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "segments"},
            {"query_id": "q_003", "question": "What was Apple's gross profit in fiscal year 2022?", "query_type": "terminology_mismatch_probe", "expected_answer_marker": "gross profit"},
            {"query_id": "q_004", "question": "Compare Apple's revenue in 2022 vs 2021.", "query_type": "multi_hop", "expected_answer_marker": "comparison"},
        ]
    },
    "apple_segments_multi_year": {
        "session_id": "apple_segments_multi_year",
        "company": "Apple",
        "period": "FY2020-2022",
        "queries": [
            {"query_id": "q_005", "question": "List Apple's operating segments as reported in 2022.", "query_type": "tier_a_factoid", "expected_answer_marker": "Americas, Europe, Greater China, Japan, Rest of Asia"},
            {"query_id": "q_006", "question": "Which segment generated the most revenue for Apple in 2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "Americas"},
            {"query_id": "q_007", "question": "What percentage of Apple's revenue came from Americas in 2022?", "query_type": "arithmetic_probe", "expected_answer_marker": "percentage"},
            {"query_id": "q_008", "question": "How did Apple's China segment revenue change from 2021 to 2022?", "query_type": "multi_hop", "expected_answer_marker": "china growth"},
        ]
    },
    "apple_products_2020_2022": {
        "session_id": "apple_products_2020_2022",
        "company": "Apple",
        "period": "FY2020-2022",
        "queries": [
            {"query_id": "q_009", "question": "What were Apple's main product categories by revenue in 2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "iPhone, Mac, iPad, Wearables, Services"},
            {"query_id": "q_010", "question": "Which product line had the highest revenue for Apple in 2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "iPhone"},
            {"query_id": "q_011", "question": "Compare Services revenue between 2020 and 2022.", "query_type": "multi_hop", "expected_answer_marker": "services growth"},
            {"query_id": "q_012", "question": "What was the growth rate for Apple's Wearables category from 2021 to 2022?", "query_type": "arithmetic_probe", "expected_answer_marker": "wearables growth"},
        ]
    },
    "apple_profitability_trend": {
        "session_id": "apple_profitability_trend",
        "company": "Apple",
        "period": "FY2020-2022",
        "queries": [
            {"query_id": "q_013", "question": "What was Apple's operating income in 2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "operating income"},
            {"query_id": "q_014", "question": "Did Apple's net income increase from 2021 to 2022?", "query_type": "multi_hop", "expected_answer_marker": "income trend"},
            {"query_id": "q_015", "question": "Calculate Apple's gross profit margin for 2022.", "query_type": "arithmetic_probe", "expected_answer_marker": "margin calculation"},
            {"query_id": "q_016", "question": "How did Apple's operating expenses trend from 2020 to 2022?", "query_type": "multi_hop", "expected_answer_marker": "expense trend"},
        ]
    },
    "microsoft_fy2022_overview": {
        "session_id": "microsoft_fy2022_overview",
        "company": "Microsoft",
        "period": "FY2022",
        "queries": [
            {"query_id": "q_017", "question": "What was Microsoft's total revenue in fiscal year 2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "revenue"},
            {"query_id": "q_018", "question": "How many business segments does Microsoft operate?", "query_type": "tier_a_factoid", "expected_answer_marker": "segments"},
            {"query_id": "q_019", "question": "What was Microsoft's net income in FY2022?", "query_type": "terminology_mismatch_probe", "expected_answer_marker": "net income"},
            {"query_id": "q_020", "question": "Compare Microsoft's FY2022 revenue to FY2021.", "query_type": "multi_hop", "expected_answer_marker": "comparison"},
        ]
    },
    "microsoft_segments_2020_2022": {
        "session_id": "microsoft_segments_2020_2022",
        "company": "Microsoft",
        "period": "FY2020-2022",
        "queries": [
            {"query_id": "q_021", "question": "What are Microsoft's three main business segments as of 2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "Productivity and Business Processes, Intelligent Cloud, More Personal Computing"},
            {"query_id": "q_022", "question": "Which segment generated the most revenue for Microsoft in 2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "Intelligent Cloud"},
            {"query_id": "q_023", "question": "What percentage of Microsoft's revenue came from Intelligent Cloud in 2022?", "query_type": "arithmetic_probe", "expected_answer_marker": "cloud percentage"},
            {"query_id": "q_024", "question": "How did the Intelligent Cloud segment revenue grow from 2021 to 2022?", "query_type": "multi_hop", "expected_answer_marker": "cloud growth"},
        ]
    },
    "microsoft_cloud_revenue": {
        "session_id": "microsoft_cloud_revenue",
        "company": "Microsoft",
        "period": "FY2020-2022",
        "queries": [
            {"query_id": "q_025", "question": "What were the main revenue sources within Microsoft's Intelligent Cloud segment in 2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "Azure, Server products, Enterprise Services"},
            {"query_id": "q_026", "question": "Compare Azure revenue growth between 2021 and 2022.", "query_type": "multi_hop", "expected_answer_marker": "azure growth"},
            {"query_id": "q_027", "question": "Calculate the combined revenue from Productivity and Cloud segments for 2022.", "query_type": "arithmetic_probe", "expected_answer_marker": "segment sum"},
            {"query_id": "q_028", "question": "What was the trend in Microsoft's total employees from 2020 to 2022?", "query_type": "multi_hop", "expected_answer_marker": "headcount trend"},
        ]
    },
    "microsoft_profitability": {
        "session_id": "microsoft_profitability",
        "company": "Microsoft",
        "period": "FY2020-2022",
        "queries": [
            {"query_id": "q_029", "question": "What was Microsoft's operating income in 2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "operating income"},
            {"query_id": "q_030", "question": "Did Microsoft's gross margin improve from 2021 to 2022?", "query_type": "arithmetic_probe", "expected_answer_marker": "margin change"},
            {"query_id": "q_031", "question": "What percentage growth occurred in Microsoft's revenue from 2020 to 2022?", "query_type": "arithmetic_probe", "expected_answer_marker": "growth rate"},
            {"query_id": "q_032", "question": "How did Microsoft's capital expenditures change from 2021 to 2022?", "query_type": "multi_hop", "expected_answer_marker": "capex trend"},
        ]
    },
    "jnj_fy2022_overview": {
        "session_id": "jnj_fy2022_overview",
        "company": "Johnson & Johnson",
        "period": "FY2022",
        "queries": [
            {"query_id": "q_033", "question": "What was Johnson & Johnson's total revenue in 2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "revenue"},
            {"query_id": "q_034", "question": "How many business segments does Johnson & Johnson operate?", "query_type": "tier_a_factoid", "expected_answer_marker": "segments"},
            {"query_id": "q_035", "question": "What was Johnson & Johnson's net earnings in 2022?", "query_type": "terminology_mismatch_probe", "expected_answer_marker": "net earnings"},
            {"query_id": "q_036", "question": "Compare Johnson & Johnson's revenue in 2022 vs 2021.", "query_type": "multi_hop", "expected_answer_marker": "comparison"},
        ]
    },
    "jnj_segments_2020_2022": {
        "session_id": "jnj_segments_2020_2022",
        "company": "Johnson & Johnson",
        "period": "FY2020-2022",
        "queries": [
            {"query_id": "q_037", "question": "What are Johnson & Johnson's three main business segments?", "query_type": "tier_a_factoid", "expected_answer_marker": "Innovative Medicine, MedTech, Consumer Health"},
            {"query_id": "q_038", "question": "Which segment had the highest revenue for Johnson & Johnson in 2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "Innovative Medicine"},
            {"query_id": "q_039", "question": "What percentage of J&J's revenue came from Innovative Medicine in 2022?", "query_type": "arithmetic_probe", "expected_answer_marker": "percentage"},
            {"query_id": "q_040", "question": "How did the MedTech segment revenue change from 2021 to 2022?", "query_type": "multi_hop", "expected_answer_marker": "medtech growth"},
        ]
    },
    "jnj_healthcare_focus": {
        "session_id": "jnj_healthcare_focus",
        "company": "Johnson & Johnson",
        "period": "FY2020-2022",
        "queries": [
            {"query_id": "q_041", "question": "What were the main product categories within Johnson & Johnson's Innovative Medicine segment?", "query_type": "tier_a_factoid", "expected_answer_marker": "pharmaceuticals, diagnostics"},
            {"query_id": "q_042", "question": "Compare Johnson & Johnson's Consumer Health revenue between 2020 and 2022.", "query_type": "multi_hop", "expected_answer_marker": "consumer health trend"},
            {"query_id": "q_043", "question": "Calculate the combined revenue from Innovative Medicine and MedTech for 2022.", "query_type": "arithmetic_probe", "expected_answer_marker": "segment sum"},
            {"query_id": "q_044", "question": "What was Johnson & Johnson's operating profit in 2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "operating profit"},
        ]
    },
    "pepsico_fy2022_financials": {
        "session_id": "pepsico_fy2022_financials",
        "company": "PepsiCo",
        "period": "FY2022",
        "queries": [
            {"query_id": "q_045", "question": "What was PepsiCo's total net revenue in 2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "revenue"},
            {"query_id": "q_046", "question": "How many business segments does PepsiCo operate?", "query_type": "tier_a_factoid", "expected_answer_marker": "segments"},
            {"query_id": "q_047", "question": "What was PepsiCo's net income in 2022?", "query_type": "terminology_mismatch_probe", "expected_answer_marker": "net income"},
            {"query_id": "q_048", "question": "Compare PepsiCo's revenue in 2022 vs 2021.", "query_type": "multi_hop", "expected_answer_marker": "comparison"},
        ]
    },
    "pepsico_segments_2020_2022": {
        "session_id": "pepsico_segments_2020_2022",
        "company": "PepsiCo",
        "period": "FY2020-2022",
        "queries": [
            {"query_id": "q_049", "question": "What are PepsiCo's main business segments as of 2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "FLNA, QFNA, PBNA, Europe, AMESA, APAC"},
            {"query_id": "q_050", "question": "Which segment generated the most revenue for PepsiCo in 2022?", "query_type": "tier_a_factoid", "expected_answer_marker": "segment revenue"},
            {"query_id": "q_051", "question": "Calculate the revenue contribution percentage from Frito-Lay (FLNA) in 2022.", "query_type": "arithmetic_probe", "expected_answer_marker": "percentage"},
            {"query_id": "q_052", "question": "How did PepsiCo's geographic segment revenues change from 2021 to 2022?", "query_type": "multi_hop", "expected_answer_marker": "geographic trend"},
        ]
    },
}

def main():
    project_root = Path.home() / "PycharmProjects" / "finsession-rag"
    sessions_dir = project_root / "data" / "sessions"

    print(f"Setting up {project_root}...")

    # Create directories
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (project_root / "src").mkdir(parents=True, exist_ok=True)

    # Write session files
    print(f"\n✓ Creating 13 session files in {sessions_dir}")
    for session_id, session_data in SESSIONS.items():
        file_path = sessions_dir / f"{session_id}.json"
        with open(file_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        print(f"  ✓ {session_id}.json")

    print(f"\n✅ All session files created ({len(SESSIONS)} total, 52 queries)")
    print(f"\nNext: python -m src.run_experiment")

if __name__ == "__main__":
    main()
