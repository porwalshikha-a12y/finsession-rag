#!/usr/bin/env python3
"""
Extract and format experiment results for Section 5 of dissertation.
Run this AFTER the full experiment completes.

Output: results_summary.json + results_formatted.md
"""

import psycopg
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def connect_db():
    """Connect to PostgreSQL."""
    try:
        conn = psycopg.connect("postgresql:///finsession")
        return conn
    except Exception as e:
        print(f"ERROR: Could not connect to database: {e}")
        exit(1)

def query_accuracy_by_arm(conn):
    """5.1 Primary Findings: Accuracy, tokens, hit rate by arm."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
              arm,
              COUNT(*) as total_queries,
              SUM(CASE WHEN verdict = 'SUPPORTED' THEN 1 ELSE 0 END) as correct,
              SUM(CASE WHEN verdict = 'PARTIAL' THEN 1 ELSE 0 END) as partial,
              SUM(CASE WHEN verdict = 'NOT_SUPPORTED' THEN 1 ELSE 0 END) as not_supported,
              SUM(CASE WHEN verdict = 'UNKNOWN' THEN 1 ELSE 0 END) as unknown,
              ROUND(100.0 * SUM(CASE WHEN verdict = 'SUPPORTED' THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy_pct,
              ROUND(AVG(tokens_spent)::numeric, 0) as avg_tokens_per_q,
              ROUND(SUM(tokens_spent)::numeric, 0) as total_tokens,
              SUM(CASE WHEN memory_hits > 0 THEN 1 ELSE 0 END) as queries_with_hits,
              ROUND(100.0 * SUM(CASE WHEN memory_hits > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as hit_rate_pct,
              ROUND(AVG(memory_hits)::numeric, 2) as avg_memory_hits
            FROM results
            WHERE error = '' OR error IS NULL
            GROUP BY arm
            ORDER BY accuracy_pct DESC
        """)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in rows]

def query_tokens_per_correct(conn):
    """5.1: Cost efficiency metric."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
              arm,
              COUNT(*) as total_q,
              SUM(CASE WHEN verdict = 'SUPPORTED' THEN 1 ELSE 0 END) as correct,
              ROUND(SUM(tokens_spent)::numeric, 0) as total_tokens,
              ROUND(SUM(tokens_spent)::numeric / NULLIF(SUM(CASE WHEN verdict = 'SUPPORTED' THEN 1 ELSE 0 END), 0), 0) as tokens_per_correct
            FROM results
            WHERE error = '' OR error IS NULL
            GROUP BY arm
            ORDER BY tokens_per_correct ASC
        """)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in rows]

def query_memory_metrics(conn):
    """5.2: Memory hit rate and consistency."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
              arm,
              COUNT(*) as total_lookups,
              SUM(memory_hits) as total_hits,
              ROUND(100.0 * SUM(memory_hits) / NULLIF(COUNT(*), 0), 1) as hit_rate_pct,
              ROUND(AVG(memory_hits)::numeric, 2) as avg_hits_per_q,
              COUNT(DISTINCT CASE WHEN memory_hits > 0 THEN 1 END) as queries_with_hits
            FROM results
            WHERE error = '' OR error IS NULL
            GROUP BY arm
            ORDER BY hit_rate_pct DESC
        """)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in rows]

def query_grounding_fidelity(conn):
    """5.3: Grounding gate performance."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
              arm,
              COUNT(*) as total_queries,
              COUNT(*) - COUNT(CASE WHEN error LIKE '%grounding%' OR error LIKE '%KeyError%' THEN 1 END) as grounded_count,
              COUNT(CASE WHEN error LIKE '%grounding%' OR error LIKE '%KeyError%' THEN 1 END) as rejection_count,
              ROUND(100.0 * COUNT(CASE WHEN error LIKE '%grounding%' OR error LIKE '%KeyError%' THEN 1 END) / COUNT(*), 1) as rejection_rate_pct
            FROM results
            GROUP BY arm
            ORDER BY arm
        """)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in rows]

def query_robustness_by_company(conn):
    """5.4: Per-company accuracy breakdown."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
              s.session_id,
              s.company,
              r.arm,
              COUNT(*) as q_count,
              SUM(CASE WHEN r.verdict = 'SUPPORTED' THEN 1 ELSE 0 END) as correct,
              ROUND(100.0 * SUM(CASE WHEN r.verdict = 'SUPPORTED' THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy_pct,
              ROUND(AVG(r.tokens_spent)::numeric, 0) as avg_tokens,
              ROUND(AVG(r.memory_hits)::numeric, 2) as avg_hits
            FROM results r
            JOIN sessions s ON r.session_id = s.session_id
            WHERE r.error = '' OR r.error IS NULL
            GROUP BY s.session_id, s.company, r.arm
            ORDER BY s.company, r.arm, accuracy_pct DESC
        """)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in rows]

def query_error_breakdown(conn):
    """5.4: Failure analysis."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
              arm,
              verdict,
              COUNT(*) as count,
              ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY arm), 1) as pct_of_arm
            FROM results
            WHERE error = '' OR error IS NULL
            GROUP BY arm, verdict
            ORDER BY arm, count DESC
        """)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in rows]

def query_error_types(conn):
    """5.4: What kinds of errors occurred?"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
              CASE
                WHEN error LIKE '%grounding%' THEN 'Grounding failure'
                WHEN error LIKE '%KeyError%' THEN 'Metadata access error'
                WHEN error LIKE '%select_victim%' THEN 'Eviction policy error'
                WHEN error LIKE '%timeout%' THEN 'Timeout'
                WHEN error = '' THEN 'No error'
                ELSE 'Other: ' || SUBSTRING(error, 1, 50)
              END as error_type,
              COUNT(*) as count,
              array_agg(DISTINCT arm) as arms_affected
            FROM results
            GROUP BY error_type
            ORDER BY count DESC
        """)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in rows]

def format_markdown(results_dict):
    """Generate formatted Markdown for copy-paste into dissertation."""
    md = []
    md.append("# Section 5: RESULTS - Extracted Data\n")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 5.1 Accuracy by arm
    md.append("## 5.1 Primary Findings: Cost Reduction and Answer Correctness\n\n")
    md.append("### Table 5.1: Accuracy, Efficiency, and Memory Utilization by Arm\n\n")

    acc_data = results_dict['accuracy_by_arm']
    md.append("| Arm | Accuracy | Tokens/Q | Total Tokens | Hit Rate | Avg Hits |\n")
    md.append("|-----|----------|----------|--------------|----------|----------|\n")
    for row in acc_data:
        md.append(f"| {row['arm']} | {row['accuracy_pct']}% | {row['avg_tokens_per_q']} | {row['total_tokens']:,} | {row['hit_rate_pct']}% | {row['avg_memory_hits']} |\n")
    md.append("\n")

    # Cost per correct
    md.append("### Table 5.2: Cost Efficiency - Tokens per Correct Answer\n\n")
    cost_data = results_dict['tokens_per_correct']
    md.append("| Arm | Total Q | Correct | Total Tokens | Tokens/Correct |\n")
    md.append("|-----|---------|---------|--------------|----------------|\n")
    for row in cost_data:
        md.append(f"| {row['arm']} | {row['total_q']} | {row['correct']} | {row['total_tokens']:,} | {row['tokens_per_correct']} |\n")
    md.append("\n")

    # 5.2 Memory metrics
    md.append("## 5.2 Memory Hit Rate and Consistency\n\n")
    md.append("### Table 5.3: Memory Performance Metrics\n\n")
    mem_data = results_dict['memory_metrics']
    md.append("| Arm | Total Lookups | Total Hits | Hit Rate | Avg Hits/Q |\n")
    md.append("|-----|---------------|------------|----------|------------|\n")
    for row in mem_data:
        md.append(f"| {row['arm']} | {row['total_lookups']} | {row['total_hits']} | {row['hit_rate_pct']}% | {row['avg_hits_per_q']} |\n")
    md.append("\n")

    # 5.3 Grounding
    md.append("## 5.3 Grounding Fidelity: Did We Block Poison?\n\n")
    md.append("### Table 5.4: Grounding Gate Performance\n\n")
    ground_data = results_dict['grounding_fidelity']
    md.append("| Arm | Total Q | Grounded | Rejected | Rejection Rate |\n")
    md.append("|-----|---------|----------|----------|----------------|\n")
    for row in ground_data:
        md.append(f"| {row['arm']} | {row['total_queries']} | {row['grounded_count']} | {row['rejection_count']} | {row['rejection_rate_pct']}% |\n")
    md.append("\n")

    # 5.4 Robustness
    md.append("## 5.4 Robustness Across Companies and Query Types\n\n")
    md.append("### Table 5.5: Accuracy by Company and Arm\n\n")
    rob_data = results_dict['robustness_by_company']

    # Pivot: rows = companies, cols = arms
    companies = sorted(set([r['company'] for r in rob_data]))
    arms = sorted(set([r['arm'] for r in rob_data]))

    md.append("| Company | " + " | ".join(arms) + " |\n")
    md.append("|---------|" + "|".join(["---"] * len(arms)) + "|\n")

    for company in companies:
        row_data = [company]
        for arm in arms:
            match = [r for r in rob_data if r['company'] == company and r['arm'] == arm]
            if match:
                row_data.append(f"{match[0]['accuracy_pct']}%")
            else:
                row_data.append("—")
        md.append("| " + " | ".join(row_data) + " |\n")
    md.append("\n")

    # Error breakdown
    md.append("### Table 5.6: Verdict Distribution by Arm\n\n")
    error_data = results_dict['error_breakdown']
    md.append("| Arm | Verdict | Count | % of Arm |\n")
    md.append("|-----|---------|-------|----------|\n")
    for row in error_data:
        md.append(f"| {row['arm']} | {row['verdict']} | {row['count']} | {row['pct_of_arm']}% |\n")
    md.append("\n")

    # Error types
    md.append("### Table 5.7: Error Classification\n\n")
    err_types = results_dict['error_types']
    md.append("| Error Type | Count | Arms Affected |\n")
    md.append("|-----------|-------|---------------|\n")
    for row in err_types:
        arms_str = ", ".join(row['arms_affected']) if row['arms_affected'] else "—"
        md.append(f"| {row['error_type']} | {row['count']} | {arms_str} |\n")
    md.append("\n")

    return "\n".join(md)

def main():
    print("Extracting experiment results...")
    conn = connect_db()

    results = {
        'accuracy_by_arm': query_accuracy_by_arm(conn),
        'tokens_per_correct': query_tokens_per_correct(conn),
        'memory_metrics': query_memory_metrics(conn),
        'grounding_fidelity': query_grounding_fidelity(conn),
        'robustness_by_company': query_robustness_by_company(conn),
        'error_breakdown': query_error_breakdown(conn),
        'error_types': query_error_types(conn),
    }

    conn.close()

    # Save as JSON
    output_json = Path("results_summary.json")
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ Saved JSON: {output_json}")

    # Save as Markdown
    output_md = Path("results_formatted.md")
    md_content = format_markdown(results)
    with open(output_md, 'w') as f:
        f.write(md_content)
    print(f"✅ Saved Markdown: {output_md}")

    # Print summary to console
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    for arm_data in results['accuracy_by_arm']:
        print(f"{arm_data['arm']:20s} | Accuracy: {arm_data['accuracy_pct']:5.1f}% | Tokens/Q: {arm_data['avg_tokens_per_q']:4.0f} | Hit Rate: {arm_data['hit_rate_pct']:5.1f}%")

    print("\n✅ Results ready for Section 5. Copy tables from results_formatted.md into your dissertation.")

if __name__ == "__main__":
    main()
