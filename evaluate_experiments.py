"""
Error Categorization Script for FinSession-RAG Experiments
Run AFTER all 1,728 queries complete across the 6 experimental arms.

Usage:
    python error_categorizer.py --results experiments_results.json --output failure_breakdown.json --pretty
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse


class FailureCategorizer:
    """Categorizes experiment failures into semantic buckets for dissertation."""

    def __init__(self):
        # Failure categories based on dissertation taxonomy
        self.categories = {
            'temporal_ambiguity': {
                'patterns': [
                    r'(fiscal year|FY|year ended|period|quarter|Q\d)',
                    r'(\d{4})\s+(to|through|-)\s*(\d{4})',  # year ranges
                    r'(last\s+\d+\s+year|past\s+\d+\s+year)',
                ],
                'keywords': ['year', 'quarter', 'fiscal', 'period', 'dated', 'ended'],
                'description': 'Query refers to time-sensitive metrics without clear temporal anchoring'
            },
            'terminology_mismatch': {
                'patterns': [
                    r'(net\s+sales|gross\s+profit|operating\s+income|EBITDA)',
                    r'(revenue|turnover|income|earnings|profit)',
                ],
                'keywords': ['net', 'gross', 'operating', 'adjusted', 'core', 'normalized'],
                'description': 'Query uses term that maps to multiple financial synonyms in documents'
            },
            'entity_confusion': {
                'patterns': [
                    r'(subsidiary|division|segment|unit|brand)',
                    r'(parent company|holding|consolidated|equity)',
                ],
                'keywords': ['entity', 'company', 'subsidiary', 'segment', 'division'],
                'description': 'Query ambiguous about which company/subsidiary/segment'
            },
            'arithmetic_error': {
                'patterns': [
                    r'(sum|total|aggregate|combined|consolidated)',
                    r'(\d+%|\d+\.\d+%)',  # percentages
                    r'(increase|decrease|growth|decline)',
                ],
                'keywords': ['total', 'sum', 'aggregate', 'percent', 'growth', 'change'],
                'description': 'Calculated field or aggregate metric computation failed'
            },
            'retrieval_silence': {
                'patterns': [],  # Detected by absence of evidence
                'keywords': [],
                'description': 'No relevant passages retrieved from document corpus'
            }
        }

        # Financial synonyms for terminology matching
        self.financial_synonyms = {
            'revenue': ['net sales', 'turnover', 'income', 'total revenue', 'revenues'],
            'profit': ['net income', 'earnings', 'net profit', 'bottom line'],
            'margin': ['margin %', 'profit margin', 'gross margin'],
            'asset': ['total assets', 'assets', 'balance sheet assets'],
            'liability': ['total liabilities', 'liabilities'],
            'equity': ['stockholders equity', 'shareholders equity', 'total equity'],
        }

    def extract_years(self, text: str) -> List[int]:
        """Extract all 4-digit years from text."""
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        return [int(y) for y in years]

    def extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers (with commas/decimals) from text."""
        numbers = re.findall(r'\d[\d,]*(?:\.\d+)?', text)
        cleaned = []
        for n in numbers:
            try:
                cleaned.append(float(n.replace(',', '')))
            except ValueError:
                pass
        return cleaned

    def _has_temporal_ambiguity(self, query: str, evidence: str) -> bool:
        """Detect if failure stems from temporal reference ambiguity."""
        query_years = self.extract_years(query)
        evidence_years = self.extract_years(evidence)

        # If query mentions years but evidence has different years = ambiguity
        if query_years and evidence_years:
            if not any(y in evidence_years for y in query_years):
                return True

        # Check for temporal keywords without explicit dates
        temporal_kw = self.categories['temporal_ambiguity']['keywords']
        query_has_temporal = any(kw in query.lower() for kw in temporal_kw)
        evidence_has_dates = bool(self.extract_years(evidence))

        return query_has_temporal and not evidence_has_dates

    def _has_terminology_mismatch(self, query: str, evidence: str) -> bool:
        """Detect if failure stems from synonym/terminology mismatch."""
        query_lower = query.lower()

        # Check if query uses a term that has known synonyms
        for term, synonyms in self.financial_synonyms.items():
            if term in query_lower:
                # If query uses term but evidence uses a synonym (not the original)
                if any(syn in evidence.lower() for syn in synonyms):
                    if term not in evidence.lower():
                        return True

        return False

    def _has_entity_confusion(self, query: str, evidence: str) -> bool:
        """Detect if query is ambiguous about entity (company/subsidiary/segment)."""
        entity_kw = self.categories['entity_confusion']['keywords']
        query_has_entity = any(kw in query.lower() for kw in entity_kw)

        # If query mentions entity but evidence scope unclear
        evidence_lower = evidence.lower()
        has_scope_indication = any(word in evidence_lower
                                   for word in ['subsidiary', 'segment', 'consolidated', 'equity'])

        return query_has_entity and not has_scope_indication

    def _has_arithmetic_error(self, query: str, answer: str, evidence: str) -> bool:
        """Detect if failure is due to calculation/aggregation issues."""
        arith_kw = self.categories['arithmetic_error']['keywords']
        query_has_arith = any(kw in query.lower() for kw in arith_kw)

        if not query_has_arith:
            return False

        # Extract numbers from query, answer, evidence
        query_nums = self.extract_numbers(query)
        answer_nums = self.extract_numbers(answer) if answer else []
        evidence_nums = self.extract_numbers(evidence)

        # If answer has numbers not in evidence = calculation error
        if answer_nums and evidence_nums:
            answer_set = set(answer_nums)
            evidence_set = set(evidence_nums)
            if answer_set and not answer_set.issubset(evidence_set):
                return True

        return False

    def _has_retrieval_silence(self, evidence: str) -> bool:
        """Detect if failure is due to no/minimal passage retrieval."""
        return len(evidence.strip()) < 50  # Less than 50 chars of evidence

    def categorize_failure(self,
                           query: str,
                           answer: str,
                           evidence: str,
                           verdict: str) -> Tuple[str, float]:
        """
        Categorize a single failure.

        Returns: (category_name, confidence_score)
        """
        # Check each category in order of specificity
        checks = [
            ('retrieval_silence', lambda: self._has_retrieval_silence(evidence)),
            ('temporal_ambiguity', lambda: self._has_temporal_ambiguity(query, evidence)),
            ('terminology_mismatch',
             lambda: self._has_terminology_mismatch(query, evidence)),
            ('entity_confusion', lambda: self._has_entity_confusion(query, evidence)),
            ('arithmetic_error', lambda: self._has_arithmetic_error(
                query, answer, evidence)),
        ]

        for category, check_fn in checks:
            try:
                if check_fn():
                    return category, 1.0
            except Exception:
                pass

        # Default to retrieval_silence if no other match
        return 'retrieval_silence', 0.5

    def analyze_results(self, results_file: Path) -> Dict[str, Any]:
        """
        Analyze experiment results and categorize all failures.

        Expected results format:
        {
            "arm": "A0",
            "queries": [
                {
                    "query_id": "q_001",
                    "question": "What was revenue in 2023?",
                    "answer": "Revenue was $X million",
                    "evidence": "<<BEGIN DATA...>>...",
                    "verdict": "SUPPORTED" or "NOT_SUPPORTED" or "PARTIAL",
                    "error": "failure reason" (if verdict != SUPPORTED)
                },
                ...
            ]
        }
        """

        with open(results_file, 'r') as f:
            data = json.load(f)

        # Ensure data is a list (from multiple arms)
        if isinstance(data, dict) and 'queries' in data:
            data = [data]
        elif not isinstance(data, list):
            raise ValueError(
                "Expected JSON array of arm results or single arm object")

        # Aggregate across all arms
        failures_by_category = defaultdict(int)
        failures_by_arm = defaultdict(lambda: defaultdict(int))
        failure_examples = defaultdict(list)
        total_queries = 0
        total_failures = 0

        for arm_result in data:
            arm_name = arm_result.get('arm', 'unknown')
            queries = arm_result.get('queries', [])

            for q in queries:
                total_queries += 1
                verdict = q.get('verdict', 'UNKNOWN')

                # Count only failures (NOT_SUPPORTED or PARTIAL)
                if verdict in ['NOT_SUPPORTED', 'PARTIAL']:
                    total_failures += 1

                    query_text = q.get('question', '')
                    answer_text = q.get('answer', '')
                    evidence_text = q.get('evidence', '')

                    category, confidence = self.categorize_failure(
                        query_text, answer_text, evidence_text, verdict
                    )

                    failures_by_category[category] += 1
                    failures_by_arm[arm_name][category] += 1

                    # Store first 2 examples per category
                    if len(failure_examples[category]) < 2:
                        failure_examples[category].append({
                            'query_id': q.get('query_id', 'N/A'),
                            'question': query_text[:100],
                            'arm': arm_name,
                            'confidence': confidence
                        })

        # Calculate percentages
        breakdown_pct = {}
        if total_failures > 0:
            breakdown_pct = {
                cat: round(100 * count / total_failures, 1)
                for cat, count in failures_by_category.items()
            }

        return {
            'summary': {
                'timestamp': datetime.now().isoformat(),
                'total_queries': total_queries,
                'total_failures': total_failures,
                'failure_rate_pct': round(100 * total_failures / total_queries, 1) if total_queries > 0 else 0,
                'arms_analyzed': len(data),
            },
            'failure_breakdown': dict(failures_by_category),
            'failure_breakdown_pct': breakdown_pct,
            'by_arm': {arm: dict(cats) for arm, cats in failures_by_arm.items()},
            'examples': {cat: examples for cat, examples in failure_examples.items()},
            'category_definitions': {
                cat: info['description']
                for cat, info in self.categories.items()
            }
        }


def format_for_dissertation(analysis: Dict[str, Any]) -> str:
    """Format analysis results for inclusion in dissertation."""

    summary = analysis['summary']
    breakdown = analysis['failure_breakdown']
    breakdown_pct = analysis['failure_breakdown_pct']

    output = []
    output.append("=" * 70)
    output.append("FAILURE CATEGORIZATION ANALYSIS")
    output.append("=" * 70)
    output.append(f"\nAnalysis Date: {summary['timestamp']}")
    output.append(f"Total Queries Analyzed: {summary['total_queries']:,}")
    output.append(f"Total Failures Detected: {summary['total_failures']:,}")
    output.append(f"Overall Failure Rate: {summary['failure_rate_pct']:.1f}%")
    output.append(f"Arms Analyzed: {summary['arms_analyzed']}")

    output.append("\n" + "=" * 70)
    output.append("FAILURE BREAKDOWN BY CATEGORY")
    output.append("=" * 70 + "\n")

    # Table for dissertation
    output.append("| Category | Count | Percentage |")
    output.append("|----------|-------|------------|")

    sorted_cats = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
    for cat, count in sorted_cats:
        pct = breakdown_pct.get(cat, 0)
        output.append(f"| {cat:25} | {count:5} | {pct:6.1f}% |")

    output.append("\n" + "=" * 70)
    output.append("BREAKDOWN BY EXPERIMENTAL ARM")
    output.append("=" * 70 + "\n")

    for arm, cats in sorted(analysis['by_arm'].items()):
        output.append(f"\n{arm}:")
        for cat, count in sorted(cats.items(), key=lambda x: x[1], reverse=True):
            output.append(f"  {cat:30}: {count:3} failures")

    output.append("\n" + "=" * 70)
    output.append("CATEGORY DEFINITIONS (for dissertation)")
    output.append("=" * 70 + "\n")

    for cat, defn in analysis['category_definitions'].items():
        output.append(f"\n{cat.upper().replace('_', ' ')}:")
        output.append(f"  {defn}")

    output.append("\n" + "=" * 70)
    output.append("EXAMPLE FAILURES BY CATEGORY")
    output.append("=" * 70 + "\n")

    for cat, examples in analysis['examples'].items():
        output.append(f"\n{cat.upper().replace('_', ' ')} - Example Queries:")
        for ex in examples:
            output.append(
                f"  • Query ID {ex['query_id']} (Arm {ex['arm']}): {ex['question']}...")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description='Categorize FinSession-RAG experiment failures for dissertation'
    )
    parser.add_argument(
        '--results',
        type=Path,
        required=True,
        help='Path to experiment results JSON file (output from all 6 arms)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('failure_breakdown.json'),
        help='Path to save categorization results (default: failure_breakdown.json)'
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Also save human-readable report to failure_breakdown_report.txt'
    )

    args = parser.parse_args()

    # Run analysis
    print(f"Loading results from {args.results}...")
    categorizer = FailureCategorizer()
    analysis = categorizer.analyze_results(args.results)

    # Save JSON output
    with open(args.output, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Saved JSON analysis to {args.output}")

    # Save readable report
    if args.pretty:
        report_file = args.output.parent / 'failure_breakdown_report.txt'
        with open(report_file, 'w') as f:
            f.write(format_for_dissertation(analysis))
        print(f"✓ Saved readable report to {report_file}")

    # Print to console
    print("\n" + format_for_dissertation(analysis))


if __name__ == '__main__':
    main()
