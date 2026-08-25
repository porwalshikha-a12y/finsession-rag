"""Session dataset format + loader.

A session file is JSON:
{
  "session_id": "AAPL_analyst_1",
  "company": "AAPL",
  "queries": [
    {"qid": "q1", "question": "...", "gold_answer": "...",
     "type": "factoid|multi_hop|cross_doc",
     "paraphrase_of": null | "q1"}   # paraphrase probes for consistency (RQ3)
  ]
}
Put files in data/sessions/*.json.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_sessions(sessions_dir: str | Path) -> list[dict]:
    sessions = []
    for path in sorted(Path(sessions_dir).glob("*.json")):
        with path.open() as f:
            s = json.load(f)
        assert "session_id" in s and "queries" in s, f"bad session file: {path}"
        sessions.append(s)
    return sessions


# Smoke-test session matching the 2020-2022 corpus.
# VERIFY every gold_answer against your own PDFs before trusting a run —
# these are starting values, and your real sessions must be hand-checked.
EXAMPLE_SESSION = {
    "session_id": "EXAMPLE_APPLE_1",
    "company": "APPLE",
    "queries": [
        {"qid": "q1", "question": "What was Apple's total net sales in fiscal 2022?",
         "gold_answer": "$394.3 billion ($394,328 million)",
         "type": "factoid", "paraphrase_of": None},
        {"qid": "q2", "question": "Which Apple product category had the highest net sales in fiscal 2022?",
         "gold_answer": "iPhone (about $205.5 billion)",
         "type": "factoid", "paraphrase_of": None},
        {"qid": "q3", "question": "How did Apple's Services net sales change from fiscal 2021 to fiscal 2022?",
         "gold_answer": "Increased from about $68.4 billion to about $78.1 billion (roughly +14%)",
         "type": "multi_hop", "paraphrase_of": None},
        {"qid": "q4", "question": "What were Apple's overall net sales for FY2022?",
         "gold_answer": "$394.3 billion ($394,328 million)",
         "type": "factoid", "paraphrase_of": "q1"},
    ],
}

if __name__ == "__main__":
    out = Path("data/sessions/example_session.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(EXAMPLE_SESSION, indent=2))
    print(f"wrote {out}")
