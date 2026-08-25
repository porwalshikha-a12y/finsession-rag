# FinSession-RAG

Code for the MSc dissertation **"Cost-Aware Semantic Memory for Agentic RAG"**.

An agentic RAG pipeline over SEC 10-K filings with a bounded, session-scoped
memory of verified evidence, comparing eviction policies (LRU, LFU,
semantic-redundancy, cost-aware) on cost, accuracy, and answer consistency.

## Architecture

```
question ──> Planner ──> sub-question ──> Memory lookup ──hit──> reuse fact
                 ▲                            │miss
                 │                            ▼
                 │                    Vector retrieval (FAISS)
                 │                            │
                 │                            ▼
                 └──── verified fact <── Verifier ──> write to Memory
                                                       (evict if over budget)
question + verified facts ──> Synthesizer ──> answer
```

Components (all in `src/`):

| File | What it does |
|---|---|
| `ingest.py` | PDF -> page text via pdfplumber (parser choice per arXiv:2604.12047) |
| `chunking.py` | Sentence chunking, 512 tokens, 25% overlap (per the same paper) |
| `index.py` | Local sentence-transformers embeddings + FAISS (free, no API) |
| `llm.py` | OpenAI-compatible client with per-role token ledger |
| `agent.py` | The plan -> retrieve -> verify -> synthesize loop |
| `memory.py` | Bounded semantic memory of verified evidence units |
| `eviction.py` | The four eviction policies (cost-aware = your contribution) |
| `sessions.py` | Session dataset format + loader |
| `judge.py` | LLM-as-judge scoring + consistency probes |
| `run_experiment.py` | Runs arms x sessions, writes results CSVs + traces |

## Setup (once)

```bash
cd finsession-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # only paid component (gpt-4o-mini)
```

## Workflow

**1. Get filings.** Download 8–12 10-K PDFs (4 companies × 2–3 years) from the
FinanceBench repo (https://github.com/patronus-ai/financebench, `pdfs/` folder)
into `data/filings/`. Name them like `AAPL_2023_10K.pdf`.

**2. Build the index** (one-time, ~minutes on CPU):

```bash
python -m scripts.build_index
```

**3. Create sessions.** Write session JSON files into `data/sessions/`
(format in `src/sessions.py`; run `python -m src.sessions` to generate an
example). Reuse FinanceBench questions for your companies as Tier-A factoids,
author multi-hop/cross-year follow-ups, and add 2–3 paraphrase probes per
session for the consistency metric.

**4. Smoke-test one arm on one session** before spending money on the grid:

```bash
python -m src.run_experiment --arm A0_no_memory
```

**5. Run the full grid:**

```bash
python -m src.run_experiment
```

Results land in `results/results_all.csv` (one row per query per arm:
tokens, retrievals, memory hits, judge verdict, consistency) plus
`traces_*.jsonl` for failure analysis (RQ4).

## Tests (no API key needed)

```bash
pytest -q
```

Covers chunking, memory hit/miss, budget-triggered eviction, and all four
eviction policies with a fake embedder — run these before anything else.

## Knobs that matter

- `memory.similarity_threshold` (τ): tune on 2 held-out dev sessions; report sensitivity.
- `memory.budgets`: tight=20 / moderate=60 entries; run A2–A5 at both for RQ2.
- `arms`: edit `config.yaml` to add/remove arms (e.g. a moderate-budget copy of each policy arm).

## Cost expectations

gpt-4o-mini at ~150 queries × 6 arms ≈ US$10–40 including judging. The
embeddings and retrieval are local and free. Set a spend limit on your
OpenAI account before the full grid.
