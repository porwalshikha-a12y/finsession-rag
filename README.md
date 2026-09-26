# FinSession-RAG

Code for the MSc dissertation **"Cost-Aware Semantic Memory for Agentic RAG"**.

An agentic RAG pipeline over SEC 10-K filings with a bounded, session-scoped
memory of verified evidence. Eleven experimental arms (A0-A10) compare
eviction policies (LRU, LFU, semantic-redundancy, cost-aware), the
reuse-verification gate, and dense-vs-hybrid retrieval on token cost,
correctness, and answer consistency.

## Architecture

```
question --> Planner --> sub-question --> Memory lookup --similarity hit--> Reuse gate (LLM) --yes--> reuse fact
                 ^                              |miss / gate-reject             |no
                 |                              v                               v
                 |                    Retrieve (pgvector dense, or          fresh retrieval
                 |                     dense+BM25 via RRF for arm A10)
                 |                              |
                 |                              v
                 |                    Security: injection filter -> spotlight
                 |                              |
                 |                              v
                 +---- verified fact <-- Verifier --grounded?--> write to Memory
                                                                  (evict if over budget)
question + verified facts --> Synthesizer --> answer
```

Components (all in `src/`, plus two thin front ends at the repo root):

| File | What it does |
|---|---|
| `ingest.py` | PDF -> page text via pdfplumber (parser choice per arXiv:2604.12047) |
| `chunking.py` | Sentence chunking, 512-token cap, 128-token (25%) overlap, cl100k_base |
| `index.py` | Local bge-small-en-v1.5 embeddings + three interchangeable stores: `PgVectorIndex` (default, PostgreSQL + pgvector), `HybridPgVectorIndex` (pgvector + BM25 via RRF, arm A10), `FaissIndex` (offline fallback) |
| `hybrid.py` | BM25 (rank_bm25) + financial-term query expansion + Reciprocal Rank Fusion for hybrid retrieval |
| `llm.py` | OpenAI-compatible client with a per-role, prompt/completion token ledger |
| `security.py` | Injection-pattern filter, spotlighting, and a grounding check that gates memory writes |
| `agent.py` | The plan -> memory-lookup/retrieve -> reuse-gate/verify -> synthesize loop |
| `memory.py` | Bounded semantic memory of verified evidence units |
| `eviction.py` | Eviction policies: `none`, `lru`, `lfu`, `redundancy`, `cost_aware` (the dissertation's contribution) |
| `sessions.py` | Session dataset format + loader |
| `judge.py` | LLM-as-judge correctness scoring + paraphrase-consistency scoring |
| `run_experiment.py` | Runs all arms x sessions, writes results (PostgreSQL/CSV) + a per-arm trace file |
| `app.py` (root) | Streamlit demo - ask questions live, watch the agent plan/retrieve/verify, inspect memory |
| `api.py` (root) | FastAPI JSON backend for the same agent, with a static HTML/JS front end |

## Setup (once)

```bash
git clone https://github.com/porwalshikha-a12y/finsession-rag.git
cd finsession-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # fill in OPENAI_API_KEY and PGVECTOR_DSN
```

Needs a running PostgreSQL instance with the `pgvector` extension enabled -
see `SETUP_POSTGRES.md`.

## Workflow

**1. Filings.** Twelve 10-K PDFs across four companies (Apple, Microsoft,
Johnson & Johnson, PepsiCo), fiscal years 2020-2022, in `data/filings/`.
Name them like `AAPL_2022_10K.pdf`.

**2. Build the index** (one-time):

```bash
python -m scripts.build_index
```

Chunks, embeds, and loads every filing into the `chunks` table in Postgres.

**3. Sessions** already live in `data/sessions/` - thirteen analyst sessions,
sixty-five questions total, gold answers extracted from the filings
themselves. To regenerate the example format: `python -m src.sessions`.

**4. Smoke-test one arm on one session** before spending money on the full grid:

```bash
python -m src.run_experiment --arm A0_no_memory
```

**5. Run the full grid** (all eleven arms):

```bash
python -m src.run_experiment
```

Results land in PostgreSQL and in `results/results_<arm>.csv` /
`results_all.csv`, plus `results/traces_<arm>.jsonl` per arm for failure
analysis.

**6. Try it live:**

```bash
streamlit run app.py
# or
uvicorn api:app --reload
```

## Tests (no API key needed)

```bash
pytest -q
```

## Knobs that matter (`config.yaml`)

- `memory.similarity_threshold` (tau = 0.80): the candidate-match threshold;
  the reuse gate verifies every candidate before it is actually reused.
- `memory.budgets`: `tight: 3`, `moderate: 5` - sized to the measured
  working set (median 4 verified facts per session).
- `memory.cost_aware.half_life`: 10 steps - decay rate for the cost-aware
  policy's retention score.
- `arms`: eleven arms (A0-A10). See the comments in `config.yaml` for what
  each isolates, including the A6/A10 note (A6 is an unintended exact
  replication of A5 and is kept as a variance control; A10 is the real
  hybrid-retrieval arm).

## Cost expectations

gpt-4o-mini across eleven arms x thirteen sessions x sixty-five questions is
roughly US$10-40 including judging. Embeddings and retrieval are local and
free.
