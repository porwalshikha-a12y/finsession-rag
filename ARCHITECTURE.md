# FinSession-RAG — How Everything Works (Plain-English Guide)

Keep this file in the project root and commit it. It explains every file and
every function in simple words. It doubles as viva preparation.

---

## The big picture

The system has two phases, like a library:

**Offline phase (done once):** take the 10-K PDFs, extract their text, cut the
text into small overlapping pieces ("chunks"), turn each chunk into a list of
numbers that captures its meaning (an "embedding"), and store all of them in a
searchable index. This is "building the library and its catalogue."

**Online phase (every question):** an agent receives a question, breaks it into
smaller sub-questions, and for each one either (a) finds the answer in its own
memory of facts it already verified earlier in the session, or (b) searches the
index, checks that the retrieved text really answers the sub-question, and
saves the verified fact into memory. When memory is full, an eviction policy
decides which old fact to throw away. Finally it composes the overall answer
from the verified facts.

Around this sits the **measurement layer**: every LLM call's token cost is
recorded, every answer is graded against a gold answer, and everything is
written to CSV files for analysis.

## The journey of one question

1. `run_experiment.py` reads a question from a session file and hands it to the agent.
2. The agent's **planner** (an LLM prompt) says: "to answer this, I first need
   to know X" — a sub-question.
3. The agent checks **memory**: is there a stored, verified fact similar enough
   to X? If yes (and a cheap re-check agrees it really matches), reuse it. Free.
4. If no: **retrieve** the 4 most relevant chunks from the FAISS index, then the
   **verifier** (another LLM prompt) reads them and either extracts the answer
   or says "not supported."
5. A verified fact is **written to memory**, stamped with how many tokens it
   cost to produce. If memory is over budget, the **eviction policy** picks a
   victim to delete.
6. Loop back to the planner ("what do I still need?") until it says DONE, then
   the **synthesizer** (a third LLM prompt) writes the final answer from the
   gathered facts.
7. The trace (tokens spent, retrievals, memory hits, sub-steps) is logged, and
   the **judge** grades the answer against the gold answer.

---

## File-by-file

### `config.yaml` — the control panel
Every knob in one place: which LLM, which embedding model, chunk size (512
tokens, 25% overlap), top-k (4 chunks per retrieval), the memory similarity
threshold τ (0.80), the memory budgets (tight=20 entries, moderate=60), and
the list of experiment arms A0–A5. You change experiments by editing this
file, not the code.

### `src/ingest.py` — PDF → text
- `parse_pdf(pdf_path)`: opens one PDF with pdfplumber and returns a list of
  `{doc_id, page, text}` — one entry per page. It also extracts tables
  separately and appends them as tab-separated rows, because plain text
  extraction often scrambles tables. `doc_id` comes from the filename, which
  is why readable filenames matter.
- `parse_to_jsonl(pdf_path)`: same, but saves the result to a `.pages.jsonl`
  file (one JSON object per line) so you can inspect what the parser saw.

### `src/chunking.py` — text → chunks
- `split_sentences(text)`: cuts text into sentences using a simple punctuation
  rule (a period/!/? followed by a capital letter starts a new sentence).
- `chunk_pages(pages, ...)`: the main function. Walks through each page's
  sentences and greedily packs them into chunks of at most 512 tokens. When a
  chunk is full, the next chunk starts with the last ~128 tokens of the
  previous one — that 25% overlap means a fact sitting on a chunk boundary
  still appears whole in at least one chunk. Table rows that came out as one
  giant "sentence" get hard-split by tokens. Returns
  `{chunk_id, doc_id, page, text, n_tokens}` per chunk.
- `get_encoding(name)`: returns the tiktoken tokenizer used to count tokens;
  falls back to counting words if tiktoken can't download its data file.

Why sentence chunking and 25% overlap? Because the benchmark paper
(arXiv:2604.12047) found this is the near-optimal low-cost configuration —
we cite them instead of re-running their experiments.

### `src/index.py` — chunks → searchable index
- `Embedder`: wraps a local sentence-transformers model (bge-small). Its
  `encode(texts)` turns text into normalized vectors — no API, no cost.
  The class caches the loaded model so it's only loaded once.
- `VectorIndex.build(chunks)`: embeds every chunk and puts the vectors in a
  FAISS index (inner product on normalized vectors = cosine similarity).
- `VectorIndex.search(query, top_k)`: embeds the query and returns the top-k
  most similar chunks, each with its similarity score.
- `save(dir)` / `load(dir)`: write/read the index and the chunk texts to disk
  so you build once and reuse forever.

### `src/llm.py` — the only paid component, with a receipt for every call
- `CostLedger`: a running total of tokens. `add(role, prompt, completion)`
  records each call under a label (planner / verifier / synthesizer / judge),
  so you can later say exactly where the money went. `snapshot()` returns it
  as a dict.
- `LLM.chat(system, user, role)`: sends one chat request to gpt-4o-mini (or
  any OpenAI-compatible endpoint via `OPENAI_BASE_URL`), records the token
  usage in the ledger, returns the text. Temperature 0 everywhere so runs are
  as repeatable as possible.

### `src/memory.py` — the heart of the dissertation
- `MemoryEntry`: one remembered fact. Stores the sub-question, its embedding,
  the evidence text, the verified sub-answer, **how many tokens it cost to
  produce** (`cost_tokens`), when it was created and last used, and how many
  times it was reused (`hits`). The cost and usage fields exist precisely so
  eviction policies can be smart.
- `SemanticMemory.lookup(sub_question)`: embeds the incoming sub-question,
  compares it to every stored entry (cosine similarity), and returns the best
  entry if similarity ≥ τ (0.80), else None. A hit updates the entry's
  recency and hit count.
- `SemanticMemory.write(...)`: stores a new verified fact. If the number of
  entries now exceeds the budget, calls `_evict()` until it fits.
- `_evict()`: asks the eviction policy to pick a victim index and deletes it.
- `reset()`: wipes memory between sessions (memory is session-scoped).
- `MemoryStats`: counters (lookups, hits, writes, evictions) — these become
  the hit-rate numbers in your results.

### `src/eviction.py` — the four "who to forget" strategies
All four implement one method, `select_victim(entries, clock)` → index of the
entry to delete. Because they share an interface, experiment arms differ ONLY
in which policy is plugged in — a clean controlled comparison.
- `LRU`: forget the entry that hasn't been used for the longest time.
- `LFU`: forget the entry used the fewest times (ties broken by age).
- `SemanticRedundancy`: compute similarity between all pairs of entries;
  forget the entry most similar to another remaining one — its information is
  nearly duplicated anyway.
- `CostAware` (your contribution, adapted from SmartEvict): score each entry
  as `cost_tokens × (1 + hits) × recency_decay` and forget the lowest score.
  Intuition: a fact that was expensive to look up and gets reused is precious;
  a cheap fact nobody reuses is disposable. `half_life` controls how fast an
  unused entry's score decays.
- `make_policy(name)`: tiny factory that maps the config string ("lru", ...)
  to the right class.

### `src/agent.py` — the agent loop
The four prompts at the top ARE the agent's behavior — read them:
- `PLANNER_SYS`: "either give me the next sub-question you still need
  (NEXT: ...), or say DONE."
- `VERIFIER_SYS`: "answer ONLY from the passages; first line SUPPORTED or
  NOT_SUPPORTED." This is the quality gate — nothing unverified enters memory
  or the final answer.
- `REUSE_CHECK_SYS`: "does this stored fact really answer this new
  sub-question? YES/NO." A cheap guard against near-miss reuse (e.g. stored
  fact is 2022 revenue, question asks 2023) — this is hypothesis H4.
- `SYNTH_SYS`: "answer using ONLY the verified facts."

Functions:
- `QueryTrace`: a record of what happened for one question — sub-steps taken,
  memory hits, retrievals, tokens before/after (so `tokens_spent` is exact).
- `RAGAgent.answer(question)`: the main loop. Ask planner → get sub-question →
  `_resolve_subquestion` → repeat until DONE or max_steps (6) → synthesize.
- `RAGAgent._resolve_subquestion(sub_q, trace)`: implements steps 3–5 of the
  journey above: memory lookup first; on miss, retrieve top-4 chunks, run the
  verifier, and if SUPPORTED, write the fact to memory with its measured cost.
- `RAGAgent._reuse_ok(sub_q, entry)`: runs the reuse check prompt on a memory
  hit; can be switched off in config to measure how much it matters.

### `src/sessions.py` — the question sequences
Defines the session file format (JSON): a session has an id, a company, and an
ordered list of queries, each with a gold answer, a type tag
(factoid / multi_hop / cross_doc), and optionally `paraphrase_of` pointing at
an earlier question — those paraphrase probes measure consistency (RQ3).
- `load_sessions(dir)`: reads all `data/sessions/*.json` files.
- Running `python -m src.sessions` writes an example session so you can see
  the format.

### `src/judge.py` — grading
- `judge_answer(llm, question, gold, candidate)`: asks the LLM "CORRECT or
  INCORRECT?" comparing the agent's answer to the gold answer. At your scale
  you also manually check every judgment — the judge is a first pass.
- `judge_consistency(llm, answer_a, answer_b)`: for paraphrase probes — do the
  two answers say the same thing? Feeds the consistency metric (RQ3).

### `src/run_experiment.py` — the conductor
- `build_memory(arm, cfg, embedder)`: reads an arm's config line and builds
  the right memory: none (A0), unbounded (A1), or bounded with the specified
  policy and budget (A2–A5).
- `run_arm(arm, ...)`: for each session: fresh LLM ledger + fresh memory →
  answer every query in order → record tokens, retrievals, memory hits →
  then grade all answers with the judge and check paraphrase consistency.
  Writes a full trace file per arm (`results/traces_<arm>.jsonl`) for failure
  analysis (RQ4).
- `main()`: loads config, index, and sessions; runs all arms (or one, with
  `--arm NAME`); saves `results/results_<arm>.csv` per arm and a combined
  `results/results_all.csv`. One row = one question in one arm, with
  everything needed for the analysis.

### `scripts/build_index.py` — the offline phase, end to end
Glues ingest → chunking → index for every PDF in `data/filings/` and saves to
`data/index/`. Run once after changing the corpus.

### `tests/test_offline.py` — the safety net
11 fast tests with a fake embedder (no API, no internet): the chunker respects
the size cap and produces overlap; memory hits on a matching question and
misses on an unrelated one; a full memory evicts; and each policy picks the
correct victim in a hand-built scenario (e.g. CostAware keeps the expensive,
reused entry and dumps the cheap unused one). Run `python -m pytest -q` after
every code change.

---

## How the arms map to research questions

| Arm | What it is | Answers |
|---|---|---|
| A0 no memory | plain agentic RAG | the baseline for RQ1 |
| A1 unbounded | remember everything | the ceiling: how good could memory be if storage were free |
| A2–A4 | bounded + LRU / LFU / redundancy | the heuristic competitors for RQ2 |
| A5 cost-aware | bounded + SmartEvict-style policy | your contribution, RQ2 |

RQ1 = A0 vs the rest (cost & accuracy). RQ2 = A2–A5 against each other at the
same budget. RQ3 = the paraphrase-probe consistency column, A0 vs memory arms.
RQ4 = read the trace files for stale/near-miss reuse failures.
