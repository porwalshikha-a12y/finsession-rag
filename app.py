"""Streamlit front end for FinSession-RAG.

A live demo of the agentic RAG pipeline with session memory: ask questions,
watch the agent plan / retrieve / verify, see memory fill up and evict, and
compare experiment arms from results CSVs.

Run from the project root (venv active, index built):
    streamlit run app.py
"""
from __future__ import annotations
import warnings
from src.memory import SemanticMemory
from src.llm import LLM
from src.index import Embedder, make_store
from src.eviction import make_policy
from src.agent import RAGAgent
import yaml
import streamlit as st
import pandas as pd
import os
warnings.filterwarnings("ignore", message="No module named 'torchvision'")


st.set_page_config(page_title="FinSession-RAG", page_icon="📄", layout="wide")

ACCENT = "#4269D0"  # single-hue accent for all single-series charts

MODES = {
    "A0 — No memory (baseline)":        {"memory": False, "eviction": None,        "bounded": False},
    "A1 — Unbounded memory (ceiling)":  {"memory": True,  "eviction": None,        "bounded": False},
    "A2 — Bounded + LRU":               {"memory": True,  "eviction": "lru",        "bounded": True},
    "A3 — Bounded + LFU":               {"memory": True,  "eviction": "lfu",        "bounded": True},
    "A4 — Bounded + Redundancy":        {"memory": True,  "eviction": "redundancy", "bounded": True},
    "A5 — Bounded + Cost-aware (ours)": {"memory": True,  "eviction": "cost_aware", "bounded": True},
}


# ---------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading embedding model and vector store…")
def load_resources():
    cfg = yaml.safe_load(open("config.yaml"))
    embedder = Embedder(**cfg["embeddings"])
    index = make_store(cfg, embedder)
    return cfg, embedder, index


def build_agent(cfg, embedder, index, mode: dict, budget: int, tau: float,
                reuse_check: bool) -> RAGAgent:
    memory = None
    if mode["memory"]:
        policy = make_policy(mode["eviction"]) if mode["eviction"] else None
        memory = SemanticMemory(
            embedder=embedder,
            eviction_policy=policy,
            budget=budget if mode["bounded"] else None,
            similarity_threshold=tau,
        )
    return RAGAgent(
        llm=LLM(**cfg["llm"]),
        index=index,
        memory=memory,
        top_k=cfg["retrieval"]["top_k"],
        max_steps=cfg["agent"]["max_steps"],
        reuse_verification=reuse_check,
        security=cfg.get("security", {}).get("enabled", True),
    )



def _steps_frame(steps: list[dict]) -> "pd.DataFrame":
    """Render the sub-step trace as a table Arrow can serialise.

    Sub-steps carry nested values - provenance is a list of (doc_id, page)
    pairs and dropped_chunks a list of dicts - which pyarrow cannot type.
    Everything non-scalar is flattened to a readable string here.
    """
    COLUMNS = ["sub_q", "source", "verified", "provenance",
               "stored_q", "entry_id", "dropped_chunks", "note"]
    LABELS = {"sub_q": "sub-question", "source": "resolved by",
              "verified": "verified", "provenance": "evidence",
              "stored_q": "matched stored question", "entry_id": "entry",
              "dropped_chunks": "chunks dropped", "note": "note"}

    def prov(v):
        if not v:
            return ""
        out = []
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append(f"{item[0]} p.{item[1]}")
            elif isinstance(item, dict):
                out.append(f"{item.get('doc_id', '?')} p.{item.get('page', '?')}")
            else:
                out.append(str(item))
        return "; ".join(out)

    rows = []
    for st_ in steps:
        row = {}
        for k in COLUMNS:
            if k not in st_:
                continue
            v = st_[k]
            if k == "provenance":
                row[LABELS[k]] = prov(v)
            elif k == "dropped_chunks":
                row[LABELS[k]] = len(v) if isinstance(v, (list, tuple)) else (v or "")
            elif isinstance(v, (list, tuple, dict)):
                row[LABELS[k]] = str(v)
            elif v is None:
                row[LABELS[k]] = ""
            else:
                row[LABELS[k]] = v
        # anything unexpected still shows, as text
        for k, v in st_.items():
            if k not in COLUMNS:
                row[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.astype({c: "string" for c in df.columns
                      if df[c].dtype == "object"}) if not df.empty else df


# ---------------------------------------------------------------- sidebar
st.sidebar.title("FinSession-RAG")
st.sidebar.caption("Cost-aware semantic memory for agentic RAG")

if not os.environ.get("OPENAI_API_KEY"):
    key = st.sidebar.text_input("OpenAI API key", type="password",
                                help="Only stored in this process, never written to disk.")
    if key:
        os.environ["OPENAI_API_KEY"] = key

mode_name = st.sidebar.selectbox(
    "Memory configuration (arm)", list(MODES.keys()), index=5)
mode = MODES[mode_name]

budget = 20
if mode["bounded"]:
    budget = st.sidebar.slider("Memory budget (entries)", 5, 100, 20, 5)
tau = st.sidebar.slider("Similarity threshold τ", 0.50, 0.95, 0.80, 0.01,
                        help="A memory hit requires cosine similarity ≥ τ between the new sub-question and a stored one.")
reuse_check = st.sidebar.toggle("Reuse-verification gate (H4)", value=True,
                                help="Cheap LLM check that a memory hit really answers the new sub-question.")

if st.sidebar.button("Start new session", type="primary", use_container_width=True):
    for k in ("agent", "messages"):
        st.session_state.pop(k, None)
    st.rerun()

st.sidebar.caption("Changing settings takes effect when you start a new session. "
                   "A session = one continuous conversation with one memory.")


# ---------------------------------------------------------------- main
try:
    cfg, embedder, index = load_resources()
except Exception as e:  # noqa: BLE001
    st.error(
        "Couldn't load the vector store. Check Postgres is running and the "
        "corpus is built: `python -m scripts.build_index` (PDFs in "
        "`data/filings/`).\n\n"
        f"Details: {e}"
    )
    st.stop()

if "agent" not in st.session_state:
    st.session_state.agent = build_agent(
        cfg, embedder, index, mode, budget, tau, reuse_check)
    st.session_state.messages = []
agent: RAGAgent = st.session_state.agent

tab_chat, tab_results = st.tabs(["💬 Ask the agent", "📊 Results explorer"])


# ---------------------------------------------------------------- chat tab
with tab_chat:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Session")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                trace = msg.get("trace")
                if trace:
                    with st.expander(
                        f"Trace — {trace['tokens']} tokens, "
                        f"{trace['retrievals']} retrievals, "
                        f"{trace['memory_hits']} memory hits"
                    ):
                        if trace["sub_steps"]:
                            st.dataframe(_steps_frame(trace["sub_steps"]),
                                         use_container_width=True, hide_index=True)
                        else:
                            st.caption(
                                "Planner answered without sub-questions.")

        question = st.chat_input(
            "Ask about the filings, e.g. “What was Apple's total net sales in FY2022?”")
        if question:
            if not os.environ.get("OPENAI_API_KEY"):
                st.warning("Enter your OpenAI API key in the sidebar first.")
                st.stop()
            st.session_state.messages.append(
                {"role": "user", "content": question})
            with st.spinner("Planning → retrieving → verifying…"):
                trace = agent.answer(question)
            st.session_state.messages.append({
                "role": "assistant",
                "content": trace.answer,
                "trace": {
                    "tokens": trace.tokens_spent,
                    "retrievals": trace.retrievals,
                    "memory_hits": trace.memory_hits,
                    "sub_steps": trace.sub_steps,
                },
            })
            st.rerun()

    with right:
        st.subheader("Cost this session")
        ledger = agent.llm.ledger
        c1, c2, c3 = st.columns(3)
        c1.metric("Total tokens", f"{ledger.total_tokens:,}")
        c2.metric("LLM calls", ledger.calls)
        est = ledger.prompt_tokens * 0.15 / 1e6 + ledger.completion_tokens * 0.60 / 1e6
        c3.metric("≈ cost (USD)", f"${est:,.4f}",
                  help="gpt-4o-mini list prices; indicative only.")
        if ledger.by_role:
            role_df = (pd.DataFrame(
                [{"role": r, "tokens": v["prompt"] + v["completion"]}
                 for r, v in ledger.by_role.items()])
                .sort_values("tokens", ascending=False).set_index("role"))
            st.bar_chart(role_df, color=ACCENT, horizontal=True)

        st.subheader("Memory")
        if agent.memory is None:
            st.caption(
                "This arm runs without memory — every sub-question retrieves fresh.")
        else:
            stats = agent.memory.stats
            m1, m2, m3 = st.columns(3)
            m1.metric("Entries", f"{len(agent.memory.entries)}"
                      + (f" / {agent.memory.budget}" if agent.memory.budget else ""))
            m2.metric("Hit rate", f"{(stats.hits / stats.lookups * 100) if stats.lookups else 0:.0f}%",
                      help=f"{stats.hits} hits / {stats.lookups} lookups")
            m3.metric("Evictions", stats.evictions)
            if agent.memory.entries:
                mem_df = pd.DataFrame([{
                    "sub-question": e.sub_question,
                    "answer": e.sub_answer,
                    "cost (tok)": e.cost_tokens,
                    "hits": e.hits,
                } for e in agent.memory.entries])
                st.dataframe(mem_df, use_container_width=True,
                             hide_index=True, height=280)
            else:
                st.caption("Memory is empty — ask something.")


# ---------------------------------------------------------------- results tab
with tab_results:
    results_path = os.path.join(cfg["paths"]["results_dir"], "results_all.csv")
    if not os.path.exists(results_path):
        st.info("No experiment results yet. Run `python -m src.run_experiment` first; "
                "this tab then compares the arms from `results/results_all.csv`.")
    else:
        df = pd.read_csv(results_path)
        df["correct"] = df["correct"].map(
            {True: 1, False: 0, "True": 1, "False": 0})
        summary = (df.groupby("arm")
                     .agg(questions=("qid", "count"),
                          accuracy=("correct", "mean"),
                          mean_tokens_per_q=("tokens", "mean"),
                          total_retrievals=("retrievals", "sum"),
                          total_memory_hits=("memory_hits", "sum"))
                     .round({"accuracy": 3, "mean_tokens_per_q": 0}))
        st.subheader("Arms compared")
        st.dataframe(summary, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Mean tokens per question (lower is better)")
            st.bar_chart(summary[["mean_tokens_per_q"]],
                         color=ACCENT, horizontal=True)
        with c2:
            st.caption("Accuracy (higher is better)")
            st.bar_chart(summary[["accuracy"]], color=ACCENT, horizontal=True)

        st.caption("Cost per correct answer — the headline metric")
        cpc = (summary["mean_tokens_per_q"] * summary["questions"]
               / (summary["accuracy"] * summary["questions"]).clip(lower=1)).rename("tokens per correct answer")
        st.bar_chart(cpc.to_frame(), color=ACCENT, horizontal=True)

        with st.expander("Raw results"):
            st.dataframe(df, use_container_width=True, hide_index=True)
