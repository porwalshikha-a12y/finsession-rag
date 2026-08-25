# Setting up PostgreSQL + pgvector on macOS

Setup guide for FinSession-RAG. Written for Apple Silicon (Homebrew at
`/opt/homebrew`). Each step has a **check** — do not move on until it passes.

---

## Step 0 — Confirm Homebrew works

```bash
brew --version
```

**Check:** prints a version (e.g. `Homebrew 4.x.x`).
If "command not found", install Homebrew from https://brew.sh first.

---

## Step 1 — Install PostgreSQL and pgvector

```bash
brew install postgresql@17 pgvector
```

Takes a few minutes. If `postgresql@17` is not found, use `brew install postgresql`
(the latest version) and substitute that name everywhere below.

**Check:**

```bash
brew list | grep -E "postgresql|pgvector"
```
prints both packages.

---

## Step 2 — Start the database server

```bash
brew services start postgresql@17
```

This starts Postgres now AND makes it start automatically on every reboot —
so this is a one-time action, not something to repeat each session.

**Check:**

```bash
brew services list
```
shows `postgresql@17` with status `started`.

---

## Step 3 — Put `psql` on your PATH

Versioned Homebrew formulae are "keg-only", meaning their commands are not on
your PATH by default.

```bash
echo 'export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

If your PyCharm terminal runs bash rather than zsh, also run:

```bash
echo 'export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"' >> ~/.bash_profile
```

**Check:**

```bash
which psql && psql --version
```
prints a path and a version. If not found, close and reopen the terminal first.

---

## Step 4 — Create the project database

```bash
createdb finsession
```

**Check:**

```bash
psql -d finsession -c "SELECT current_database();"
```
prints `finsession`.

*If you get "role does not exist":* run
`createuser -s $(whoami)` then retry `createdb finsession`.

---

## Step 5 — Enable the pgvector extension

The extension is installed on the machine (Step 1) but must be enabled
**inside the database**:

```bash
psql -d finsession -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -d finsession -c "SELECT extname, extversion FROM pg_extension WHERE extname='vector';"
```

**Check:** the second command prints a row like `vector | 0.8.x`.
This is the moment the database is ready.

---

## Step 6 — Install the Python drivers

In the project folder, with the venv active (`(.venv)` in the prompt):

```bash
pip install -r requirements.txt
```

**Check:**

```bash
python -c "import psycopg, pgvector; print('drivers ok')"
```

---

## Step 7 — Confirm the project can talk to the database

```bash
python -m pytest -q
```

**Check:** `27 passed`.
If it says `18 passed, 9 skipped`, the tests could not reach Postgres — the
skip reason names the DSN it tried. Go back to Step 2 and Step 5.

---

## Step 8 — Load the corpus

With your 12 PDFs in `data/filings/`:

```bash
python -m scripts.build_index
```

Expect: parsing progress bars, a one-time embedding-model download (~130 MB),
then `stored N chunks in Postgres (table 'chunks')`.

**Check:**

```bash
psql -d finsession -c "SELECT count(*), count(DISTINCT doc_id) FROM chunks;"
psql -d finsession -c "SELECT DISTINCT company, year FROM chunks ORDER BY company, year;"
```
The first prints your chunk and document counts; the second should list your
4 companies × 3 years — proof the metadata parsed correctly from the filenames.

---

## Everyday use

Nothing to start manually — Postgres runs in the background from Step 2 onward.

| Task | Command |
|---|---|
| Check the server is running | `brew services list` |
| Restart the server | `brew services restart postgresql@17` |
| Stop it (frees a little RAM) | `brew services stop postgresql@17` |
| Open a SQL shell | `psql -d finsession` (`\q` to quit, `\dt` to list tables) |
| Rebuild the corpus from scratch | `python -m scripts.build_index` |

---

## Troubleshooting

**"connection refused" / "could not connect to server"**
Postgres isn't running: `brew services restart postgresql@17`, wait 5 seconds, retry.

**"database finsession does not exist"**
Step 4 didn't complete: `createdb finsession`, then redo Step 5.

**"extension vector is not available"**
pgvector was installed for a *different* Postgres version than the one running.
Check `psql --version` matches the formula you started, then
`brew reinstall pgvector`.

**"role <yourname> does not exist"**
`createuser -s $(whoami)`

**Tests skip with "no Postgres at postgresql:///finsession"**
The DSN in `config.yaml` (`vector_store.dsn`) must match your database name.
Override per-run with `PGVECTOR_DSN=postgresql:///finsession python -m pytest -q`.

**Everything is broken and the deadline is close**
Set `vector_store.backend: faiss` in `config.yaml` and carry on with the
in-process index. Nothing else in the project changes.
