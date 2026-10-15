# Mem++ (local ONNX + PostgreSQL/pgvector)

This adapter returns dated source evidence and source artifact IDs. It leaves
`self_answers=False`, so the harness uses its unchanged neutral answerer and
judge. It declares RECALL, PROVENANCE, and event-date TEMPORAL filtering. It does
not claim role-relative SCOPE or calibrated NEGATIVE/abstention capabilities.
Event-date filtering is not a claim of full two-clock bi-temporal database semantics.

## Install the optional engine

Use a Python **3.13+** environment (the core harness otherwise supports 3.10+).
From the OrgMemBench checkout:

```sh
uv venv --python 3.13
uv pip install -e '.[dev]'
uv pip install 'memory-research @ git+https://github.com/AIDAChip-Inc/mem-plus-plus.git@9a04af72d949a486e71aa7094150ae1a537ab0fe'
```

The adapter imports the engine lazily. Dry-run construction/ingestion/query work
without the engine, PostgreSQL, embedding model, or API keys. The engine runtime
version/commit is recorded and checked against `config/mempp.yaml` on live use.
A dirty editable engine checkout is marked `.dirty` and fails that pin check.

Create a dedicated PostgreSQL/pgvector database and apply the **engine's**
Alembic migrations through 0003 from its checkout (see the
[engine quickstart](https://github.com/AIDAChip-Inc/mem-plus-plus#quickstart)).
Set `MEMORY_DATABASE_URL` externally; never commit a real DSN or credentials.
This is an in-process adapter with a private DB connection, not a public HTTP API.

## Configure the organizational retrieval profile

Export these **before starting Python** because the engine reads them on import:

```sh
export MEMORY_EMBEDDING_BACKEND=onnx
export MEMORY_EMBEDDINGS_ENABLED=1
export MEMORY_RRF_W_FUZZY=1
export MEMORY_RRF_W_TAG=1
export MEMORY_RRF_W_VECTOR=4
export MEMORY_CANDIDATE_LIMIT=50
export MEMORY_LEXICAL_OR=0
export MEMPP_ORGMEMBENCH_RUN_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
```

The adapter fails on the wrong database/schema, disabled/non-ONNX embeddings,
wrong retrieval profile, or engine pin. Startup checks are cached per adapter.
The adapter uses private engine indexing helpers under the explicit commit pin. It does not change global engine flags.
The first ONNX load downloads the pinned MiniLM weights from Hugging Face. Allow
outbound access or populate the cache first. No generative inference occurs in
the adapter's ingest/retrieval code.

## Scope, source retention, and temporal semantics

The run UUID plus the runner's explicit scope (normally company-tier) derives a
user namespace; each company within it gets a separate project/customer mapping.
All source authors and threads in one organizational world share records.
Independent runs and tiers use distinct namespaces. The adapter never clears
stored organizational history: `ensure_clean_scope()` refuses a nonempty scope.
Generate a fresh run UUID for a clean run. Reusing a UUID/scope intentionally
resumes its old records; use `ORGMEMBENCH_REUSE_INGEST=1` only deliberately.

Each artifact is stored whole as `content`; author tags support retrieval. Exact
source IDs and author display names are preserved in JSON in the engine's optional
`agent_summary` recall-payload field (not generated text and not embedded).
Thread identity and source dates are preserved. The adapter does not ingest raw
annotations, relevance-role labels, gold answers, or rubric fields. Stable record
IDs include the source slot ID and thread because the upstream small tier reuses
some slot IDs across threads. Citations retain the original slot IDs; evidence also
reports thread and record IDs. Atomic receipts make identical ingests idempotent; changed content or
metadata under an existing source identity (slot ID plus thread) fails instead of overwriting. A missing
embedding rolls back the complete batch. Sequential corpus ingestion is expected;
a concurrent conflicting ingest fails atomically and can be retried unchanged.

`as_of=YYYY-MM-DD` includes the entire UTC day. Explicit timestamps retain their
instant. Records without source dates stay undated and are excluded from bounded
as-of retrieval, rather than receiving invented event dates. The shortlist is
passed to the official neutral answerer with source IDs, dates, and supplied
authors. Querying does not update memory hit counts.

## Free validation and eventual scoring

```sh
orgmembench list
orgmembench stats --tier small
python tests/test_mempp_smoke.py
# Full pipeline plumbing only; no service or model calls:
orgmembench run --system mempp --tier small
```

Optional engine tests use isolated SQLite/PG schemas and deterministic fake
vectors; PostgreSQL cases require `MEMORY_DATABASE_URL`. A real ONNX retrieval
validation can call `adapter.ingest()` / `adapter.query()` with `dry_run=False`
without calling the answerer or judge. With the live database/profile configured,
run the supplied audit (fresh run UUID generated automatically):

```sh
python scripts/validate_mempp_retrieval.py --output-dir /tmp/mempp-retrieval-audit
```

It writes a report plus per-question evidence traces and checks source IDs,
user/project/customer scopes, embeddings, idempotence, and explicit temporal
bounds. Run without `python -O` or `PYTHONOPTIMIZE`: the audit uses fail-fast
assertions. A zero-violation report is written only after every check passes.
It never invokes the answerer or judge. This is **retrieval validation**, not a
leaderboard score or a reproduction of the historical Mem++ paper scores.

A scored run is separate:

```sh
# Only after arranging an explicit inference budget and credentials:
ORGMEMBENCH_DRY_RUN=0 orgmembench run --system mempp --tier medium --execute
```

This invokes the benchmark's answerer and judge and can incur inference charges.
No scored results accompany this adapter contribution. Inclusion of an adapter
or a dry-run artifact does not establish leaderboard acceptance.
