"""Free contract smoke; optional engine tests use isolated SQLite/PG schemas."""
from __future__ import annotations

from pathlib import Path
import os
import sys
import uuid

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orgmembench.adapters.mempp_adapter import MemppAdapter, _date
from orgmembench.adapters.registry import get_adapter
from orgmembench.schemas import Artifact, Question

RUN = "29d22745-64d1-4e19-a9b2-0710b06c1793"


def artifact(slot="a", company="alpha", **changes):
    return Artifact(slot_id=slot, company=company, source_type="email_thread",
                    text=changes.pop("text", "Atlas launch approved"),
                    timestamp=changes.pop("timestamp", "2024-01-31T16:00:00Z"),
                    author=changes.pop("author", "Alice"), **changes)


def question(company="alpha"):
    return Question(id="q", company=company, tier="small", category="C4", text="Atlas launch?",
                    ground_truth_answer={"answer": "SECRET-GOLD"})


def test_free_construction_and_default_dry_run(monkeypatch):
    monkeypatch.delenv("MEMPP_ORGMEMBENCH_RUN_ID", raising=False)
    monkeypatch.setenv("ORGMEMBENCH_DRY_RUN", "1")
    adapter = get_adapter("mempp")
    monkeypatch.setattr(adapter, "_sessions", lambda: pytest.fail("dry-run touched database"))
    assert adapter.ingest([artifact()]).n_artifacts == 1
    assert adapter.query(question()).dry_run
    adapter.ensure_clean_scope()
    assert not adapter.self_answers


def test_date_bounds_and_missing_run_id():
    assert _date("2024-01-31", upper=True).isoformat() == "2024-01-31T23:59:59.999999+00:00"
    assert _date("2024-01-31T16:00:00-08:00").isoformat() == "2024-02-01T00:00:00+00:00"
    assert _date(None) is None
    with pytest.raises(ValueError):
        _date("not-a-date")
    with pytest.raises((ValueError, RuntimeError)):
        MemppAdapter({"run_id": "invalid"}, dry_run=False)._ids("alpha")


@pytest.fixture(params=["sqlite", "postgres"])
def live(request, tmp_path, monkeypatch):
    pytest.importorskip("memory.models")
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from memory.models import Base
    schema = None
    if request.param == "postgres":
        dsn = os.environ.get("MEMORY_DATABASE_URL", "")
        if not dsn.startswith("postgresql"):
            pytest.skip("Set MEMORY_DATABASE_URL for PostgreSQL tests")
        admin = create_engine(dsn, connect_args={"options": "-c timezone=America/Los_Angeles"})
        schema = "orgmem_test_" + uuid.uuid4().hex
        with admin.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine = admin.execution_options(schema_translate_map={None: schema})
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'memory.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    if request.param == "sqlite":
        monkeypatch.setattr("memory.store.PostgresMemoryStore._populate_search_fields",
                            lambda self, row, **kwargs: setattr(row, "embedding", [1.] + [0.] * 383))
    else:
        class Embedder:
            def embed(self, text, *, intent="document"):
                return [1.] + [0.] * 383
        monkeypatch.setattr("memory.store.get_embedding_service", lambda _: Embedder())
    monkeypatch.setattr(MemppAdapter, "_sessions", lambda _: sessions)
    try:
        yield MemppAdapter({"scope": "alpha-small", "run_id": RUN}, dry_run=False), sessions
    finally:
        if schema:
            with admin.begin() as conn:
                conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        engine.dispose()


def test_sources_temporal_provenance_and_scope(live):
    adapter, sessions = live
    from memory.models import AgentMemory
    arts = [artifact(raw={"answer": "SECRET-ROLE"}, role="noise"),
            artifact("b", text="Atlas launch cancelled", timestamp="2024-03-01", author="Bob"),
            artifact("c", company="beta", text="Atlas foreign project")]
    adapter.ensure_clean_scope()
    assert adapter.ingest(arts).raw["inserted"] == 3
    result = adapter.query(question())
    assert set(result.cited_artifact_ids) == {"a", "b"}
    assert "Alice" in result.retrieved_context and "Bob" in result.retrieved_context
    assert "foreign project" not in result.retrieved_context and "SECRET" not in result.retrieved_context
    assert not result.answer_text and not result.answer_source
    earlier = adapter.query(question(), as_of="2024-01-31")
    assert earlier.cited_artifact_ids == ["a"]
    assert earlier.raw["evidence"][0]["timestamp"] == "2024-01-31T16:00:00+00:00"
    assert adapter.ingest(arts).raw["reused"] == 3
    assert adapter.query(question()).raw["evidence"] == result.raw["evidence"]
    with pytest.raises(RuntimeError, match="still holds"):
        adapter.ensure_clean_scope()
    assert adapter.scope_count() == 3
    assert MemppAdapter({"scope": "alpha-medium", "run_id": RUN}, dry_run=False).scope_count() == 0
    assert MemppAdapter({"scope": "alpha-small", "run_id": str(uuid.uuid4())}, dry_run=False).scope_count() == 0
    with sessions() as db:
        assert all(row.hit_count == 0 for row in db.query(AgentMemory))


def test_changed_id_batch_rolls_back(live):
    adapter, _ = live
    adapter.ingest([artifact()])
    with pytest.raises(ValueError, match="changed"):
        adapter.ingest([artifact("new"), artifact(text="Changed source")])
    assert adapter.scope_count() == 1


def test_embedding_failure_rolls_back(live, monkeypatch):
    adapter, sessions = live
    from memory.models import MemoryAPIRequest
    monkeypatch.setattr("memory.store.PostgresMemoryStore._populate_search_fields", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="rolled back"):
        adapter.ingest([artifact()])
    assert adapter.scope_count() == 0
    with sessions() as db:
        assert db.query(MemoryAPIRequest).count() == 0


def test_invalid_metadata_no_partial_write(live):
    adapter, _ = live
    with pytest.raises(ValueError):
        adapter.ingest([artifact(), artifact("b", timestamp="not-a-date")])
    assert adapter.scope_count() == 0




def test_exact_source_id_and_author_case(live):
    adapter, _ = live
    adapter.ingest([artifact("Source-UPPER-ID", author="Alice McKay")])
    result = adapter.query(question())
    assert result.cited_artifact_ids == ["Source-UPPER-ID"]
    assert result.raw["evidence"][0]["author"] == "Alice McKay"


def test_unknown_dates_not_invented(live):
    adapter, _ = live
    adapter.ingest([artifact(timestamp=None, author=None)])
    result = adapter.query(question())
    assert result.raw["evidence"][0]["timestamp"] is None
    assert "undated" in result.retrieved_context
    assert not adapter.query(question(), as_of="2024-01-31").cited_artifact_ids


def test_duplicate_slot_ids_in_distinct_threads_preserved(live):
    adapter, _ = live
    adapter.ingest([artifact("SAME", thread_id="thread-A"),
                    artifact("SAME", thread_id="thread-B", text="Atlas second source")])
    assert adapter.scope_count() == 2
    result = adapter.query(question())
    assert result.cited_artifact_ids == ["SAME"]
    assert {x["thread_id"] for x in result.raw["evidence"]} == {"thread-A", "thread-B"}
    assert len({x["record_id"] for x in result.raw["evidence"]}) == 2


def test_preflight_rejects_unsupported_runtime(monkeypatch):
    pytest.importorskip("memory")
    from memory import config, db
    from types import SimpleNamespace
    from contextlib import contextmanager
    monkeypatch.setattr(db, "get_engine", lambda: SimpleNamespace(dialect=SimpleNamespace(name="sqlite")))
    with pytest.raises(RuntimeError, match="PostgreSQL"):
        MemppAdapter(dry_run=False)._sessions()
    monkeypatch.setattr(db, "get_engine", lambda: SimpleNamespace(dialect=SimpleNamespace(name="postgresql")))
    monkeypatch.setattr(config, "MEMORY_EMBEDDING_BACKEND", "openai")
    with pytest.raises(RuntimeError, match="ONNX"):
        MemppAdapter(dry_run=False)._sessions()
    monkeypatch.setattr(config, "MEMORY_EMBEDDING_BACKEND", "onnx")
    monkeypatch.setattr(config, "MEMORY_EMBEDDINGS_ENABLED", True)
    for key, value in [("MEMORY_RRF_W_FUZZY", 1.), ("MEMORY_RRF_W_TAG", 1.),
                       ("MEMORY_RRF_W_VECTOR", 4.), ("CANDIDATE_LIMIT", 50), ("MEMORY_LEXICAL_OR", False)]:
        monkeypatch.setattr(config, key, value)
    monkeypatch.setattr(config, "CANDIDATE_LIMIT", 200)
    with pytest.raises(RuntimeError, match="profile"):
        MemppAdapter(dry_run=False)._sessions()
    monkeypatch.setattr(config, "CANDIDATE_LIMIT", 50)
    with pytest.raises(RuntimeError, match="pinned"):
        MemppAdapter({"version": "impossible"}, dry_run=False)._sessions()
    @contextmanager
    def wrong_schema():
        yield SimpleNamespace(execute=lambda _: SimpleNamespace(scalar_one=lambda: "vector(1536)"))
    monkeypatch.setattr(db, "get_session", wrong_schema)
    with pytest.raises(RuntimeError, match="vector"):
        MemppAdapter(dry_run=False)._sessions()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
