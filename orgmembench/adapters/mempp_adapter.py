"""Mem++ verbatim retrieval adapter; the runner supplies the neutral answerer.

The optional Python 3.13+ engine is imported only on live use/version resolution.
No network, database or embedding work happens during construction or dry-run.
"""
from __future__ import annotations

from datetime import datetime, time as day_time, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from ..schemas import Artifact, Capability, IngestStats, QueryResult, Question
from .base import MemoryAdapter, pkg_version

_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "orgmembench.mempp.aidachip.com")
_AGENT = "orgmembench-mempp"


def _date(value: str | None, *, upper: bool = False):
    if value is None:
        return None
    if len(value) == 10:
        date = datetime.strptime(value, "%Y-%m-%d").date()
        return datetime.combine(date, day_time.max if upper else day_time.min, timezone.utc)
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


class MemppAdapter(MemoryAdapter):
    name = "mempp"
    capabilities = {Capability.RECALL, Capability.PROVENANCE, Capability.TEMPORAL}
    self_answers = False

    def _resolve_version(self) -> str:
        version = pkg_version("memory-research")
        if version == "unknown":
            return version
        # VCS installs expose the exact commit without loading the engine.
        dist = importlib.metadata.distribution("memory-research")
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        commit = direct.get("vcs_info", {}).get("commit_id")
        if not commit and direct.get("dir_info", {}).get("editable"):
            from urllib.parse import unquote, urlparse
            root = Path(unquote(urlparse(direct["url"]).path))
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=5
            ).strip()
            dirty = subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain", "--", "memory", "pyproject.toml", "uv.lock"],
                text=True, timeout=5,
            ).strip()
            if dirty:
                commit += ".dirty"
        return f"{version}+git.{commit}" if commit else f"{version}+unknown-commit"

    def _ids(self, company: str | None = None):
        run = self.config.get("run_id") or os.environ.get("MEMPP_ORGMEMBENCH_RUN_ID", "")
        if not run:
            raise RuntimeError("Set MEMPP_ORGMEMBENCH_RUN_ID to a fresh UUID for each independent run")
        run = str(uuid.UUID(run))
        uid = uuid.uuid5(_NS, json.dumps([run, self.scope], ensure_ascii=True))
        return uid, uuid.uuid5(uid, company) if company is not None else None

    def _sessions(self):
        if hasattr(self, "_checked_sessions"):
            return self._checked_sessions
        from memory import config
        from memory.db import get_engine, get_session
        from sqlalchemy import text
        if get_engine().dialect.name != "postgresql":
            raise RuntimeError("Mem++ requires migrated PostgreSQL/pgvector")
        if config.MEMORY_EMBEDDING_BACKEND != "onnx" or not config.MEMORY_EMBEDDINGS_ENABLED:
            raise RuntimeError("This profile requires enabled local ONNX embeddings; no paid inference")
        actual_profile = (config.MEMORY_RRF_W_FUZZY, config.MEMORY_RRF_W_TAG,
                          config.MEMORY_RRF_W_VECTOR, config.CANDIDATE_LIMIT, config.MEMORY_LEXICAL_OR)
        if actual_profile != (1., 1., 4., 50, False):
            raise RuntimeError("Use the documented OrgMemBench profile: RRF weights 1/1/4, candidates 50, lexical OR off")
        expected = self.config.get("version")
        if expected and self.version() != expected:
            raise RuntimeError("Mem++ runtime differs from the pinned engine version")
        with get_session() as db:
            dim = db.execute(text("SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                                  "WHERE attrelid='agent_memory'::regclass AND attname='embedding'"))
            if dim.scalar_one() != "vector(384)":
                raise RuntimeError("Mem++ requires vector(384)")
        self._checked_sessions = get_session
        return get_session

    def scope_count(self) -> int:
        if self.dry_run:
            return 0
        from memory.models import AgentMemory
        uid, _ = self._ids()
        with self._sessions()() as db:
            return db.query(AgentMemory).filter_by(user_id=uid, agent_type=_AGENT).count()

    # Deliberately keep the base no-op reset_scope: nonempty scopes fail the
    # base ensure_clean_scope check. A fresh run UUID avoids deleting history.

    def _ingest(self, artifacts: list[Artifact]) -> IngestStats:
        from memory.extraction import reconcile_tags
        from memory.models import AgentMemory, MemoryAPIRequest
        from memory.store import PostgresMemoryStore, make_scope_key
        started = time.monotonic()
        prepared = []
        for art in artifacts:
            if not art.text.strip() or not art.slot_id or len(art.slot_id) > 2048:
                raise ValueError("Each artifact needs nonblank source text and a slot ID of 1..2048 characters")
            if art.author and len(art.author) > 250:
                raise ValueError("Author exceeds tag storage limit")
            occurred = _date(art.timestamp)
            # No raw annotations, relevance roles or question/gold data enter memory.
            payload = {key: getattr(art, key) for key in
                       ("slot_id", "company", "source_type", "text", "timestamp", "author", "thread_id")}
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
            uid, pid = self._ids(art.company)
            source_key = json.dumps([art.slot_id, art.thread_id], ensure_ascii=True)
            mid = uuid.uuid5(pid, source_key)
            prepared.append((art, occurred, digest, uid, pid, mid))
        inserted = 0
        with self._sessions()() as db:
            store = PostgresMemoryStore(db)
            for art, occurred, digest, uid, pid, mid in prepared:
                receipt = db.get(MemoryAPIRequest, mid)
                if receipt is not None:
                    if receipt.payload_sha256 != digest:
                        raise ValueError("Artifact ID reused with changed source/metadata; use a new run UUID")
                    continue
                db.add(MemoryAPIRequest(id=mid, payload_sha256=digest))
                row = AgentMemory(id=mid, user_id=uid, project_id=pid,
                                  customer_id=uuid.uuid5(uid, "customer:" + art.company), agent_type=_AGENT,
                                  session_id=uuid.uuid5(pid, art.thread_id) if art.thread_id else None,
                                  source_message_id=mid, content=art.text, context_summary=art.text,
                                  agent_summary=json.dumps({"slot_id": art.slot_id, "author": art.author,
                                                            "source_type": art.source_type, "thread_id": art.thread_id}),
                                  occurred_at=occurred)
                if occurred is not None:
                    row.valid_from = occurred
                store._populate_search_fields(row, tsv_source=art.text, embed_source=art.text)
                if row.embedding is None:
                    raise RuntimeError("Embedding unavailable; the ingestion batch is rolled back")
                tags = []
                if art.author:
                    tags.append("by:" + art.author)
                row.tags = reconcile_tags(db, make_scope_key(uid, _AGENT, pid), tags)
                db.add(row)
                inserted += 1
            db.commit()
        return IngestStats(n_artifacts=len(artifacts), ingest_seconds=time.monotonic() - started,
                           raw={"inserted": inserted, "reused": len(artifacts) - inserted,
                                "engine_version": self.version(), "scope": self.scope,
                                "paid_ingest_calls": 0})

    def _query(self, question: Question, as_of: str | None) -> QueryResult:
        from memory.store import PostgresMemoryStore
        uid, pid = self._ids(question.company)
        top_k = self.config.get("retrieval_top_k", 50)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("retrieval_top_k must be an integer in 1..100")
        upper = _date(as_of, upper=True)
        with self._sessions()() as db:
            rows = PostgresMemoryStore(db).recall(user_id=uid, project_id=pid, agent_type=_AGENT,
                                                 query=question.text, k=top_k, update_hits=False,
                                                 occurred_before=upper)
            snippets, ids, evidence = [], [], []
            for row in rows:
                # Tags normalize case; keep exact source IDs/display authors in
                # the non-indexed recall payload instead of reconstructing them.
                metadata = json.loads(row.agent_summary)
                source, author = metadata["slot_id"], metadata["author"]
                occurred = row.occurred_at
                if occurred is not None and occurred.tzinfo is None:
                    occurred = occurred.replace(tzinfo=timezone.utc)
                date = occurred.astimezone(timezone.utc).isoformat() if occurred else None
                snippets.append(f"[{source}; thread={metadata['thread_id'] or 'not supplied'}; {date or 'undated'}; {author or 'author not supplied'}]\n{row.content}")
                if source not in ids:
                    ids.append(source)
                evidence.append({"artifact_id": source, "record_id": str(row.id), "timestamp": date,
                                 "author": author, "thread_id": metadata["thread_id"],
                                 "source_type": metadata["source_type"]})
        return QueryResult(question_id=question.id, system=self.name,
                           retrieved_context="\n\n".join(snippets) or "No relevant evidence found.",
                           cited_artifact_ids=ids,
                           raw={"evidence": evidence, "as_of": as_of, "top_k": top_k,
                                "engine_version": self.version()})
