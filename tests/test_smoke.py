"""Free smoke tests — no LLM, no external systems, no token spend.

Validates the P1 foundation: corpus normalization + question loading on all
three tiers, the reference adapter (dry-run stub + live free path), and the
metrics rollup. Run with pytest, or `python tests/test_smoke.py`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orgmembench.corpus import corpus_stats, load_corpus            # noqa: E402
from orgmembench.questions import load_questions, questions_stats   # noqa: E402
from orgmembench.adapters.reference import ReferenceAdapter         # noqa: E402
from orgmembench.metrics import compute_metrics                     # noqa: E402
from orgmembench.schemas import JudgeResult, QueryResult, RunResult, FailureMode  # noqa: E402

TIERS = ["small", "medium", "large"]
EXPECTED_Q = {"small": 14, "medium": 73, "large": 113}


def test_corpus_loads_and_normalizes():
    for tier in TIERS:
        arts = load_corpus(tier)
        assert arts, f"{tier}: no artifacts"
        # normalization invariants
        assert all(a.slot_id and a.text for a in arts), f"{tier}: empty slot/text"
        # most artifacts should resolve a timestamp
        dated = sum(1 for a in arts if a.timestamp)
        assert dated / len(arts) > 0.5, f"{tier}: too few dated ({dated}/{len(arts)})"
        # at least some authors resolved to display names (not raw P-ids)
        named = [a.author for a in arts if a.author]
        assert named and not any(a.startswith("P-") for a in named), f"{tier}: authors not resolved"
        print(f"  corpus {tier}: {corpus_stats(tier)}")


def test_questions_load():
    for tier in TIERS:
        qs = load_questions(tier)
        assert len(qs) == EXPECTED_Q[tier], f"{tier}: {len(qs)} != {EXPECTED_Q[tier]}"
        assert all(q.text and q.ground_truth_answer is not None for q in qs)
        print(f"  questions {tier}: {questions_stats(tier)}")


def test_reference_adapter_dry_run_and_live():
    qs = load_questions("small")
    arts = load_corpus("small")
    # dry-run: stub, no work
    dry = ReferenceAdapter(dry_run=True)
    stats = dry.ingest(arts)
    assert stats.n_artifacts == len(arts)
    qr = dry.query(qs[0])
    assert qr.dry_run and "DRY_RUN" in qr.answer_text
    # live (free — no LLM): real keyword answer
    live = ReferenceAdapter(dry_run=False)
    live.ingest(arts)
    qr2 = live.query(qs[0])
    assert not qr2.dry_run and qr2.answer_text and qr2.cited_artifact_ids
    assert live.version().startswith("orgmembench-")
    print(f"  reference live answer (truncated): {qr2.answer_text[:80]!r} cites {qr2.cited_artifact_ids}")


def test_metrics_rollup():
    qs = load_questions("small")
    run = RunResult(system="reference", company="helix", tier="small",
                    capabilities=[], dry_run=False)
    for q in qs:
        run.queries.append(QueryResult(question_id=q.id, system="reference", answer_text="x"))
        run.judgements.append(JudgeResult(question_id=q.id, system="reference",
                                           score=0.5, semantic_score=0.6, faithful=True,
                                           failure_mode=FailureMode.NONE))
    m = compute_metrics(run, qs)
    assert m["n_questions"] == len(qs)
    assert 0.0 <= m["accuracy_mean"] <= 1.0
    # temporal categories must be N/A for a system with no TEMPORAL capability
    assert m["by_category"].get("bitemporal") == "N/A", m["by_category"]
    print(f"  metrics: acc={m['accuracy_mean']} by_cat={m['by_category']}")


def test_runner_end_to_end_dry_run():
    """Full ingest→query→judge→metrics loop, dry-run, zero token spend."""
    os.environ["ORGMEMBENCH_DRY_RUN"] = "1"
    from orgmembench.runner import run_system
    run = run_system("reference", "small", limit=8)
    assert run.dry_run is True
    assert len(run.queries) == 8 and len(run.judgements) == 8
    assert all(qr.dry_run for qr in run.queries)
    assert all(jr.judged_dry_run for jr in run.judgements)
    assert "accuracy_mean" in run.metrics and "by_category" in run.metrics
    assert run.system_version  # version resolved
    print(f"  runner dry-run metrics: acc={run.metrics['accuracy_mean']} v={run.system_version}")


def test_runner_parallel_dry_run_preserves_question_order():
    """Parallel question execution returns final results in benchmark order."""
    os.environ["ORGMEMBENCH_DRY_RUN"] = "1"
    from orgmembench.runner import run_system
    run = run_system("reference", "small", limit=8, parallel=4)
    assert [qr.question_id for qr in run.queries] == [f"Q-{i:04d}" for i in range(1, 9)]
    assert [jr.question_id for jr in run.judgements] == [f"Q-{i:04d}" for i in range(1, 9)]
    assert all(qr.dry_run for qr in run.queries)
    assert all(jr.judged_dry_run for jr in run.judgements)


def test_leaderboard_renders():
    from orgmembench.leaderboard import build_leaderboard
    md = build_leaderboard()
    assert isinstance(md, str) and "OrgMemBench leaderboard" in md
    print("  leaderboard renders ok")


def test_orchestration_imports():
    import orgmembench.cli, orgmembench.judge, orgmembench.submission, orgmembench.metrics  # noqa
    from orgmembench.adapters.registry import available
    print(f"  imports ok; adapters available: {available()}")


if __name__ == "__main__":
    os.environ.setdefault("ORGMEMBENCH_DRY_RUN", "1")
    for fn in [test_corpus_loads_and_normalizes, test_questions_load,
               test_reference_adapter_dry_run_and_live, test_metrics_rollup,
               test_runner_end_to_end_dry_run, test_runner_parallel_dry_run_preserves_question_order,
               test_leaderboard_renders,
               test_orchestration_imports]:
        print(f"\n=== {fn.__name__} ===")
        fn()
        print(f"  PASS")
    print("\nALL SMOKE TESTS PASS")
