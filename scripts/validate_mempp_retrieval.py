"""Real local retrieval validation only: never invoke the answerer or judge."""
import hashlib
import json
from pathlib import Path
import argparse
import sys
import subprocess
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import time
import uuid

from orgmembench.adapters.mempp_adapter import MemppAdapter,_date
from orgmembench.corpus import load_corpus
from orgmembench.questions import load_questions
from orgmembench.runner import _derive_as_of
from memory.db import get_session
from memory.models import AgentMemory
import orgmembench.adapters.mempp_adapter as adapter_module

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',required=True)
args=parser.parse_args()
out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
run_id=str(uuid.uuid4())
summary=[];traces=[];checked=0;temporal_checked=0
start=time.monotonic()
for tier in ['small','medium']:
    artifacts=load_corpus(tier);questions=load_questions(tier)
    adapter=MemppAdapter({'scope':f'helix-{tier}','run_id':run_id,
                        'version':'0.1.0+git.9a04af72d949a486e71aa7094150ae1a537ab0fe'},dry_run=False)
    adapter.ensure_clean_scope();stats=adapter.ingest(artifacts)
    assert stats.n_artifacts==len(artifacts) and adapter.scope_count()==len(artifacts)
    # Stable artifact/receipt retry, no duplicate source records.
    assert adapter.ingest(artifacts).raw['inserted']==0
    with get_session() as db:
        uid,_=adapter._ids()
        rows=db.query(AgentMemory).filter_by(user_id=uid,agent_type='orgmembench-mempp').all()
        assert all(row.embedding is not None for row in rows)
        stored={str(row.id):(row.user_id,row.project_id,row.customer_id,row.occurred_at,json.loads(row.agent_summary)) for row in rows}
    for q in questions:
        bound=_derive_as_of(q)
        result=adapter.query(q,as_of=bound)
        assert not result.answer_text and result.answer_source==''
        uid,pid=adapter._ids(q.company)
        evidence=result.raw['evidence']
        for hit in evidence:
            suid,spid,customer,occurred,meta=stored[hit['record_id']]
            assert suid==uid and spid==pid and customer==uuid.uuid5(uid,"customer:"+q.company)
            assert meta['slot_id']==hit['artifact_id']
            if bound:
                assert occurred is not None and occurred<=_date(bound,upper=True)
                temporal_checked+=1
            checked+=1
        traces.append({'tier':tier,'question_id':q.id,'as_of':bound,'evidence':evidence})
    digest=hashlib.sha256(json.dumps([a.model_dump() for a in artifacts],sort_keys=True).encode()).hexdigest()
    summary.append({'tier':tier,'artifacts':len(artifacts),'questions':len(questions),
                    'normalized_corpus_sha256':digest,'ingestion_seconds':stats.ingest_seconds,
                    'embedding_count':len(stored),'unique_slot_ids':len({a.slot_id for a in artifacts}),'idempotent_reuse_verified':True})
report={'result':'passed','purpose':'Real local retrieval integration validation; not answer accuracy, official scored run, or reproduction of paper scores','upstream_base_commit':'f17fc92128c560de5c5db0bf982c33845dd98c58','harness_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'adapter_sha256':hashlib.sha256(Path(adapter_module.__file__).read_bytes()).hexdigest(),'engine_version':adapter.version(),'database':'dedicated PostgreSQL17+pgvector UTF8 migration0003','embedding':'real local ONNX all-MiniLM-L6-v2 384d','retrieval_profile':{'k':50,'weights':[1,1,4],'candidate_limit':50,'lexical_or':False},'tiers':summary,'retrieved_record_appearances_checked':checked,'date_bounded_appearances_checked':temporal_checked,'foreign_scope_records':0,'temporal_bound_violations':0,'source_id_mismatches':0,'paid_model_calls':0,'elapsed_seconds':time.monotonic()-start,'run_id':run_id}
(out/'retrieval-traces.json').write_text(json.dumps(traces,indent=2)+'\n')
(out/'retrieval-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
