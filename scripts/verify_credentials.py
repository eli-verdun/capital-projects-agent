"""Fail-fast real credential checks for local setup and GitHub Actions.

Only names, IDs and statuses are reported. Secret values are never emitted.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys
import time
import uuid
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
load_dotenv(ROOT/'.env',override=False)
from langsmith import Client
from langsmith.anonymizer import create_anonymizer
from langsmith.utils import LangSmithNotFoundError
from langchain_openai import ChatOpenAI
from project_advisor.llm import agent_model, guardrail_model, judge_model, openai_gateway_kwargs
from project_advisor.retrieval import retrieve
from project_advisor.redaction import REDACTION_RULES
from project_advisor.context import CONTEXT_HUB_REPO
from project_advisor.governance import APPLICATION_NAME
from project_advisor.agent_address import address, agent_id, conflicting_destination_vars, environment_session, refuse_production


def main():
    refuse_production()
    for name in ('LANGSMITH_API_KEY','LANGSMITH_GATEWAY_API_KEY','LANGSMITH_WORKSPACE_ID','LANGSMITH_AGENT_ID','LANGSMITH_AGENT_ENVIRONMENT'):
        if not os.getenv(name): raise RuntimeError(f'Missing required configuration: {name}')
    assert not conflicting_destination_vars(),'Unset LANGSMITH_PROJECT/LANGCHAIN_PROJECT/LANGCHAIN_SESSION: they conflict with agent addressing'
    assert agent_id()==CONTEXT_HUB_REPO==APPLICATION_NAME=='capital-projects-agent','Application naming mismatch'
    client=Client(anonymizer=create_anonymizer(REDACTION_RULES),auto_batch_tracing=False)
    env=environment_session(client)
    p=client.read_project(project_id=env.session_id)
    assert str(p.tenant_id)==os.environ['LANGSMITH_WORKSPACE_ID'],'Agent environment is in a different workspace'
    print(f'PASS: tracing credential; workspace={p.tenant_id}; agent={env.agent_id}; environment={env.environment}; session={env.session_id}')
    ctx=client.pull_agent(CONTEXT_HUB_REPO,version='production')
    required=['AGENTS.md','guardrail.md','skills/deviation-triage/SKILL.md','skills/maintenance-and-data-review/SKILL.md','skills/spares-and-warranty/SKILL.md']
    assert all(k in ctx.files for k in required),'Context Hub files missing'
    for alias,count in [('golden',16),('hallucinations',14),('pii',12),('guardrail',22)]:
        ds=client.read_dataset(dataset_name=f'capital-projects-{alias}')
        assert len(list(client.list_examples(dataset_id=ds.id)))==count,f'Dataset count mismatch: {alias}'
    print('PASS: production Context Hub and all four datasets')
    for name,model in [('AGENT_MODEL',agent_model()),('GUARDRAIL_MODEL',guardrail_model()),('EVAL_JUDGE_MODEL',judge_model())]:
        response=ChatOpenAI(model=model,timeout=60,max_retries=1,**openai_gateway_kwargs()).invoke('Reply with OK only.')
        assert response.content,f'Empty response for {name}'
        print(f'PASS: gateway credential for {name} ({model})')
    hits=retrieve('hydrostatic test pressure stress ratio',k=2)
    assert hits and hits[0].metadata['source']=='hydrotest_pressure_calculation.md','Local KB retrieval returned the wrong document'
    print('PASS: local KB retrieval (no embedding provider needed)')
    run_id=uuid.uuid4();now=datetime.now(timezone.utc)
    sentinel='900-55-0142'
    # Address the run to the agent environment, the same way the app traces.
    client.create_run(id=run_id,name='credential-verification',run_type='chain',address=address(),inputs={'message':'Synthetic identifier '+sentinel},outputs={'message':'Verified'},start_time=now,end_time=now,tags=['credential-verification','synthetic'])
    for attempt in range(20):
        try:
            run=client.read_run(run_id);break
        except LangSmithNotFoundError:
            if attempt==19: raise
            time.sleep(1)
    assert str(run.session_id)==env.session_id,'Run did not land in the agent environment'
    assert sentinel not in json.dumps(run.inputs) and '<ssn>' in json.dumps(run.inputs),'Trace redaction/readback failed'
    print(f'PASS: anonymized trace write/readback to the agent environment; run={run_id}')
    print('CREDENTIAL_VERIFICATION=PASS')


if __name__=='__main__':
    try: main()
    except Exception as exc:
        # Credentials may appear inside library errors; report type and safe local assertions only.
        print(f'CREDENTIAL_VERIFICATION=FAIL ({type(exc).__name__})',file=sys.stderr)
        if isinstance(exc,(AssertionError,LookupError)):print(str(exc),file=sys.stderr)
        sys.exit(1)
