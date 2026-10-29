"""Read-only synthetic quality-record, project-controls and engineering-knowledge tools.
No tool accepts a test, waives a hold point, closes a query, approves a change order,
signs a certificate or operates equipment. Open IDs are demo-only.

SECURITY NOTE: lookups take any package or project ID without an entitlement check.
That is acceptable only because every record is fabricated. A real deployment must
scope each lookup to the caller's projects before returning a record.
"""
from __future__ import annotations
from copy import deepcopy
import zlib
from langchain_core.tools import tool
from project_advisor.mock_data import PACKAGES, MODELS, DEVIATIONS, QUALITY_RECORDS, PROJECT_SNAPSHOTS, MAINTENANCE_SCHEDULE, WARRANTIES, SPARES, SPARES_AVAILABILITY, PROJECTS, VENDORS, SERVICE_CENTERS, TECHNICAL_QUERIES
from project_advisor.retrieval import retrieve

def _package(package_id: str) -> dict:
    key=package_id.strip().upper()
    if key not in PACKAGES: raise ValueError(f"Unknown package {package_id!r}; use an ID such as CP-2500-4417.")
    return deepcopy(PACKAGES[key])

def _project(project_id: str) -> dict:
    key=project_id.strip().upper()
    if key.startswith('X'): raise RuntimeError('Synthetic project directory outage. Try later.')
    if key not in PROJECTS: raise ValueError(f"Unknown project {project_id!r}; use PRJ-700001.")
    return deepcopy(PROJECTS[key])

@tool
def search_project_docs(query: str, k: int = 4) -> str:
    """Search project documentation."""
    chunks=retrieve(query,k=max(1,min(k,8)))
    return '\n\n---\n\n'.join(f"[source: {c.metadata.get('source','unknown')}]\n{c.page_content}" for c in chunks) or 'No relevant documentation found.'

@tool
def get_package(package_id: str) -> dict:
    """Read equipment package identity, project, run hours, active deviation and model specification."""
    row=_package(package_id)
    return {'package':row,'model_spec':deepcopy(MODELS[row['model']])}

@tool
def list_packages_for_project(project_id: str) -> list[dict]:
    """List equipment packages belonging to a synthetic project such as PRJ-700001."""
    return [_package(key) for key in _project(project_id)['packages']]

@tool
def lookup_deviation(code: str) -> dict:
    """Read a test deviation threshold, severity and evidence checks by TD code or name words."""
    key=code.strip().upper()
    if key in DEVIATIONS: return deepcopy(DEVIATIONS[key])
    for row in DEVIATIONS.values():
        if key and all(t in row['name'].lower() for t in key.lower().split()): return deepcopy(row)
    raise ValueError(f'Unknown deviation {code!r}; use TD-441.')

@tool
def query_quality_records(package_id: str) -> dict:
    """Read timestamped test, inspection and vendor-surveillance records, including waived hold points.
    Fixed synthetic history, newest first. Compare low-load runs with trip conditions.
    """
    row=_package(package_id)
    return {'package_id':row['package_id'],'source':'simulated quality records register','as_of':row['as_of'],'records':deepcopy(QUALITY_RECORDS[row['package_id']])}

@tool
def get_project_snapshot(package_id: str) -> dict:
    """Read simulated current test-bed values and schedule float with units, quality and snapshot timestamp.
    These are NOT live site values. Distinguish post-trip readings from recorded test peaks.
    """
    return deepcopy(PROJECT_SNAPSHOTS[_package(package_id)['package_id']])

@tool
def get_maintenance_schedule(package_id: str) -> dict:
    """Read hour AND calendar maintenance limits. Both apply, whichever comes first."""
    row=_package(package_id);family=MODELS[row['model']]['family']
    return {k:row[k] for k in ('package_id','run_hours','commissioned','as_of')} | {'family':family,'schedule':deepcopy(MAINTENANCE_SCHEDULE[family])}

@tool
def get_warranty(package_id: str) -> dict:
    """Read vendor warranty dates, hours remaining and exclusions. No coverage approval."""
    row=_package(package_id)
    return deepcopy(WARRANTIES[row['package_id']]) | {'as_of':row['as_of']}

@tool
def find_spare(query: str) -> dict | list[dict]:
    """Find a spare part by PN ID, superseded ID or name. Compatibility is synthetic."""
    key=query.strip().upper()
    if key in SPARES: return deepcopy(SPARES[key])
    replacement=next((r for r in SPARES.values() if r.get('replaces')==key),None)
    if replacement: return deepcopy(replacement) | {'requested_part_id':key,'superseded':True}
    matches=[deepcopy(r) for r in SPARES.values() if key and all(t in r['name'].lower() for t in key.lower().split())]
    if not matches: raise ValueError(f'Unknown spare {query!r}; use PN-4938966.')
    return matches[0] if len(matches)==1 else matches

@tool
def check_spares_availability(part_id: str) -> dict:
    """Read stock and estimated lead days by vendor service center. No order, expedite or promise."""
    key=part_id.strip().upper()
    if key not in SPARES: raise ValueError(f'Unknown spare {part_id!r}; use PN-4938966.')
    return {'part_id':key,'name':SPARES[key]['name'],'source':'simulated vendor inventory','availability':[dict(r,city=SERVICE_CENTERS[r['center']]['city']) for r in SPARES_AVAILABILITY[key]]}

@tool
def get_project(project_id: str) -> dict:
    """Read project and package vendor. Sensitive contact and billing fields are masked."""
    row=_project(project_id)
    for key in ('ssn','phone','email'): row['client_contact'][key]='[REDACTED]'
    row['billing']['tax_id']='[REDACTED]'
    vendor=deepcopy(VENDORS[row['vendor_id']])
    for key in ('phone','email'): vendor['expediter'][key]='[REDACTED]'
    return row | {'vendor':vendor}

@tool
def list_open_queries(package_id: str | None = None) -> list[dict]:
    """List open technical queries (TQs), optionally for one package ID."""
    key=_package(package_id)['package_id'] if package_id else None
    return [deepcopy(r) for r in TECHNICAL_QUERIES.values() if r['status']=='open' and (key is None or r['package_id']==key)]

@tool
def raise_technical_query(package_id: str, reason: str) -> dict:
    """Draft a simulated technical query to the vendor. Not persisted; no record or approval changes."""
    row=_package(package_id);key=row['package_id']
    return {'status':'simulated_draft','tq_id':f"TQ-{zlib.crc32(f'{key}|{reason}'.encode())%100000:05d}",'package_id':key,'reason':reason,'assigned_vendor':PROJECTS[row['project_id']]['vendor_id'],'note':'Demo draft only; not persisted. No test was accepted, no hold point waived and no certificate, order or permit changed.'}

TOOLS=[search_project_docs,get_package,list_packages_for_project,lookup_deviation,query_quality_records,get_project_snapshot,get_maintenance_schedule,get_warranty,find_spare,check_spares_availability,get_project,list_open_queries,raise_technical_query]
