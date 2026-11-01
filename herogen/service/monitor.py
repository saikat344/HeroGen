"""Read-only generation health checks with local alert deduplication; never starts model work."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time


def assess(report: dict, previous: dict, now: float) -> tuple[dict, dict]:
    summary, policy = report['summary'], report.get('policy')
    conditions = set(summary['alerts']) - {'capacity_full'}
    full = 'capacity_full' in summary['alerts']
    capacity_since = (previous.get('capacity_since') or now) if full else None
    if full and now - capacity_since >= 600:
        conditions.add('capacity_full_10_minutes')
    if policy and not policy['enabled']:
        conditions.add('generation_paused')
    if summary['oldest_active_seconds'] >= 6 * 3600:
        conditions.add('long_running_generation')
    overhead = report.get('overhead')
    if not overhead or now - overhead['updated_at'] >= 86400:
        conditions.add('billing_evidence_missing_or_stale')
    current = sorted(conditions)
    before = previous.get('conditions', [])
    changes = {'new': sorted(set(current)-set(before)), 'resolved': sorted(set(before)-set(current)),
               'allocated_usd': summary['committed_micro']/1_000_000, 'active': summary['active']}
    state = {'checked_at': now, 'conditions': current, 'capacity_since': capacity_since}
    return changes, state


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--app', help='deployed service app; uses its existing admin function without deploying')
    source.add_argument('--report', type=Path, help='saved budget report for local verification')
    parser.add_argument('--state', required=True, type=Path, help='private local deduplication state')
    args=parser.parse_args(argv)
    previous=json.loads(args.state.read_text()) if args.state.exists() else {}
    now=time.time()
    try:
        if args.report:
            report=json.loads(args.report.read_text())
        else:
            import modal
            command=modal.Function.from_name(args.app,'admin_command')
            report=json.loads(command.remote(['generation-budget-report']))
        changes,state=assess(report,previous,now)
    except Exception as error:
        # Never interpret an unreachable meter as zero spend. No exception body or credentials in output.
        state={'checked_at':now,'conditions':['monitor_unavailable'],'capacity_since':previous.get('capacity_since')}
        changes={'new':[] if 'monitor_unavailable' in previous.get('conditions',[]) else ['monitor_unavailable'],
                 'resolved':[], 'error_type':type(error).__name__}
    args.state.parent.mkdir(parents=True,exist_ok=True)
    pending=args.state.with_name(args.state.name+'.tmp')
    pending.write_text(json.dumps(state,indent=2)+'\n'); pending.chmod(0o600); pending.replace(args.state)
    print(json.dumps({'changed':bool(changes['new'] or changes['resolved']),**changes}))
    return 0 if 'monitor_unavailable' not in state['conditions'] else 1


if __name__=='__main__':
    raise SystemExit(main())
