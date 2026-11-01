"""Basic monitor alert transitions, not paid service calls."""
import contextlib
import io
import json
from test_herogen_service_support import TempDirCase
from herogen.service.monitor import assess, main

class TestMonitor(TempDirCase):
    def report(self,alerts=()):
        return {'policy':{'enabled':1},'summary':{'alerts':list(alerts),'committed_micro':500_000_000,
            'active':5,'oldest_active_seconds':30},'overhead':{'updated_at':1000}}
    def test_threshold_alert_is_deduplicated_and_recovery_is_visible(self):
        r=self.report(['budget_50_percent']); change,state=assess(r,{},1001)
        self.assertEqual(change['new'],['budget_50_percent']); self.assertEqual(change['allocated_usd'],500)
        change,state=assess(r,state,1002); self.assertEqual(change['new'],[])
        change,_=assess(self.report(),state,1003); self.assertEqual(change['resolved'],['budget_50_percent'])
    def test_capacity_alert_requires_ten_continuous_minutes(self):
        r=self.report(['capacity_full']); change,state=assess(r,{},1000)
        self.assertEqual(change['new'],[])
        change,state=assess(r,state,1599); self.assertEqual(change['new'],[])
        change,state=assess(r,state,1600); self.assertIn('capacity_full_10_minutes',change['new'])
        change,state=assess(self.report(),state,1601); self.assertIsNone(state['capacity_since'])
        change,_=assess(r,state,1602); self.assertEqual(change['new'],[])
    def test_paused_long_running_and_stale_evidence_are_loud(self):
        r=self.report(); r['policy']['enabled']=0; r['overhead']=None
        r['summary']['oldest_active_seconds']=21600
        change,_=assess(r,{},2000)
        self.assertEqual(set(change['new']),{'generation_paused','billing_evidence_missing_or_stale','long_running_generation'})
    def test_unavailable_source_is_not_zero_and_recovery_is_reported(self):
        report,state=self.tmp/'report.json',self.tmp/'state.json'
        def run():
            out=io.StringIO()
            with contextlib.redirect_stdout(out): result=main(['--report',str(report),'--state',str(state)])
            return result,json.loads(out.getvalue())
        result,out=run(); self.assertEqual(result,1); self.assertEqual(out['new'],['monitor_unavailable'])
        result,out=run(); self.assertEqual(out['new'],[])
        report.write_text(json.dumps(self.report()))
        result,out=run(); self.assertEqual(result,0); self.assertIn('monitor_unavailable',out['resolved'])
        self.assertEqual(state.stat().st_mode & 0o777,0o600)
