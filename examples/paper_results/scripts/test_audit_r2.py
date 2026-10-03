"""Unit fixtures only: these are not experimental observations."""
import json
from pathlib import Path
import tempfile
import unittest
from audit_r2 import binomial_upper, records, summarize_decisions, summarize_runs


def decision(action='Hybrid', score=0.02, eligible=True, y=None):
    return dict(run_id='unit-fixture', event_id='event-fixture', step=10,
                policy_id='fixture-policy', action=action, reason='fixture-reason',
                guard_eligible=eligible, risk_upper=score, alpha_run=0.05,
                hybrid_Y=y, label_scope='complete_candidate_continuation',
                label_source='unit-test-only')


class AuditTests(unittest.TestCase):
    def test_zero_exceedance_sample_size(self):
        self.assertGreater(binomial_upper(0, 58), 0.05)
        self.assertLess(binomial_upper(0, 59), 0.05)
        self.assertAlmostEqual(binomial_upper(0, 1), 0.95)

    def test_nonzero_exact_limit(self):
        self.assertAlmostEqual(binomial_upper(1, 2), 0.9746794344808963)
        self.assertEqual(binomial_upper(2, 2), 1.0)

    def test_missing_label_is_not_safe(self):
        r = summarize_decisions([decision()], .05)
        self.assertIsNone(r['unsafe_fraction_among_labelled_admissions']['fraction'])
        self.assertEqual(r['hybrid_label_coverage']['fraction'], 0)
        self.assertFalse(r['complete_decision_labels'])

    def test_guard_failure_not_quality_error(self):
        r = summarize_decisions([decision('Restart', None, False)], .05)
        self.assertEqual(r['quality_gate_rejections'], 0)
        self.assertIsNone(r['safe_fraction_among_labelled_quality_rejections']['fraction'])

    def test_unsafe_admission_and_safe_rejection(self):
        r = summarize_decisions([decision(y=1.1), decision('Restart', .1, y=.4)], .05)
        self.assertEqual(r['unsafe_fraction_among_labelled_admissions']['fraction'], 1)
        self.assertEqual(r['safe_fraction_among_labelled_quality_rejections']['fraction'], 1)

    def test_admission_above_budget_rejected(self):
        with self.assertRaises(ValueError): summarize_decisions([decision(score=.1)], .05)

    def test_short_label_rejected(self):
        row = decision(y=.2); row['label_scope'] = '100_step_window'
        with self.assertRaises(ValueError): summarize_decisions([row], .05)

    def test_conflicting_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'unit-fixtures.jsonl'; a=decision(); b=decision('Restart', .2)
            p.write_text(json.dumps(a)+'\n'+json.dumps(a)+'\n')
            self.assertEqual(records([p], ('run_id', 'event_id'))[1], 1)
            p.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n')
            with self.assertRaises(ValueError): records([p], ('run_id', 'event_id'))

    def test_shared_prefix_has_no_interval(self):
        rows=[dict(run_id=str(i), complete=True, policy_Y=0, independent_unit_id='same-prefix') for i in range(59)]
        r=summarize_runs(rows, None, .05, .05)
        self.assertIsNone(r['iid_binomial_upper'])
        self.assertIsNone(r['certificate_under_declared_assumptions'])

    def test_declared_independent_frozen_audit(self):
        manifest=dict(iid_complete_runs_declared=True, policy_frozen_before_audit=True,
                      audit_size_fixed_before_outcomes=True, no_audit_tuning_declared=True,
                      alpha_run=.05, delta=.05, eta_final=.005, eta_peak=.01,
                      policy_id='test-policy', policy_artifact_sha256='fixture-only',
                      audit_protocol_id='fixture-only', distribution_description='unit test',
                      fixed_evaluation_grid_id='test-grid', training_unit_ids=[],
                      calibration_unit_ids=[], planned_independent_audit_runs=59)
        rows=[dict(run_id=str(i), complete=True, policy_Y=.1, independent_unit_id=str(i),
                   policy_id='test-policy', evaluation_grid_id='test-grid',
                   reference_policy='whole_job_Restart', outcome_source='fixture-only') for i in range(59)]
        r=summarize_runs(rows, manifest, .05, .05)
        self.assertTrue(r['certificate_under_declared_assumptions'])
        manifest['calibration_unit_ids']=['0']
        self.assertIsNone(summarize_runs(rows, manifest, .05, .05)['iid_binomial_upper'])


if __name__ == '__main__': unittest.main()
