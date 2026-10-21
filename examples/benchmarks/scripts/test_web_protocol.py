"""Protocol correction and immutable measurements; no fabricated observations."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from web_protocol import load_web_exports

DATA = Path(__file__).resolve().parents[1] / 'data'


class WebProtocolTests(unittest.TestCase):
    def copy_inputs(self, target):
        for name in ('web_update_pipeline.csv', 'web_update_plan.json',
                     'fiqa_quality.csv', 'web_protocol_correction.json'):
            shutil.copyfile(DATA / name, target / name)
        shutil.copytree(DATA / 'archive', target / 'archive')

    def test_author_corrected_protocol_preserves_measurements(self):
        meta, rows, correction = load_web_exports(DATA)
        self.assertTrue(meta['plan']['r2_enabled'])
        self.assertEqual(len(rows), 12)
        self.assertFalse(correction['numerical_measurements_changed'])
        self.assertEqual({r['r2_role'] for r in rows if r['arm'] == 'restart'},
                         {'restart_reference'})
        self.assertEqual({r['r2_role'] for r in rows if r['arm'] == 'hybrid'},
                         {'quality_admission'})

    def test_active_r2_disabled_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d); self.copy_inputs(target)
            path = target / 'web_update_plan.json'
            obj = json.loads(path.read_text()); obj['plan']['r2_enabled'] = False
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, 'R2 admission enabled'):
                load_web_exports(target)

    def test_new_measurements_cannot_hide_in_metadata_correction(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d); self.copy_inputs(target)
            path = target / 'web_update_pipeline.csv'
            path.write_text(path.read_text().replace('1030.5431196689606', '1031.5431196689606'))
            correction_path = target / 'web_protocol_correction.json'
            obj = json.loads(correction_path.read_text())
            obj['corrected_input_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            correction_path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, 'Measurement changed'):
                load_web_exports(target)

    def test_frozen_original_archive_is_verified(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d); self.copy_inputs(target)
            path = target / 'archive/web_update_20261020/web_update_plan.json'
            path.write_text(path.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'Archived source hash mismatch'):
                load_web_exports(target)


if __name__ == '__main__':
    unittest.main()
