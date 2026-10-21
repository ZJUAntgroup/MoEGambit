"""Validate the author-confirmed Web R2 protocol without synthesizing decisions.

Original export bytes remain archived. Only R2 protocol metadata is corrected;
all measurements and non-R2 scientific parameters must match the archive.
"""
import csv
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def load_web_exports(data):
    data = Path(data)
    correction = json.loads((data / 'web_protocol_correction.json').read_text())
    meta = json.loads((data / 'web_update_plan.json').read_text())
    rows = read_csv(data / 'web_update_pipeline.csv')
    require(correction['schema_version'] == 1 and
            correction['confirmation_type'] == 'author_reported_protocol_correction',
            'Expected explicit author-confirmed protocol correction')
    require(correction['r2_enabled'] is True and meta['plan']['r2_enabled'] is True,
            'Web rerun requires R2 admission enabled')
    require(correction['r2_admission_stage'] == meta['plan']['r2_admission_stage'] ==
            'before_hybrid_restoration', 'R2 must precede Hybrid restoration')
    require(correction['numerical_measurements_changed'] is False and
            correction['r2_decision_records_bundled'] is False,
            'Protocol correction must not create measurements or decision traces')
    for name, digest in correction['archived_inputs_sha256'].items():
        path = data / name
        require(path.resolve().is_relative_to(data.resolve()), 'Archive path leaves input directory')
        require(sha256(path) == digest, 'Archived source hash mismatch: ' + name)
    archive = data / 'archive/web_update_20261020'
    original = json.loads((archive / 'web_update_plan.json').read_text())
    require(original['plan']['r2_enabled'] is False and
            correction['superseded_r2_enabled'] is False,
            'Expected preserved historical flag in archived export')
    expected_plan = dict(original['plan'], r2_enabled=True,
                         r2_admission_stage='before_hybrid_restoration')
    require(meta['plan'] == expected_plan, 'Non-R2 scientific parameters changed')
    require(meta['config_hash'] == original['config_hash'] ==
            correction['original_cluster_config_hash'], 'Archived cluster identifier changed')
    require('archiv' in meta['config_hash_scope'].lower(),
            'Original configuration hash must be labelled archival')
    normalized = hashlib.sha256(json.dumps(meta['plan'], sort_keys=True,
                                           separators=(',', ':')).encode()).hexdigest()
    require(normalized == meta['normalized_plan_sha256'] == correction['normalized_plan_sha256'],
            'Normalized corrected-plan hash mismatch')
    for key, value in original.items():
        if key not in ('plan', 'interpret_as'):
            require(meta[key] == value, 'Original aggregate metadata changed: ' + key)
    old_rows = read_csv(archive / 'web_update_pipeline.csv')
    require(len(rows) == len(old_rows), 'Original measurement row count changed')
    for row, old in zip(rows, old_rows):
        role = 'quality_admission' if old['arm'] == 'hybrid' else 'restart_reference'
        require(old['r2_enabled'] == 'False' and row['r2_enabled'] == 'True',
                'Expected corrected R2 flag in active rows')
        require(row == dict(old, r2_enabled='True', r2_role=role),
                'Measurement changed or incorrect R2 role: ' + str(row.get('case_id')))
    for name, digest in correction['corrected_input_sha256'].items():
        require(sha256(data / name) == digest, 'Corrected input hash mismatch: ' + name)
    require(sha256(data / 'fiqa_quality.csv') == correction['unchanged_task_export_sha256'],
            'FiQA observations changed during protocol correction')
    return meta, rows, correction


def main():
    from paths import DATA, OUT
    meta, rows, correction = load_web_exports(DATA)
    report = dict(protocol_verified=True, r2_enabled=True,
                  r2_admission_stage=meta['plan']['r2_admission_stage'],
                  protocol_correction_id=correction['id'],
                  evidence_source=correction['confirmation_type'],
                  rows=len(rows), numerical_measurements_changed=False,
                  r2_decision_records_bundled=False,
                  original_cluster_config_hash=meta['config_hash'],
                  normalized_plan_sha256=meta['normalized_plan_sha256'])
    (OUT / 'web_protocol_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
