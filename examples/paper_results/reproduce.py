#!/usr/bin/env python3
"""Check paper aggregates and replot supplied exports; does not start training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parent / 'scripts'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path('/personal/moegambit/paper_results'))
    parser.add_argument('--no-plots', action='store_true', help='Standard-library aggregate checks only')
    args = parser.parse_args()
    out = args.out_dir.expanduser().resolve()
    # Preserve immutable input exports and publication assets.
    bundle = SCRIPTS.parent.resolve()
    assets = bundle.parents[1] / 'docs/assets'
    if out == bundle or bundle in out.parents or out == assets or assets in out.parents:
        parser.error('Output must be outside the checked-in inputs/assets')
    out.mkdir(parents=True, exist_ok=True)
    logs = out / 'logs'; logs.mkdir(exist_ok=True)
    env = dict(os.environ, MOEGAMBIT_PAPER_RESULTS_OUT=str(out), MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    stages = ['report_results.py']
    if not args.no_plots:
        stages += ['plot_recovery_scaling.py', 'plot_checkpoint.py', 'plot_experts.py',
                   'plot_terminal.py', 'plot_architecture.py', 'plot_downstream.py']
    completed = []
    for stage in stages:
        log = logs / (Path(stage).stem + '.log')
        print(f'[paper-results] {stage}: {log}', flush=True)
        with log.open('w') as f:
            result = subprocess.run([sys.executable, str(SCRIPTS / stage)], env=env,
                                    stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text()[-12000:], file=sys.stderr)
            raise SystemExit(f'{stage} failed ({result.returncode}); inspect {log}')
        completed.append(stage)
    (out / 'reproduction_summary.json').write_text(json.dumps(dict(
        complete=True, stages=completed, mode='aggregate verification and replotting',
        gpu_training_started=False, result_dir=str(out)), indent=2) + '\n')
    print(f'[paper-results] complete: {out}', flush=True)


if __name__ == '__main__':
    main()
