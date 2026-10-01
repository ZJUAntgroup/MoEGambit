"""State audit, independent run-risk audit, completion check and compact export."""
from __future__ import annotations

import argparse
from typing import Optional, Sequence
from ..audit.io import read_json, write_json
from ..audit.state import audit_state
from ..audit.run import collect_evidence, verify_run
from ..audit.risk import risk_audit


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    state = sub.add_parser("state", help="compare independently expected and observed state manifests")
    state.add_argument("--expected", required=True)
    state.add_argument("--observed", required=True)
    state.add_argument("--before")
    state.add_argument("--affected", action="append", default=[])
    state.add_argument("--output", required=True)
    run = sub.add_parser("verify-run", help="require completion/commit evidence from every rank")
    run.add_argument("--root", required=True)
    run.add_argument("--output", required=True)
    collect = sub.add_parser("collect", help="export small evidence files, excluding checkpoints")
    collect.add_argument("--root", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--include-logs", action="store_true")
    collect.add_argument("--max-mib", type=int, default=100)
    collect.add_argument("--max-files", type=int, default=10_000)
    risk = sub.add_parser("risk", help="exact upper bound for independent whole-run violations")
    risk.add_argument("--trials", type=int, required=True)
    risk.add_argument("--violations", type=int, required=True)
    risk.add_argument("--confidence", type=float, default=.95)
    risk.add_argument("--alpha-run", type=float, default=.05)
    risk.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "collect":
            report = collect_evidence(args.root, args.output, include_logs=args.include_logs,
                                      max_bytes=args.max_mib * 1024 * 1024, max_files=args.max_files)
            print(f"Exported {len(report['files'])} files ({report['total_bytes']} bytes) to {args.output}")
            print(f"Run verification: {'passed' if report['verification']['passed'] else 'incomplete/failed'}")
            return 0  # Gathering incomplete evidence is allowed and reported.
        if args.command == "state":
            report = audit_state(read_json(args.expected), read_json(args.observed),
                                 before=read_json(args.before) if args.before else None,
                                 affected=args.affected)
        elif args.command == "verify-run":
            report = verify_run(args.root)
        else:
            report = risk_audit(args.trials, args.violations, confidence=args.confidence,
                                alpha_run=args.alpha_run)
        write_json(args.output, report)
        passed = report.get("passed", report.get("within_budget", False))
        print(f"{'PASS' if passed else 'FAIL'}: {args.output}")
        return 0 if passed else 1
    except (OSError, ValueError, TypeError, KeyError) as exc:
        error = {"schema_version": 1, "passed": False, "error": str(exc), "command": args.command}
        # A collect output is a directory; never clobber it with an error file.
        if args.command != "collect":
            write_json(args.output, error)
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
