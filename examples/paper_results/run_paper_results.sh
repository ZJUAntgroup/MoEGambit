#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# CPU only. Figures, reports and stage diagnostics stay under /personal by default.
exec "${PYTHON:-python3}" -u "$SCRIPT_DIR/reproduce.py" \
  --out-dir "${PAPER_RESULTS_DIR:-/personal/moegambit/paper_results}" "$@"
