"""Read-only paper inputs; all generated files go to the requested output root."""
import os
from pathlib import Path
DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(os.environ.get("MOEGAMBIT_PAPER_RESULTS_OUT", "/personal/moegambit/paper_results")).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
