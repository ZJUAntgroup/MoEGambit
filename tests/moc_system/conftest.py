"""Load the source-checkout benchmark modules without optional GPU frameworks."""
from pathlib import Path
import sys

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "moc_system"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
