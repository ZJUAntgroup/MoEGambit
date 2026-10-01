"""Expose pytest failures as GitHub annotations and a readable job summary."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def _escape(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(argv=None) -> int:
    paths = list(sys.argv[1:] if argv is None else argv)
    if len(paths) != 1:
        print("usage: report_ci_results.py PYTEST_JUNIT_XML", file=sys.stderr)
        return 2
    path = Path(paths[0])
    if not path.is_file():
        print("No pytest results: the test step did not produce a JUnit report.")
        return 0
    root = ET.parse(path).getroot()
    failures = []
    for case in root.iter("testcase"):
        for issue in case:
            if issue.tag not in {"failure", "error"}:
                continue
            name = f"{case.get('classname', '')}.{case.get('name', '')}"
            detail = (issue.text or issue.get("message", "")).strip()
            failures.append((name, detail))
            print(f"::error::{_escape(name + ': ' + detail)}")
    summary = [f"## CPU test results: {len(failures)} failures\n"]
    for name, detail in failures:
        escaped_detail = detail.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        summary.extend([f"### {name}\n", "<pre>\n", escaped_detail + "\n", "</pre>\n"])
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
            stream.write("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
