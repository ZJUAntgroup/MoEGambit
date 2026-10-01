"""CI exposes the actual failing test without losing multiline diagnostics."""

from pathlib import Path
import runpy


def test_junit_failures_become_escaped_annotations_and_summary(tmp_path, monkeypatch, capsys):
    report = tmp_path / "pytest.xml"
    report.write_text('<testsuites><testsuite><testcase classname="tests.check" name="ok"/>'
                      '<testcase classname="tests.check" name="bad"><failure>50%\n'
                      'AssertionError: &lt;state&gt;</failure></testcase></testsuite></testsuites>')
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/report_ci_results.py"))
    assert script["main"]([str(report)]) == 0
    assert capsys.readouterr().out == '::error::tests.check.bad: 50%25%0AAssertionError: <state>\n'
    assert "1 failures" in summary.read_text()
    assert "&lt;state&gt;" in summary.read_text()


def test_no_junit_report_is_reported_explicitly(tmp_path, capsys):
    script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/report_ci_results.py"))
    assert script["main"]([str(tmp_path / "missing.xml")]) == 0
    assert "did not produce" in capsys.readouterr().out
