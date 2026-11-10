"""Regressions for information loss in the course-template conversion."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("build_cvpr", Path(__file__).parents[1] / "report" / "build_cvpr.py")
cvpr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cvpr)


def test_abstract_percentages_are_escaped_by_markdown_conversion():
    text = "# Title\n\n## Abstract\n\nSuccess is 87.6% and this sentence must survive.\n\n## 1. Question\n\nQuestion."
    prepared = cvpr.prepare_markdown(text)
    # Raw-TeX abstract blocks bypass Pandoc escaping and comment out everything
    # after the first result's percent sign. Keep prose as Markdown until conversion.
    assert r"\begin{abstract}" not in prepared
    assert "Success is 87.6% and this sentence must survive." in prepared
    latex = cvpr.prepare_latex(
        "\\section{Abstract}\\label{abstract}\nSuccess is 87.6\\% and this sentence must survive.\n"
        "\\section{Question}\\label{question}\nQuestion."
    )
    assert latex.count(r"\begin{abstract}") == latex.count(r"\end{abstract}") == 1
    assert r"87.6\% and this sentence must survive." in latex
