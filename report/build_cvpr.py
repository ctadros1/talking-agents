"""Typeset the existing Markdown report using the pinned CVPR 2026 style.

Requires a TeX installation and uvx (Pandoc is acquired in an isolated tool
environment, not added to the scientific experiment dependencies).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _escape_caption(value: str) -> str:
    for old, new in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        value = value.replace(old, new)
    return value.replace("±", r"$\pm$").replace("|", r"\textbar{}")


def prepare_markdown(text: str) -> str:
    """Preserve report prose; give figures genuine LaTeX floats and captions."""
    text = text[text.index("## Abstract") :]
    text = text.replace("## Abstract", "# Abstract", 1)

    def figure(match: re.Match) -> str:
        path, caption = match.group(2), match.group(3)
        source = (ROOT / "report" / path).resolve()
        images = ROOT / "report" / "cvpr" / "images"
        images.mkdir(exist_ok=True)
        shutil.copyfile(source, images / source.name)
        caption = re.sub(r"^Figure \d+\. ", "", caption)
        return (
            "\\begin{figure}[tb]\n\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{images/{source.name}}}\n"
            f"\\caption{{{_escape_caption(caption)}}}\n\\end{{figure}}\n"
        )

    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)\n<em class="cap">(.*?)</em>', figure, text, flags=re.S)
    text = re.sub(r"^## \d+\. ", "# ", text, flags=re.M)
    text = re.sub(r"^### ", "## ", text, flags=re.M)
    return text


def prepare_latex(text: str) -> str:
    """Pandoc longtables cannot run in two-column mode; keep tables in a column."""
    text = text.replace(r"\section{Abstract}\label{abstract}", r"\begin{abstract}")
    text = text.replace(r"\section{Question}", "\\end{abstract}\n\n\\section{Question}", 1)
    pattern = r"\{\\def\\LTcaptype\{none\}.*?\\begin\{longtable\}\[\](.*?)\\end\{longtable\}\s*\}"

    def table(match: re.Match) -> str:
        body = match.group(1).replace(r"\endhead", "").replace(r"\endlastfoot", "")
        # Pandoc puts the footer before the data. Put the closing rule after it.
        body = body.replace(r"\bottomrule\noalign{}", "")
        return (
            "\\begin{center}\n\\small\n\\resizebox{\\columnwidth}{!}{%\n"
            f"\\begin{{tabular}}{body}\n\\bottomrule\n\\end{{tabular}}"
            "}\n\\end{center}"
        )

    return re.sub(pattern, table, text, flags=re.S)


def build_cvpr(text: str) -> None:
    folder = ROOT / "report" / "cvpr"
    rendered = folder / "rendered.md"
    rendered.write_text(prepare_markdown(text))
    subprocess.run(
        [
            "uvx",
            "--from",
            "pypandoc-binary==1.17",
            "python",
            "-c",
            "import pypandoc,sys; pypandoc.convert_file(sys.argv[1], 'latex', "
            "format='markdown+raw_tex', outputfile=sys.argv[2])",
            str(rendered),
            str(folder / "body.tex"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    body = folder / "body.tex"
    body.write_text(prepare_latex(body.read_text()))
    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=folder,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout[-6000:])
    shutil.copyfile(folder / "main.pdf", ROOT / "report" / "report.pdf")
    print(f"wrote {ROOT / 'report' / 'report.pdf'} using CVPR 2026")
