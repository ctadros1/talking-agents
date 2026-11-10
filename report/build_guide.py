"""Build docs/guide/how-it-works.pdf, the learning guide, from docs/guide/how-it-works.md.

Result numbers in the guide are `{{placeholders}}` filled from the same results files as the report
(see report/build_report.py), so the guide and the report always agree.

Usage: uv run python report/guide_figures.py && uv run python report/build_guide.py
"""

from __future__ import annotations

from build_report import ROOT, load_data, render, write_pdf

GUIDE = ROOT / "docs" / "guide"
CSS = """
@page { size: letter; margin: 0.75in 0.8in; }
body { font: 10.4pt/1.52 "Helvetica Neue", Helvetica, Arial, sans-serif; color: #1f2933; }
h1 { font-size: 30pt; color: #102a43; margin: 1.6in 0 6pt; letter-spacing: -0.5pt; }
.subtitle { font-size: 13pt; color: #486581; margin: 0 0 28pt; }
.toc { border-top: 2px solid #bcccdc; border-bottom: 2px solid #bcccdc; padding: 8pt 0 4pt; margin-bottom: 18pt; }
.toc p { margin: 0 0 4pt; color: #102a43; }
.toc ol { margin: 0 0 6pt; padding-left: 20pt; } .toc li { margin: 2pt 0; }
h2 { break-before: page; font-size: 18pt; color: #102a43; border-bottom: 2px solid #bcccdc; padding-bottom: 4pt;
     margin: 0 0 10pt; }
h3 { font-size: 12.5pt; color: #243b53; margin: 16pt 0 5pt; break-after: avoid; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 8pt; padding-left: 18pt; } li { margin: 0 0 3pt; }
strong { color: #102a43; }
code { font: 8.8pt/1.4 Menlo, "SF Mono", monospace; background: #f0f4f8; padding: 0.5pt 3pt; border-radius: 3pt; }
pre { background: #f0f4f8; border: 1px solid #d9e2ec; border-radius: 6pt; padding: 8pt 10pt; margin: 6pt 0 10pt;
      white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.4pt; }
table { width: 100%; border-collapse: collapse; font-size: 9.2pt; margin: 6pt 0 11pt; break-inside: avoid; }
th { background: #102a43; color: #fff; text-align: left; font-weight: 600; }
th, td { border: 0.6pt solid #bcccdc; padding: 3.5pt 6pt; vertical-align: top; }
tr:nth-child(even) td { background: #f5f7fa; }
blockquote { margin: 8pt 0 12pt; padding: 8pt 12pt; background: #fffbea; border-left: 4px solid #f0b429;
             border-radius: 4pt; break-inside: avoid; }
blockquote p { margin: 0; }
img { display: block; margin: 8pt auto 3pt; max-width: 100%; break-inside: avoid; }
svg { display: block; width: 100%; height: auto; margin: 8pt 0 10pt; break-inside: avoid; }
p.cap { font-size: 8.8pt; color: #52606d; text-align: center; margin: 0 auto 12pt; max-width: 88%; }
"""


def main() -> None:
    text = render((GUIDE / "how-it-works.md").read_text(), load_data())
    write_pdf(text, CSS, GUIDE / "how-it-works.html", GUIDE / "how-it-works.pdf")


if __name__ == "__main__":
    main()
