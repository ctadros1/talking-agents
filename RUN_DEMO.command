#!/bin/zsh
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  print "Please install uv first: https://docs.astral.sh/uv/"
  read "?Press Return to close. "
  exit 1
fi
uv sync --frozen
uv run python scripts/demo_inference.py --phase both
print ""
print "Fresh inference finished. Open the new viewer files at the paths printed above."
read "?Press Return to close. "
