#!/usr/bin/env bash
# Regenerate Figures 5.1-5.3 from the recorded experiment runs and copy them
# to every folder that holds a set, so no stale version is left behind.
#
#   ./refresh_figures.sh
#
set -euo pipefail

PROJECT="${PROJECT_DIR:-$HOME/PycharmProjects/finsession-rag}"
FIGURES="$PROJECT/figures"

# Extra folders that keep their own copy of the figures. Add or remove lines.
MIRRORS=(
  "$HOME/Documents/MSC Dissertation/finsession-rag/supporting-files"
)

cd "$PROJECT"

echo "==> Regenerating figures from results/"
python3 scripts/make_figures.py

echo
echo "==> Mirroring to other folders"
for dir in "${MIRRORS[@]}"; do
  if [ -d "$dir" ]; then
    cp "$FIGURES"/fig5_*.png "$FIGURES"/fig5_*.pdf "$dir"/
    echo "    updated: $dir"
  else
    echo "    skipped (not found): $dir"
  fi
done

echo
echo "==> Current figures"
ls -lh "$FIGURES"/fig5_* | awk '{print "   ", $9, $5, $6, $7, $8}'
echo
echo "Done. Re-insert these into the .docx - Word will not refresh them on its own."
