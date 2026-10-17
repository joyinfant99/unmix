#!/bin/bash
# Usage: ./release.sh 1.2.0 "what changed"
# Builds the installer, uploads it to the Band OS download storage (behind the email-code gate on
# bandosapp.com/unmix), and tags the source on GitHub. Installed copies of Unmix see the new version
# and offer "Update and restart".
#
# Needs the BandOS backend checkout (for the publish script and its R2 + UNMIX_SECRET settings):
#   BANDOS_BACKEND=~/path/to/BandOS/Production/backend   (default below)
set -euo pipefail
cd "$(dirname "$0")"
V="${1:?version, e.g. 1.2.0}"; NOTES="${2:-Unmix $V}"
BACKEND="${BANDOS_BACKEND:-$HOME/Desktop/COLLECTIONS/CODING/LIVE/BandOS/Production/backend}"
[ -f "$BACKEND/scripts/publish-unmix-release.js" ] || { echo "BandOS backend not found at $BACKEND (set BANDOS_BACKEND)"; exit 1; }

./build_app.sh "$V"
node "$BACKEND/scripts/publish-unmix-release.js" "dist/Unmix-$V.pkg" "$V" "$NOTES"

git add -A && git commit -m "Release $V" && git tag "v$V"
git push origin HEAD:main "v$V"
gh release create "v$V" --title "Unmix $V" --notes "$NOTES

The installer is on https://bandosapp.com/unmix (free, confirm your email to download). Installed copies update themselves."
