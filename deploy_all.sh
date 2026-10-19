#!/bin/bash
# Usage: ./deploy_all.sh 1.2.0 "what changed"
# Deploys the BandOS backend (Fly), frontend and admin (Vercel), then publishes the Unmix release.
# Order matters: the backend must accept the new app requests before the app is released.
# Skip steps with SKIP_BACKEND=1, SKIP_FRONTEND=1, SKIP_ADMIN=1 (e.g. one that already succeeded).
# The database migration (027_unmix_installs.sql) must already be applied.
set -euo pipefail
cd "$(dirname "$0")"
V="${1:?version, e.g. 1.2.0}"; NOTES="${2:-Unmix $V}"
BANDOS="${BANDOS_ROOT:-$(cd .. && pwd)/Production}"

if [ -z "${SKIP_BACKEND:-}" ]; then
  echo "==> Backend (Fly)"
  (cd "$BANDOS/backend" && fly deploy -a bandos-backend)
fi

if [ -z "${SKIP_FRONTEND:-}" ]; then
  echo "==> Frontend (Vercel)"
  (cd "$BANDOS/frontend" && vercel --prod)
fi

if [ -z "${SKIP_ADMIN:-}" ]; then
  echo "==> Admin (Vercel)"
  (cd "$BANDOS/admin" && vercel --prod)
fi

echo "==> Unmix release $V"
./release.sh "$V" "$NOTES"

echo "Done."
