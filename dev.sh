#!/bin/bash
# Dev mode: run Unmix from source, no build and no install. Restarts the window when a .py file changes.
# Edits to index.html need no restart: just reload the page (Cmd+R in the window).
# Uses port 8799 and ./data, so it never touches an installed Unmix (port 8765, Application Support).
# Pass "plain" to run once without auto-restart.
cd "$(dirname "$0")"
export UNMIX_PORT="${UNMIX_PORT:-8799}"
if [ "$1" = "plain" ]; then exec .venv/bin/python desktop.py; fi
exec .venv/bin/watchfiles --filter python --grace-period 1 ".venv/bin/python desktop.py" app.py desktop.py openq.py
