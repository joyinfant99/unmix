#!/bin/bash
# Start Unmix as a native window (server + UI in one process)
cd "$(dirname "$0")"
exec .venv/bin/python desktop.py
