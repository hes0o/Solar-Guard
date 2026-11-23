#!/bin/sh
# Start the SolarGuard website, loading settings from .env if present.
cd "$(dirname "$0")/.." || exit 1
set -a; [ -f .env ] && . ./.env; set +a
exec .venv/bin/uvicorn solarguard.web:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
