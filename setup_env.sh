#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if command -v uv >/dev/null 2>&1; then
  uv sync
else
  echo "uv is required. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi
