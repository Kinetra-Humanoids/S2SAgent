#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV

uv run streamlit run app/configurator.py
uv run python app/s2s_no_ros.py --config config.toml
