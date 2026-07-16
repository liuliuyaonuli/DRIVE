#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <site> <trajectory-dir> [output-dir] [pipeline-args...]" >&2
  echo "sites: reddit, gitlab, shopping, shopping_admin, map" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE="$1"
TRAJECTORY_DIR="$2"
shift 2
OUTPUT_DIR="$ROOT_DIR/skills/$SITE"
if [[ $# -gt 0 && "$1" != --* ]]; then
  OUTPUT_DIR="$1"
  shift
fi

if [[ -f "$ROOT_DIR/.env.local" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env.local"
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"
exec "${PYTHON:-python3}" tools/site_specific/site_skill_pipeline.py \
  --site "$SITE" \
  --trajectories "$TRAJECTORY_DIR" \
  --out-dir "$OUTPUT_DIR" \
  --model "${LLM_MODEL:-gpt-4.1}" \
  "$@"
