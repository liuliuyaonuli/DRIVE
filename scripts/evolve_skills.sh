#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <site-or-library-path> [extra manage_drive_library.py args...]" >&2
  exit 2
fi

site="$1"
shift
python tools/site_specific/manage_drive_library.py --site "$site" "$@"
