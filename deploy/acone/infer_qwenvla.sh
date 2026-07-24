#!/usr/bin/env bash
# Compatibility alias for the old development-script name.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/infer_internvla_a1_5.sh" "$@"
