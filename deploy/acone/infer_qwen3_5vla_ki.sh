#!/usr/bin/env bash
# Compatibility alias for the former qwen3_5vla_ki deployment name.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/infer_internvla_a1_5.sh" "$@"
