#!/usr/bin/env bash
set -eo pipefail

SDK_ROOT="${ASTRIBOT_SDK_ROOT:-/home/astribot/Downloads/astribot_sdk_aarch64}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "${SDK_ROOT}/env.sh" ]; then
    echo "[start_async] SDK not found: ${SDK_ROOT}" >&2
    exit 1
fi

set +u
source "${SDK_ROOT}/env.sh"
set -u
export ASTRIBOT_SDK_ROOT="${SDK_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Defaults before "$@" so explicit CLI flags can override.
exec python3 "${REPO_ROOT}/run_async_deploy.py" \
    --chunk-hz 30 \
    "$@"
