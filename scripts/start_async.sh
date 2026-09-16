#!/usr/bin/env bash
set -eo pipefail

# This script must remain Bash: the SDK env.sh uses BASH_SOURCE[0].
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_ROOT="${ASTRIBOT_SDK_ROOT:-/home/astribot/Downloads/astribot_sdk_aarch64}"

if [ ! -f "${SDK_ROOT}/env.sh" ]; then
    echo "[start_async] SDK not found: ${SDK_ROOT}" >&2
    exit 1
fi

# Prevent a previously sourced SDK or another workspace from leaking in.
unset ASTRIBOT_SDK_ROOT
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset FASTRTPS_DEFAULT_PROFILES_FILE

# Load the SDK only inside Bash; env.sh relies on BASH_SOURCE[0].
set +e
set +u
source "${SDK_ROOT}/env.sh"
env_status=$?
set -u
set -e
if [ "$env_status" -ne 0 ]; then
    echo "[start_async][WARN] SDK env.sh returned ${env_status}; continuing after environment setup." >&2
fi

export ASTRIBOT_SDK_ROOT="${SDK_ROOT}"
export ROS_DISTRO="${ROS_DISTRO:-humble}"
# tf_transformations is provided by the system ROS Humble installation.
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:${REPO_ROOT}:${PYTHONPATH:-}"

# Prefer the active Conda environment; allow an explicit override.
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

# Defaults before "$@" so explicit CLI flags can override.
exec "${PYTHON_BIN}" "${REPO_ROOT}/run_async_deploy.py" \
    --chunk-hz 30 \
    "$@"
