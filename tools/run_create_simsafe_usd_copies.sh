#!/usr/bin/env bash
set -euo pipefail

USD_LIB_DIR="$(find /home/zyp/isaacsim/extscache -maxdepth 1 -type d -name 'omni.usd.libs-*' | sort | tail -1)"
if [[ -z "${USD_LIB_DIR}" ]]; then
  echo "Could not find Isaac Sim omni.usd.libs under /home/zyp/isaacsim/extscache" >&2
  exit 1
fi

export PYTHONPATH="${USD_LIB_DIR}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${USD_LIB_DIR}/bin:${LD_LIBRARY_PATH:-}"

exec /home/zyp/isaacsim/kit/python/bin/python3 \
  /home/zyp/GraspGen/tools/create_simsafe_usd_copies.py "$@"
