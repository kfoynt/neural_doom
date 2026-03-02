#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAD_PATH="${SCRIPT_DIR}/DOOM.WAD"
MOD_PATH="${SCRIPT_DIR}/enemy_nn_backend_mod.pk3"

if [[ ! -f "${WAD_PATH}" ]]; then
  echo "DOOM.WAD not found at: ${WAD_PATH}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
python3 build_enemy_nn_mod.py

python3 e1m1_transformer_backend.py \
  --wad "${WAD_PATH}" \
  --map E1M1 \
  --resolution 1280x960 \
  --keyboard-source pygame_window \
  --enemy-backend-transformer \
  --enemy-backend-mod "${MOD_PATH}" \
  --enemy-slots 16 \
  --enemy-kinematics-transformer \
  --enemy-combat-transformer \
  --nn-world-sim \
  --nn-world-sim-strict \
  --nn-world-sim-pure \
  --log-interval 1 \
  "$@"
