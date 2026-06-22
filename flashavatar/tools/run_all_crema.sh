#!/usr/bin/env bash

#   bash tools/run_all_crema.sh

set -euo pipefail

DOWNSCALE="${1:-0.65}"
FULL_ITERS="${2:-25000}"

CLIPS=(
  "dataset/1001/1001_DFA_NEU_XX 1001_DFA_NEU_XX"
  "dataset/1001/1001_DFA_SAD_XX 1001_DFA_SAD_XX"
  "dataset/1001/1001_IEO_ANG_HI 1001_IEO_ANG_HI"
  "dataset/1001/1001_IEO_HAP_HI 1001_IEO_HAP_HI"
  "dataset/1001/1001_IEO_HAP_LO 1001_IEO_HAP_LO"
  "dataset/1005/1005_DFA_ANG_XX 1005_DFA_ANG_XX"
  "dataset/1005/1005_DFA_DIS_XX 1005_DFA_DIS_XX"
  "dataset/1005/1005_DFA_FEA_XX 1005_DFA_FEA_XX"
  "dataset/1005/1005_DFA_HAP_XX 1005_DFA_HAP_XX"
  "dataset/1005/1005_DFA_NEU_XX 1005_DFA_NEU_XX"
  "dataset/1005/1005_DFA_SAD_XX 1005_DFA_SAD_XX"
  "dataset/1005/1005_IEO_ANG_HI 1005_IEO_ANG_HI"
  "dataset/1005/1005_IEO_HAP_HI 1005_IEO_HAP_HI"
  "dataset/1005/1005_IEO_HAP_LO 1005_IEO_HAP_LO"
  "dataset/1010/1010_DFA_ANG_XX 1010_DFA_ANG_XX"
  "dataset/1010/1010_DFA_DIS_XX 1010_DFA_DIS_XX"
  "dataset/1010/1010_DFA_FEA_XX 1010_DFA_FEA_XX"
  "dataset/1010/1010_DFA_HAP_XX 1010_DFA_HAP_XX"
  "dataset/1010/1010_DFA_NEU_XX 1010_DFA_NEU_XX"
  "dataset/1010/1010_DFA_SAD_XX 1010_DFA_SAD_XX"
  "dataset/1010/1010_IEO_ANG_HI 1010_IEO_ANG_HI"
  "dataset/1010/1010_IEO_HAP_HI 1010_IEO_HAP_HI"
  "dataset/1010/1010_IEO_HAP_LO 1010_IEO_HAP_LO"
)

TOTAL=${#CLIPS[@]}
DONE=0
SKIP=0

for entry in "${CLIPS[@]}"; do
  clip_dir="${entry%% *}"
  idname="${entry##* }"

  if ls "dataset/${idname}/log/ckpt/"chkpnt*.pth 2>/dev/null | grep -q .; then
    echo "[run_all] SKIP ${idname} (already trained)"
    ((SKIP++)) || true
    continue
  fi

  echo ""
  echo "============================================================"
  echo "[run_all] ($((DONE+SKIP+1))/${TOTAL}) ${idname}"
  echo "============================================================"

  # .frame
  if ! ls "metrical-tracker/output/${idname}/checkpoint/"*.frame 2>/dev/null | grep -q .; then
    echo "[run_all] preparing ${idname} ..."
    python tools/prepare_crema_clip.py \
      --clip_dir "${clip_dir}" \
      --idname "${idname}"
  else
    echo "[run_all] ${idname} already prepared — skipping prepare step"
  fi

  # train amd render
  tools/run_identity.sh "${idname}" "${DOWNSCALE}" "${FULL_ITERS}" both 0

  ((DONE++)) || true
done

echo ""
echo "[run_all] Finished. Processed=${DONE}, Skipped=${SKIP}/${TOTAL}"
