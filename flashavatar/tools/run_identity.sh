#!/usr/bin/env bash
# usage instructions by LLM:
#
# Pre-reqs (do once per identity, before calling this script):
#   1. dataset/<id>/imgs/00000.png ... 0NNNN.png        (frames)
#   2. dataset/<id>/alpha/00000.png ... 0NNNN.png       (RVM matting)
#   3. dataset/<id>/parsing/...                         (BiSeNet, optional)
#   4. metrical-tracker/output/<id>/checkpoint/*.frame  (run package.py)
#
# Usage:
#   tools/run_identity.sh <idname>                             # defaults
#   tools/run_identity.sh <idname> 0.65 25000                 # custom downscale + iters
#   tools/run_identity.sh <idname> 0.5 20000 both 0           # frame_delta=0 (CREMA-D)
#   tools/run_identity.sh <idname> 0.5 20000 train_only 0     # train only, CREMA-D
#   tools/run_identity.sh <idname> 0.5 20000 test_only  0     # test only,  CREMA-D

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <idname> [downscale] [full_iters] [train_only|test_only|both] [frame_delta]" >&2
  exit 1
fi

ID="$1"
DOWNSCALE="${2:-0.5}"
FULL_ITERS="${3:-20000}"
MODE="${4:-both}"
FRAME_DELTA="${5:-1}"

DATA_DIR="dataset/${ID}"
MICA_DIR="metrical-tracker/output/${ID}/checkpoint"
CKPT_DIR="${DATA_DIR}/log/ckpt"

if [[ ! -d "${DATA_DIR}/imgs" ]]; then
  echo "Missing ${DATA_DIR}/imgs — extract frames first." >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}/alpha" ]]; then
  echo "Missing ${DATA_DIR}/alpha — run RVM first." >&2
  exit 1
fi
if [[ ! -d "${MICA_DIR}" || -z "$(ls -A "${MICA_DIR}" 2>/dev/null || true)" ]]; then
  echo "Missing ${MICA_DIR} — run package.py on your SMIRK output first:" >&2
  echo "  python package.py --idname ${ID} --pt_file <path/to/track_params.pt>" >&2
  exit 1
fi

if [[ "${MODE}" != "test_only" ]]; then
  echo "[run_identity] training ${ID} (downscale=${DOWNSCALE}, full_iters=${FULL_ITERS})"
  python train.py \
    --idname "${ID}" \
    --downscale "${DOWNSCALE}" \
    --full_iters "${FULL_ITERS}" \
    --frame_delta "${FRAME_DELTA}" \
    --camera_z 1.0 --force_camera_z
fi

if [[ "${MODE}" != "train_only" ]]; then
  if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "No checkpoints in ${CKPT_DIR}" >&2
    exit 1
  fi
  LATEST="$(ls -t "${CKPT_DIR}"/chkpnt*.pth 2>/dev/null | head -n1 || true)"
  if [[ -z "${LATEST}" ]]; then
    echo "No chkpnt*.pth files in ${CKPT_DIR}" >&2
    exit 1
  fi
  echo "[run_identity] rendering with ${LATEST}"
  RENDER_ALL_FLAG=""
  if [[ "${FRAME_DELTA}" == "0" ]]; then
    RENDER_ALL_FLAG="--render_all"
  fi
  python test.py \
    --idname "${ID}" \
    --downscale "${DOWNSCALE}" \
    --frame_delta "${FRAME_DELTA}" \
    --camera_z 1.0 --force_camera_z \
    ${RENDER_ALL_FLAG} \
    --checkpoint "${LATEST}"

  AVI="${DATA_DIR}/log/test.avi"
  MP4="${DATA_DIR}/log/test.mp4"
  if [[ -f "${AVI}" ]] && command -v ffmpeg >/dev/null 2>&1; then
    echo "[run_identity] re-encoding to mp4"
    ffmpeg -y -loglevel error -i "${AVI}" -c:v libx264 -crf 17 -preset slow \
           -pix_fmt yuv420p "${MP4}"
    echo "[run_identity] done -> ${MP4}"
  else
    echo "[run_identity] done -> ${AVI}"
  fi
fi
