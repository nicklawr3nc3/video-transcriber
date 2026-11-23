#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/fallenfang/video-transcriber"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/transcribe-$(date +%Y%m%d).log"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

{
  echo "=== $(date -Iseconds) starting transcribe ==="
  "${PROJECT_DIR}/.venv/bin/python" transcribe.py -l
  echo "=== $(date -Iseconds) finished transcribe ==="
} >> "${LOG_FILE}" 2>&1
