#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_CUTLASS="${ROOT_DIR}/flashdecoding/src/cutlass"

echo "Repository: ${ROOT_DIR}"
echo "nvcc: $(which nvcc)"
nvcc --version | tail -n 1
echo

echo "Detected GPU:"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
echo

echo "Repo-local CUTLASS:"
if [ -f "${LOCAL_CUTLASS}/include/cute/tensor.hpp" ]; then
  echo "OK: ${LOCAL_CUTLASS}"
else
  echo "MISSING: ${LOCAL_CUTLASS}"
  echo "Run: git submodule update --init --recursive flashdecoding/src/cutlass"
fi
echo

echo "Cute-Learning make info:"
make -C "${ROOT_DIR}/gemm/gemm_v1" info
