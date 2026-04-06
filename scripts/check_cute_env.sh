#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Repository: ${ROOT_DIR}"
echo "nvcc: $(which nvcc)"
nvcc --version | tail -n 1
echo

echo "Detected GPU:"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
echo

echo "Cute-Learning make info:"
make -C "${ROOT_DIR}/gemm/gemm_v1" info
