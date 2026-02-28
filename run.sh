#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME=${1:-charge_discharge}

case "$SCRIPT_NAME" in
  charge_discharge)
    python src/charge_discharge.py
    ;;
  online_learning)
    python src/current_integrated_online_learning.py
    ;;
  *)
    echo "Usage: $0 {charge_discharge|online_learning}"
    exit 1
    ;;
esac
