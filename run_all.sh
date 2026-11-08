#!/usr/bin/env bash
# N3 全流程：00 校验 → 01 EDA → 02 基线/GBDT → 03 LSTM → 04 诊断 → 05 归因
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-D:/项目/venvs/flsoh-py312/Scripts/python.exe}"
for s in 00_verify_data 01_eda 02_baselines 03_train_lstm 04_inverter_diagnosis 05_error_attribution; do
  echo "======== $s ========"
  "$PY" "scripts/${s}.py"
done
echo "全部完成。结果见 results/ 与 figures/"
