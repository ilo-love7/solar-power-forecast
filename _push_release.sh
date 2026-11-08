#!/usr/bin/env bash
# N3 推送 + tag + release：网络瞬断时长重试
set -o pipefail
export https_proxy=http://127.0.0.1:64061 http_proxy=http://127.0.0.1:64061
cd "$(dirname "$0")"
TOKEN=$(gh auth token)

push_with_retry() {
  local what="$1"; shift
  for i in $(seq 1 120); do
    echo "--- $what try $i $(date +%H:%M:%S) ---"
    if "$@" 2>&1 | tail -4; then
      echo "${what}_OK"
      return 0
    fi
    sleep 30
  done
  echo "${what}_FAIL"
  return 1
}

push_with_retry push_main git push -u origin main || exit 1
git tag -f v1.0.0 2>/dev/null
push_with_retry push_tag git push -f origin v1.0.0 || exit 1

for i in $(seq 1 120); do
  echo "--- release try $i $(date +%H:%M:%S) ---"
  code=$(curl -s -o /tmp/rel_resp.json -w "%{http_code}" -X POST \
    "https://api.github.com/repos/ilo-love7/solar-power-forecast/releases" \
    -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" \
    -d '{"tag_name":"v1.0.0","name":"v1.0.0 光伏功率预测与损失诊断","body":"N3 v1.0.0\n\n- 1h 提前量站级 AC 功率预测: LightGBM 最优 nRMSE 9.98%(P1)/11.87%(P2), 双口径 nowcast/weather\n- 逆变器诊断: P1 DC 量纲异常(x10.2)+2 台离群; P2 14 重点+8 告警(4 台通讯中断/重灾停机 272 槽)\n- 误差归因: 正午强光时段 RMSE 最大\n- 35 项单测全绿, run_all.sh 一键复现","draft":false,"prerelease":false}')
  echo "release http $code"
  if [ "$code" = "201" ]; then echo RELEASE_OK; break; fi
  sleep 30
done

export https_proxy=http://127.0.0.1:64061 http_proxy=http://127.0.0.1:64061
for i in 1 2 3 4 5; do
  out=$(git ls-remote origin 2>/dev/null | grep -E "main|v1.0.0")
  if [ -n "$out" ]; then echo "REMOTE_VERIFY:"; echo "$out"; exit 0; fi
  sleep 10
done
echo VERIFY_FAIL
exit 1
