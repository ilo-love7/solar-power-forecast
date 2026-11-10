#!/usr/bin/env bash
# N3 推送 + 双 tag + 双 release：网络瞬断长重试（v1.0.0 + v1.1.0）
set -o pipefail
export https_proxy=http://127.0.0.1:64061 http_proxy=http://127.0.0.1:64061
cd "$(dirname "$0")"
TOKEN=$(gh auth token)

retry() {
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

retry push_main git push -u origin main || exit 1
retry push_tags git push -f origin v1.0.0 v1.1.0 || exit 1

release() {
  local tag="$1" name="$2" body="$3"
  for i in $(seq 1 60); do
    echo "--- release $tag try $i $(date +%H:%M:%S) ---"
    code=$(curl -s -o /tmp/rel_${tag}.json -w "%{http_code}" -X POST \
      "https://api.github.com/repos/ilo-love7/solar-power-forecast/releases" \
      -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" \
      -d "{\"tag_name\":\"$tag\",\"name\":\"$name\",\"body\":\"$body\",\"draft\":false,\"prerelease\":false}")
    echo "release $tag http $code"
    if [ "$code" = "201" ] || [ "$code" = "422" ]; then echo "RELEASE_${tag}_OK"; return 0; fi
    sleep 30
  done
  return 1
}

release v1.0.0 "v1.0.0 光伏功率预测与损失诊断" "N3 v1.0.0\n\n- 1h 提前量站级 AC 功率预测: LightGBM 最优 nRMSE 9.98%(P1)/11.87%(P2), 双口径 nowcast/weather\n- 逆变器诊断: P1 DC 量纲异常(x10.2)+2 台离群; P2 14 重点+8 告警(4 台通讯中断/重灾停机 272 槽)\n- 误差归因: 正午强光时段 RMSE 最大\n- 35 项单测全绿" || true

release v1.1.0 "v1.1.0 多提前量/跨站迁移/分位数预测" "N3 v1.1.0\n\n- 多提前量: LightGBM 15min->4h nRMSE 8.4%->11.3%, 持久性 36% 且 R2 翻负\n- 跨站迁移: 容量归一化 ZeroShot 16.0%->12.1%; 7 天少样本 P1->P2 追平域内\n- 分位数: q10-q90 覆盖 88.5%(名义 80%) 免校准; 强光桶覆盖塌到 45%\n- 修正 SlotMean 基线泄漏; 37 项单测全绿" || true

export https_proxy=http://127.0.0.1:64061 http_proxy=http://127.0.0.1:64061
for i in 1 2 3 4 5; do
  out=$(git ls-remote origin 2>/dev/null | grep -E "main|v1\.0\.0|v1\.1\.0")
  if [ -n "$out" ]; then echo "REMOTE_VERIFY:"; echo "$out"; exit 0; fi
  sleep 10
done
echo VERIFY_FAIL
exit 1
