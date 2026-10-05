# Delay-plan compiler service

把超声探头逐阵元标定得到的整数目标延迟，编译为固件可容纳的少量**整数斜坡**
（相邻差序列的极大相等段），避免逐点取整产生超出硬件换挡能力的跳变。

- 零第三方依赖：仅使用 Python 3.11 标准库。
- 全部裁决使用整数算术；优化严格按
  **最大绝对误差 → 总绝对误差 → 实际斜坡数 → 延迟序列字典序** 逐级最小化。
- 锚点必须精确命中；冲突时稳定返回 `422` 与冲突区间，不输出任何部分延迟表。

## 接口

`POST /api/delay-plans/compile`

```json
{
  "targets": [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
  "delay_min": 0,
  "delay_max": 100,
  "max_step": 4,
  "max_ramps": 4,
  "anchors": [
    {"index": 0, "value": 10},
    {"index": 8, "value": 26},
    {"index": 15, "value": 40}
  ]
}
```

| 字段 | 约束 |
| --- | --- |
| `targets` | 12–48 个整数，每个阵元一个目标延迟 |
| `delay_min` / `delay_max` | 全局延迟闭区间（整数，含端点） |
| `max_step` | 相邻阵元延迟差绝对值上限（非负整数） |
| `max_ramps` | 最多斜坡数（1…n−1） |
| `anchors` | 2–8 个必须精确命中的阵元 `{index, value}` |

成功响应（200）：

```json
{
  "status": "ok",
  "plan": {
    "n": 16,
    "delays": [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
    "errors": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ramps": [
      {"start": 0, "end": 15, "delta": 2}
    ],
    "ramp_count": 1,
    "max_abs_error": 0,
    "total_abs_error": 0
  }
}
```

- `errors[i] = delays[i] - targets[i]`（带符号整数）。
- `ramps` 给出每个极大相等差段的起止阵元（含端点）与该段整数差值，
  各段首尾相接且相邻段 `delta` 不同；`ramp_count` 即实际斜坡数。

不可行响应（422，不含 `delays`/部分表）：

```json
{
  "error": "infeasible",
  "message": "no feasible delay plan: anchor/step conflict",
  "conflicts": [
    {"kind": "step_unreachable", "start": 0, "end": 15,
     "from_value": 10, "to_value": 40, "steps": 15,
     "required_min_step": 2, "max_step": 1,
     "min_total_change": 30, "max_total_change": 15}
  ]
}
```

冲突类型：

- `anchor_out_of_bounds`：锚点值落在全局闭区间外（`start == end` 为该锚点）。
- `step_unreachable`：相邻锚点在给定距离与 `max_step` 下不可达，
  `[start, end]` 为这对锚点的阵元区间。
- `empty_band`：区间/步长传播后某阵元无可行整数值。
- `ramp_budget`：可达性满足但任何可行序列的斜坡数都超过 `max_ramps`，
  同时返回各锚点段信息便于定位。

请求格式错误返回 400；JSON 非法返回 400；未知路由 404；错误方法 405。

## 算法

1. 结构检查：从锚点向两侧做步长锥传播，得到每个阵元的整数可行带，
   并定位锚点/区间/步长冲突区间。
2. 最小最大误差：对误差预算 E 做二分；可行性是
   `(阵元位置, 取值, 上一条边差值)` 上的动态规划，状态值为最少斜坡数；
   利用每层每个取值的最优/次优前驱把转移降为 O(W·(2·max_step+1))。
3. 固定最优 E 后，带“剩余斜坡预算”维的后向 DP 计算
   `(后缀总绝对误差, 后缀新增斜坡数)`，再从左到右贪心恢复，
   得到总误差、斜坡数最优前提下的字典序最小序列。

## 本地运行（无需 Docker）

```bash
python3 -m unittest discover -s tests -v   # 25 个测试（含 440+ 暴力枚举对照）
API_PORT=8080 python3 -m app.server         # 启动服务
curl -s localhost:8080/healthz
```

一键汇总校验（代码测试 + 构建产物清单 + API 冒烟，退出码为失败分组数）：

```bash
python3 scripts/make_build_manifest.py
python3 scripts/verify.py
```

## Docker / Docker Compose

```bash
# 构建并启动带健康检查的服务（主机端口可配置）
API_PORT=9090 docker compose up -d --build api
curl -s localhost:9090/healthz

# 一次性 verify 服务：等待 api 健康后，汇总
# 代码测试、镜像构建产物哈希与端到端 API 冒烟结果，以退出码裁决
docker compose --profile verify run --rm verify
```

`verify` 服务与 `api` 使用同一镜像；镜像构建时
`scripts/make_build_manifest.py` 会把全部代码/测试/脚本文件的 SHA-256
写入 `build-artifacts/manifest.json`，verify 时逐项复核，确保镜像内构建产物
与被测代码一致。
