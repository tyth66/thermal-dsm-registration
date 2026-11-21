# 当前算法基线冻结（candidate-2026-11-21）

状态：冻结默认结构、参数、评价集合与输出口径。  
代码基线：24c7af0（默认模型）；V3-A profiler：861be7c  
测试：121 passed

## 1. 冻结默认配置

| 项 | 值 |
|---|---|
| 位移域 | mode |
| ECC 掩膜 | joint |
| 观测 | masked ECC + prior + reverse + ambiguity/alias conflict |
| 场 | q-only similarity + Gaussian RBF |
| 正则 | coefficient ridge, lambda=0.1 |
| RBF 核长 | 自动 max(2.5*tile, 4*spacing_median) |
| 面板映射 | rigid |
| Sigma_i 加权 | 关闭 |
| separate mask | 关闭 |
| usable-only domain | 关闭 |
| joint 域 | 关闭 |
| P2 正交化/软惩罚/KRR | 关闭 |
| 交付口径 | conservative_valid |

## 2. 冻结评价集合

站点：

- 141
- 247
- 290

13 个空间方案：

- pca4
- grid3
- grid4
- edge_ring
- edge_N / edge_S / edge_E / edge_W
- edge_half
- corner4
- checker
- random25
- local2x2

共 39 格。所有版本必须使用同一批方案、同一站点、同一 tile/step 和同一 accepted 规则，否则数字不可直接比较。

## 3. 冻结内部指标

| 站 | panels | corrected_valid | conservative_valid | low_confidence | observations accepted/records | 13 方案平均误差 |
|---|---:|---:|---:|---:|---:|---:|
| 141 | 5285 | 5267 | 4971 | 323 | 89/105 | 0.2094 m |
| 247 | 490 | 490 | 490 | 0 | 39/59 | 0.5017 m |
| 290 | 1576 | 1576 | 1576 | 0 | 31/37 | 0.1956 m |

观测空间分布：

- 141：raw 108；domain 保留 105；ambiguous 16；accepted 89；
- 247：raw 101；domain 保留 59；ambiguous 20；accepted 39；
- 290：raw 37；domain 保留 37；ambiguous 6；accepted 31。

247 轮廓 Gate 0：

| 场 | 均值 | P95 |
|---|---:|---:|
| MAD | 0.280 m | 0.822 m |
| mode | 0.261 m | 0.707 m |
| joint | 0.273 m | 0.703 m |

## 4. 冻结输出定义

逐点：

conservative_valid = source_valid AND field_supported AND target_valid

组件：

component_conservative_valid =
center.conservative_valid
AND
sum(corner.conservative_valid) >= 2

摘要：

- panel_points_conservative_valid
- panel_ids_conservative_valid

兼容字段：

valid = component_source

valid 不得作为交付验收口径。

## 5. 回归基线

命令：

    cd analysis/photo-bridge
    PYTHONDONTWRITEBYTECODE=1 python3 -m pytest registration_algorithms/tests -q

期望：

    119 passed

评测脚本：

- tools/qa_candidate_baseline.py（仓库内，13 方案，输出 JSON）
- 参考输出：outputs/final_candidate/baseline_eval.json（本地生成，outputs 路径可能被忽略）

命令：

    python3 tools/qa_candidate_baseline.py \
      --station 141:VISIBLE_DSM:THERMAL_DSM \
      --station 247:VISIBLE_DSM:THERMAL_DSM \
      --station 290:VISIBLE_DSM:THERMAL_DSM \
      --out outputs/final_candidate/baseline_eval.json

期望均值：

- 141: 0.2094 m，accepted 89；
- 247: 0.5017 m，accepted 39；
- 290: 0.1956 m，accepted 31。

## 6. 能力边界

明确不能声称：

- coherent alias 模式下选对；
- 绝对地理定位精度；
- 141/290 内部几何正确；
- V5 原片投影精度；
- 内部误差改善等于外部几何改善。

## 6.1 V3-A 只读诊断

- tools/profile_site_geometry.py 已加入，不参与默认参数；
- 首轮结果：V3A_SITE_GEOMETRY_PROFILER.md；
- profiler 只读，不改变本冻结基线；
- 任何自适应参数必须走独立 A/B，不修改 V2 默认。

## 7. 冻结规则

1. 不改默认场形状；
2. 不调 L/lambda 追内部指标；
3. 新方案必须 opt-in，并与本基线使用同一评价集合；
4. 新方案通过 A/B 后才允许讨论默认切换；
5. P3 锚点与 V5 相机链恢复前，不改变保守候选定位。
