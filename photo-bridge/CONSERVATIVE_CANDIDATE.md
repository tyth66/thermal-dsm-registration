# 保守候选模型交付说明

状态：按用户决定，接受“保守候选”，停止继续调场模型。  
代码基线：24c7af0  
测试：127 passed  
数学模型审阅：MATH_MODEL_AUDIT.md  
V3-B 247 实验：V3B_247_OBSERVATION_PARAMETERS.md  
V3-B 固定评价：V3B_FIXED_EVALUATION_247.md  
冻结基线：VERSION_FREEZE.md
日期：2026-11-21

## 1. 定位

这是一个**内部验证充分、可复现、带保守有效性口径的候选模型**，不是几何精度可验收的生产模型。

它回答：
- 在当前数据与重叠区上，映射是否稳定、可复现；
- 哪些面板有源/场/目标支撑；
- 哪些结果应标低置信或不可交付。

它不回答：
- 绝对地理精度；
- 247 的周期模式是否选对；
- 141/290 内部几何是否正确；
- V5 原片投影精度。

## 2. 冻结配置

默认：

- 位移域：domain_mode=mode；
- 观测：P0 masked ECC + prior + 反向一致性 + 歧义/alias 冲突；
- 场：P0 q-only，coefficient ridge；
- L：自动；
- ridge：0.1；
- P1 Σ_i 加权：关闭；
- P2 正交化/软惩罚/KRR：关闭；
- joint 2D 模式：opt-in，不作为默认。

## 3. 交付口径

面板行同时提供：

- valid：兼容字段，表示组件至少有源 alpha；
- conservative_valid：逐点 source_valid AND field_supported AND target_valid；
- component_conservative_valid：中心点 conservative_valid AND 至少 2 个角点 conservative_valid；
- low_confidence：场/目标不满足时的低置信标志。

摘要提供：

- panel_points_conservative_valid；
- panel_ids_conservative_valid。

**交付时必须使用 conservative_valid，不得使用 valid 作为验收口径。**

247 验证：

- panel_points 2450；
- conservative 2450；
- panel_ids 490；
- conservative 490。

说明 247 当前没有被保守口径额外剔除；这不代表几何正确，只说明没有 target nodata/field unsupported。

## 4. 已验证内容

内部留出（mode 默认）：

| 站 | 平均误差 | accepted |
|---|---|---|
| 141 | 0.2094 m | 89 |
| 247 | 0.5017 m | 39 |
| 290 | 0.1956 m | 31 |

247 轮廓 Gate 0：

| 场 | 均值 | P95 |
|---|---|---|
| MAD | 0.280 m | 0.822 m |
| mode | 0.261 m | 0.707 m |
| joint | 0.273 m | 0.703 m |

合成测试：

- 111 passed；
- 已知非周期位移恢复到毫米级；
- 噪声 sigma=0.25、部分 nodata 下仍可用；
- 域选择能力边界：多数簇可选出；多数 alias 不可由内部解决。

## 5. 不能声称

- 不能声称绝对地理精度；
- 不能声称 247 的 mode 就是正确模式；
- 不能声称 141/290 内部几何正确；
- 不能声称 V5 原片投影已验收；
- 不能把内部留出改善当成外部可靠性。

## 6. 运行边界

- 只在可见/热像重叠区使用；
- 247 只有 39/101 accepted，覆盖必须随结果报告；
- 边缘/稀疏区优先使用 conservative_valid 与 low_confidence；
- 出现 mode/joint/MAD 明显分歧时，标低置信，不自动选一个；
- 不用于安全关键或需要绝对坐标的场景。

## 7. 后续升级条件

要升级为“可靠几何模型”，至少需要：

1. 每站 8–12 个独立内部锚点，做 P3 Gate 1；
2. 141/290 人工边界顶点或可靠面板分割，做 Gate 0；
3. 热相机内参/外参/畸变链，做 V5 Gate 2。

在这些数据到位前，不再修改默认场模型。
