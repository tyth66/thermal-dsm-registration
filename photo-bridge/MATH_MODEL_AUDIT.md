# DSM–DSM 配准 V2 数学模型审阅：核对与修正记录

审阅基线：6492fba  
测试：114 passed

## 1. 六个问题的核对结果

| 问题 | 结论 | 处理 |
|---|---|---|
| 掩膜相关使用零位移联合掩膜 | 确认：代码使用 J(x)J(x+s)，比 Jv(x)Jt(x+s) 更严格 | 默认不改；列为 A/B |
| 逐轴 mode 可能误判二维主模式 | 确认：mode 默认，joint 仅 opt-in | 不改默认；P3 诊断 |
| component_conservative 与逐点交集不等价 | 确认：旧定义只组合三个组件级条件 | **已修正** |
| 支撑判据 kappa>0 几乎无效 | 确认：指数核恒正，实际主要靠最近邻距离 | 不改默认；保留 N_eff/C_p 诊断 |
| Sigma_i 像素协方差到 EN 缺符号翻转 | 确认：旧代码只乘 r^2，未翻转交叉项 | **已修正** |
| 分步相似 + RBF 非联合最优 | 确认：这是有意的工程先验 | 不改结构 |

## 2. 掩膜 ECC

当前实现：

- register 只传入一个 joint 掩膜；
- masked_ecc_surface 内部 M_s(x)=J(x)J(x+s)；
- 真正需要的匹配掩膜是 M_s*(x)=Jv(x)Jt(x+s)；
- 因此当前 M_s 是 M_s* 的子集。

结论：不会因此产生“使用了无效像元”的错误，但会丢掉本可参与匹配的有效像元。对 141/290 这种 DSM 覆盖边界不一致的站点，损失需要实测。

处理：

- 不直接改默认；
- 下一步做受控 A/B：同一候选、同一 prior，比较 joint 掩膜 vs 分离 Jv/Jt 掩膜；
- 记录每瓦片 |M_s|/|M_s*|、候选峰、accepted、合成缺失区测试；
- 若分离掩膜只增加样本但不改变候选，收益有限；若产生新候选，必须看 P3。

## 3. Mode 与二维主模式

当前：

- 默认 mode 是逐轴最密簇；
- joint 已实现但默认关闭；
- 域估计在一致性/歧义过滤之前，对全部域内记录进行；
- 247 mode accepted 39/101，joint 35/101。

确认：

- 最密簇不等于真实簇；
- 逐轴组合不保证二维占据；
- 域估计包含随后被标记 ambiguous 的观测。

处理：

- 默认不改；
- 增加两个诊断变量：
  - 二维主簇占比；
  - 域估计只用 usable 观测 vs 全部域内观测的 A/B。
- P3 用同一锚点比较 MAD/mode/joint，不只比内部留出。

## 4. component_conservative：已修正

旧定义：

- C_source = 任一点 source_valid；
- C_field = center field + >=3 corners field；
- C_target = >=3 点 target_valid；
- component_conservative = 三者相与。

问题：可能组件级为真，但没有任何一点同时满足 source & field & target。

新定义：

component_conservative_valid =
center.conservative_valid
and
sum(corner.conservative_valid) >= 2

其中逐点：

conservative_valid = source_valid AND field_supported AND target_valid

这要求“存在共同有效核心”，不再是三个独立组件门限的组合。

计数：

| 站 | panels | point conservative | component conservative |
|---|---|---|---|
| 141 | 5285 | 4971 | 4971 |
| 247 | 490 | 490 | 490 |
| 290 | 1576 | 1576 | 1576 |

当前三站两种计数相同，但接口语义已不同；测试覆盖了旧定义的反例。

## 5. 场支撑

kappa(p)=1/N sum_i k_i(p)>0 在精确算术下恒成立，所以支撑判据主要等价于：

nearest(p) <= D_s

确认。处理：

- 不把 kappa>0 当作有效筛选；
- 保留 N_eff、C_p、lambda_min/lambda_max、nearest 作为诊断；
- 不设未经 P3 标定的硬阈值。

## 6. Sigma_i 的 EN 协方差

像素到 EN：

d^e = r s_x
d^n = -r s_y

所以：

Sigma_EN = B Sigma_px B^T
B = diag(r, -r)

即：

Sigma_EN = r^2 [[sigma_x^2, -sigma_xy], [-sigma_xy, sigma_y^2]]

旧代码只乘 r^2；已修正为 pixel_covariance_to_world，并加测试：

- diag 不变；
- 交叉项符号翻转。

当前 Sigma_i 加权默认关闭，因此这次修正不影响默认场，只影响可选异方差实验和不确定性解释。

## 7. 分步拟合

当前：

1. 加权 IRLS 相似变换；
2. 以相似残差为目标拟合 RBF。

不是联合最优：

min_{theta,c} ||W^(1/2)(t-X theta-Kc)||^2 + lambda||c||^2

确认。处理：不改结构。P2 的硬正交、软惩罚、KRR 都未通过；分步“先相似、后残差”的工程先验暂时保留。

## 8. 新增数学一致性测试

- 协方差符号：pixel_covariance_to_world；
- 组件保守语义：旧反例 + 新共同核心；
- 解析 Jacobian vs 中心差分；
- 合成端到端、噪声/nodata、mode 能力边界。

当前 114 tests passed。

## 9. 仍需 A/B 的项目

1. 分离 Jv/Jt 掩膜 vs 当前 joint 掩膜；
2. 域估计只用 usable 观测 vs 全部域内观测；
3. 逐轴 mode vs joint 的二维簇占比与覆盖；
4. 小分离竞争峰诊断 d_competing；
5. P3 独立锚点上的 E_geo / C_supported。

在这些完成前，不改变默认场模型。

## 10. 当前定位

- 默认为 mode + P0 q-only + coefficient ridge；
- Sigma_i、正交化、软惩罚、KRR 关闭；
- conservative_valid 逐点交集已明确；
- component conservative 改为共同有效核心；
- 仍缺独立几何锚点，不能宣称可靠几何精度。
