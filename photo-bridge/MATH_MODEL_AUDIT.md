# DSM–DSM 配准 V2 数学模型审阅：核对与修正记录

审阅基线：ae0519d  
测试：116 passed

## 1. 六个问题的核对结果

| 问题 | 结论 | 处理 |
|---|---|---|
| 掩膜相关使用零位移联合掩膜 | 确认；已实现 opt-in 分离掩膜 | A/B 收益小且混合；默认 joint |
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

- 已实现 opt-in：mask_mode="joint" / "separate"；
- 默认仍为 joint；
- 已做受控 A/B（同一 tile、同一 prior、同一 mode 域）：

| 站 | joint 平均误差 | separate 平均误差 | accepted | 面板 Delta P50/max |
|---|---|---|---|---|
| 141 | 0.1988 | 0.2009 | 89 / 89 | 0.009 / 0.058 m |
| 247 | 0.5068 | 0.4980 | 39 / 40 | 0.007 / 0.034 m |
| 290 | 0.1825 | 0.1813 | 31 / 31 | 0.009 / 0.016 m |

结论：分离掩膜的影响很小且混合；141 略差，247/290 略好，accepted 只多 1 个。**默认保留 joint**，separate 仅作为 A/B 开关。

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
- 已实现 domain_use_usable，opt-in；
- 已做 usable-only vs all 的 A/B：

| 站 | all 平均误差 | usable-only | accepted |
|---|---|---|---|
| 141 | 0.1988 | 0.2156 | 89 / 91 |
| 247 | 0.5068 | 0.5068 | 39 / 39 |
| 290 | 0.1825 | 0.1825 | 31 / 31 |

141 的 usable-only 多收 2 条但误差变差；247/290 完全不变。默认仍用全部域内观测。

- 已实现 competing_separation_m 纯诊断：
  - 141：P95 1.51 m，0 个 <0.6 m；
  - 247：P95 2.61 m，0 个 <0.6 m；
  - 290：P95 1.12 m，0 个 <0.6 m。
  当前没有观测到“<1 m 分离但高竞争”的盲区样本。

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

当前 116 tests passed。

## 9. 仍需 A/B 的项目

1. 逐轴 mode vs joint 的二维簇占比与覆盖；
2. P3 独立锚点上的 E_geo / C_supported；
3. 若 P3 需要，再对 141 做 usable-only 的站点特例。

在这些完成前，不改变默认场模型。

## 10. 当前定位

- 默认为 mode + P0 q-only + coefficient ridge；
- Sigma_i、正交化、软惩罚、KRR 关闭；
- conservative_valid 逐点交集已明确；
- component conservative 改为共同有效核心；
- 仍缺独立几何锚点，不能宣称可靠几何精度。
