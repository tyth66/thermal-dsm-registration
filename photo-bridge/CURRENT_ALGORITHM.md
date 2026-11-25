# 当前正式算法（run 296）

更新时间：2026-11-25

测试：122 passed

2026-11-25 变更（边界）：① 按切向线性混合相邻两个 bin 的修正量（不再取最近 bin），消除 bin 接缝跳变，296 上接缝最大跳变 1.28 m → 0.011 m；② 四角按到边的距离平方反比混合相邻两条边（不再按最近边硬切换），跨角相邻组件位移差 >5 cm 的对数 43 → 0、最大相邻差 14.7 → 10.0 cm，交付计数不变（41417 点 / 8291 组件）。内部区域（距边 ≥2T）两次修正均为 0。

2026-11-24 变更：位移域改为二维 KDE + mean-shift 99.5% 椭圆，删除逐轴 mode 域；歧义筛选改为二维候选 mode 质量；移除恒真的反向一致性检查与仅诊断的 `edge_confidence`；RBF 使用 ECC 2×2 协方差的广义加权残差；移除高通。

2026-11-25 文档变更：新增全原始候选联合研究原型说明（`CANDIDATE_FIELD_EM.md`）；生产路径未切换。

## 1. 算法链路

可见光/热像 DSM
→ 公共米制网格
→ 1% 面积 tile / 50% 重叠
→ masked ECC
→ 二维候选 mode 质量 / alias
→ 二维 KDE + mean-shift 位移域
→ 核心连续场（全局相似 + RBF）
→ 覆盖数归一化
→ 高斯支撑置信度
→ 边界局部残差修正
→ panel rigid 映射
→ conservative 交付

## 2. 采样

| 项目 | 值 |
|---|---:|
| 公共网格 | 0.1 m |
| tile 面积 | 图像 1% |
| tile | 29.6 m |
| step | 14.8 m |
| 重叠 | 50% |

完整 tile 使用完整窗口；边界 tile 由有效 data 轮廓向内生成，只使用 mask 有效区域。

### 位移域

二维 KDE + mean-shift：

- 联合密度 mode，不再逐轴估计；
- 初始协方差用逐轴 MAD，带宽系数 0.8；
- 99.5% Mahalanobis 椭圆，χ²=10.5966；
- 协方差特征值下限 0.25 m²；
- 逐轴 mode / 1.5 m 轴对齐矩形已删除；
- 输出 center_dx_m、center_dy_m、covariance_m2、semiaxis_m、radius、confidence、estimator、rejected。

### 歧义 / alias

- top-5 候选按 `softmax(50ρ)` 加权；
- 在候选位移空间做二维 mean-shift，带宽 2 px；
- 主 mode 质量 `M_1 < 0.73` 且两个 mode 间隔 ≥ `max(1,0.5P)` 时判歧义；
- `alias_conflict` 保留；
- 输出 candidate_mode_count、candidate_top_mass、candidate_second_mass、candidate_mode_separation_m、candidate_ambiguity；
- 每个候选还携带 `covariance_px`（归一化逆 Hessian，均值 1 px²），当前仅用于诊断/联合 EM，不参与生产歧义门。

## 3. 核心场

T(p)=p+f(p)

f(p)=B(p-p_c)+t_0+g(p)

B=A-I

d_c(p)=f(p)

- A：坐标映射的旋转 + 尺度相似变换；
- B=A−I：位移梯度；
- t_0：中心化平移；
- p_c：场站参考中心；
- g：Gaussian RBF 残差；
- 代码内部保存 `B,t_0,p_c`，求值返回 `B(p−p_c)+t_0+g`；序列化仍输出 `A=I+B`、`t_abs=t_0−B·p_c` 以兼容旧消费者；
- 完整 tile 与边界 tile 均进入拟合；
- 观测位置为实际窗口中心 p_w。

### 权重

w_i = cc_i · valid_i^1.5 / n_i · s_i

- n_i：tile footprint 内平均覆盖数；
- s_i：tile 间距支撑因子。

### 参数

| 参数 | 值 |
|---|---:|
| IRLS | 4 次 |
| 鲁棒阈值 | max(3×MAD, 0.5 m) |
| 离群权重 | 0.2 |
| RBF 核长 | 74.0 m |
| RBF ridge | 0.1 |
| 协方差 GLS | 唯一路径 |
| 位移域 | 二维 KDE + mean-shift，99.5% |
| 域协方差下限 | 0.25 m² |
| 候选 mode 质量下限 | 0.73 |
| 候选 mean-shift 带宽 | 2 px |
| 旧歧义 margin（诊断） | 0.02 |

ECC 的 2×2 协方差通过广义加权残差进入 RBF 拟合，是唯一 RBF 权重路径；不再提供标量权重回退。高通已从算法与 CLI 中移除。

## 4. 支撑

连续支撑置信度：

S(p) = Σ w_i exp(-||p-p_i||^2/(2(T/2)^2)) / Σ w_i

- T=29.6 m；
- sigma_support=T/2=14.8 m；
- 输出 support_confidence；
- 不设 C_min。

硬支撑：

field_supported = [d_t <= 2T = 59.2 m]

## 5. 边界修正

边界残差：

r_i=d_i-d_c(p_w,i)

按边和切向 bin 拟合局部残差；每个 bin 6 个系数。

查询时按切向线性混合相邻两个 bin 的修正量：bin 中心处等于该 bin 自身的修正，bin 之间按距离线性过渡，因此修正量沿边连续、接缝处不再跳变。

四角处相邻两条边同时生效：每边权重取到该边距离的平方反比，幅度取最强参与边的 taper（不叠加两侧 taper），因此跨角平分线连续、单边区域的修正量与之前完全一致。

### 当前参数

| 参数 | 值 |
|---|---:|
| bin | 2T = 59.2 m |
| 局部半窗 | 2T = 59.2 m |
| 拟合 taper | 4T = 118.4 m |
| 最小局部 tile | 4 |
| 局部 ridge | 0.01 |
| 局部权重 | cc |
| 边界衰减 | Gaussian sigma=T，范围 2T |

边界衰减归一化到范围端点为零：

alpha(s) = (exp(-s^2/(2T^2)) - exp(-2)) / (1 - exp(-2))

s in [0, 2T]

### 交付

conservative = source_valid AND field_supported AND target_valid

生产覆盖来自热像 DSM 的原生网格。原始五点中至少一点有有效高程时，组件纳入映射结果。修正后覆盖和场支持用于诊断，输出仍采用热像正射像素坐标。

## 6. 验收结果

边界修正前后 signed normal residual：

| 边界 | 修正前 mean | 修正后 mean |
|---|---:|---:|
| 左侧 | -0.0131 m | -0.0112 m |
| 右侧 | -0.0175 m | -0.0047 m |
| 底部 | +0.1533 m | +0.0549 m |
| 顶部 | +0.0480 m | +0.0168 m |

内部 >2T 区域：dd=0，边界修正没有泄漏到内部。

## 7. 正式结果

| 指标 | 值 |
|---|---:|
| panels | 8295 |
| panel_points | 41475 |
| conservative 点位 | 41419 |
| conservative 组件 | 8293 |
| target nodata | 56 |
| edge band | 59.2 m |
| edge tail 点位 | 15430 |

GLS 相对旧标量场的面板中心位移：mean 0.0174 m，P95 0.0496 m，max 0.0680 m；conservative 计数不变。

## 8. 正式文件

当前 GLS + 无高通重跑：

- `outputs/run296/current_run_gls/visible_to_thermal_mapping/visible_to_thermal_mapping_result.json`
- `outputs/run296/visualizations_final_v4/296-final-v4_full_arrays.png`
- `outputs/run296/visualizations_final_v4/296-final-v4_visible_vs_thermal.png`

历史工件（移除前/标量场）：

- `outputs/run296/final/296_mapping_result.json`
- `outputs/run296/final/296_panel_mapping.csv`
- `outputs/run296/final/296_summary.json`
- `outputs/run296/visualizations_final_v2/296-final-v2_full_arrays.png`

历史工件仍含 `edge_confidence`、`consistent`、`fb_error_m`、`reverse_cc` 等旧字段；当前 GLS 工件不再输出这些字段。

## 8.1 研究原型：全原始候选联合

`registration_algorithms/research/` 包另有一条**未进入生产**的全原始候选联合路径（`research/candidate_field.py`）：

- 保留每个 tile 的全部 ECC 候选；
- tile 级 outlier 质量 \(m_i\) 与候选竞争 \(w_{ij}\) 分层；
- 证据比 \(B_{ij}=N(s_{ij};f_i,V_{ij})/u_i(s_{ij})\)；
- 精确 \(W_i,h_i\) M 步，不使用 Moment Matching；
- \(\sigma_f\) 由有界 MAP-MLE 估计；
- 可选 DSM profile 协方差；
- 输出后验曲率、\(\Sigma_f\)、熵分解。

完整推导、实验状态和限制见 [CANDIDATE_FIELD_EM.md](CANDIDATE_FIELD_EM.md)。`registration_algorithms/research/` 不被生产入口导入，生产只通过 `register(..., apply_domain=False)` 和可选 `covariance_fn` 暴露原始观测接口。当前精确 M 步和 DSM 协方差在 296 上的 panel 差异仍大于 Moment 诊断，未完成生产标定。

## 9. 已知限制

1. 边界残差仍是核心场样本内残差，尚未做 grouped cross-fit；
2. 顶部/底部修正后仍有残余：top +1.7 cm，bottom +5.5 cm；
3. 边界局部模型仍是 6 系数、最少 4 观测、ridge 0.01，可辨识性偏弱；
4. 全原始候选联合仍是研究原型：精确 M 步、DSM 协方差和 \(\sigma_f\) MAP-MLE 均未完成生产标定；
5. 边界/外推区域的小后验方差可能主要来自先验，必须与 support 诊断一起解释；
6. `period_m` 现在按需计算：唯一候选模式不跑额外自相关 FFT，记录为 `None`；只有弱模式且候选间距足够时才计算。
