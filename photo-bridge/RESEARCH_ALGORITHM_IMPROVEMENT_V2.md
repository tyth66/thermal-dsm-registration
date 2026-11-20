# 可见光—热红外 DSM 位移场配准：算法改进方案 v2

**状态**：方案定稿，待实施  
**日期**：2026-11-21  
**适用代码**：analysis/photo-bridge/registration_algorithms  
**关系**：本文件定义后续实施顺序；历史分析文件保留，但实施以本文件为准。

---

## 0. 一句话

保留 \(f(p)=Ap+b+g(p)\) 与 rigid 五点映射；先改观测层（掩膜 ECC、反向一致性、逐瓦片协方差 \(\Sigma_i\)），再做场诊断（Jacobian、相似/RBF 低频耦合、支撑几何、三态有效性），最后用封存独立锚点和 V5 做外部验收。

\[
\boxed{
\text{P0 观测层} \rightarrow \text{P1 场与诊断} \rightarrow \text{P2 模型选择} \rightarrow \text{P3 独立锚定/V5}
}
\]

---

## 1. 当前状态

### 1.1 当前代码

- 瓦片覆盖门槛：\(vf \ge 0.5\)。
- 瓦片权重：\(q=vf^{1.5}\)。
- ECC：半分辨率粗解 \(\rightarrow\) 全分辨率精解，单初值。
- 位移域：逐轴 \(c \pm \max(1.5, 3 \times 1.4826 \times \mathrm{MAD})\)。
- 场：全局相似 + Gaussian RBF，auto \(L\)，ridge 0.1。
- 面板映射：rigid（中心 + 四角 5 点均值）。

### 1.2 当前验证

- 93 个单元测试通过。
- 141/247/290，13 种留出方案，39 格平均 +5.9%：
  - 141：+0.7%
  - 247：+7.7%
  - 290：+9.5%
- 面板 \(\Delta\) P50：0.027 / 0.096 / 0.063 m。
- 残余负项：
  - 141 edge_half：−13.1%
  - 247 edge_W：−11.1%
  - 290 edge_W：−14.0%（2 瓦片）

### 1.3 关键判断

这些数字是**留出误差**，不是几何正确性。当前模型仍未解决周期歧义：

\[
d_i=\delta^\star(p_i)+a_i+\epsilon_i
\]

若整片区域共享同一个周期错位 \(a_i\)，高 \(cc\)、空间一致、低训练残差都可能同时成立，但几何位置错误。因此 v2 的第一优先级是观测层，不是 RBF 超参数。

---

## 2. 目标、原则与边界

### 2.1 目标

1. 每条瓦片观测显式携带唯一性、反向一致性和不确定性。
2. 错误但平滑的观测不能进入位移场。
3. 位移场的局部形变可诊断。
4. 最终精度由封存的独立锚点与 V5 验收，不由训练残差或留出误差单独决定。

### 2.2 原则

- 观测层先于场层。
- 保留相似 + RBF 结构，不新增核族。
- P0 未通过，不调 \(L,\lambda\)。
- 独立锚点最后做，且封存，不参与调参。
- 面板位移只表示变更影响，不作为精度。
- \(cc\) 不是匹配正确概率。

### 2.3 范围边界

本方案解决：

\[
\text{可见光 DSM} \rightarrow \text{热红外 DSM/正射像元}
\]

的**相对**平面配准。

不解决：

- 绝对地理坐标正确性：需要 GCP/控制点；
- 热红外原片像素：需要相机位姿、内参、面板三维高程和投影模型；
- 绝对高程/垂直配准。

### 2.4 非目标

- 不新增 RBF 核族（Matérn/Wendland/B-spline 等）。
- 不调 \(L,\lambda\) 去修周期歧义。
- 不用随机瓦片留出替代空间块/独立锚点。
- 不默认改 pointwise 面板映射。
- 不把 bootstrap 方差当成几何正确性的证明。

---

## 3. 观测模型

### 3.1 带周期歧义的观测

\[
d_i=\delta^\star(p_i)+a_i+\epsilon_i
\]

其中：

- \(\delta^\star(p_i)\)：真实可见 \(\rightarrow\) 热红外位移；
- \(a_i\)：周期匹配歧义引起的离散错位；
- \(\epsilon_i\)：常规测量噪声。

规则阵列：

\[
a_i=m_i u_1+n_i u_2
\]

\(u_1,u_2\) 是局部周期方向，\(m_i,n_i\) 为整数。

### 3.2 不可辨识性

若所有观测共享同一周期错位 \(a\)：

\[
d_i^{obs}=\delta^\star(p_i)+a
\]

则

\[
p'=Ap+(b+a)+g(p)
\]

与真实模型在现有观测上残差相同。仅靠 ECC、MAD、相似拟合和 RBF 平滑，数学上无法区分。

**因此必须有观测唯一性检查（P0-2）和独立几何证据（P3）。**

---

## 4. 总体架构

    可见光 DSM + 热红外 DSM
      -> 公共 0.1 m 网格 / 分掩膜高通
      -> P0-1 掩膜 ECC
      -> P0-2 多候选 + 反向一致性
      -> P0-3 逐瓦片协方差 Sigma_i
      -> 位移域（第一道粗差过滤）
      -> P1-1 带 Sigma_i 的加权相似 + RBF
      -> P1-2 Jacobian + 低频耦合 + rigid 判据
      -> P1-3 支撑几何
      -> P1-4 三态有效性
      -> P1-5 块 Bootstrap 不确定性
      -> P2 KRR / L,lambda / 正交化
      -> P3 封存独立锚点 + V5

---

## 5. P0：观测层

P0 通过条件：合成测试通过、反向/歧义可拦截、\(\Sigma_i\) 正定且可标定。  
P0 不使用独立锚点。

### 5.1 P0-1 掩膜 ECC

#### 数学

对每个候选平移 \(s\)：

\[
M_s(x)=m_v(x)\land m_t(x+s)
\]

\[
\mu_v(s)=\frac{\sum_x M_s H_v}{\sum_x M_s},
\qquad
\mu_t(s)=\frac{\sum_x M_s H_t(\cdot+s)}{\sum_x M_s}
\]

\[
ECC(s)=
\frac{
\sum_x M_s (H_v-\mu_v)(H_t(\cdot+s)-\mu_t)
}{
\sqrt{\sum_x M_s(H_v-\mu_v)^2}
\sqrt{\sum_x M_s(H_t(\cdot+s)-\mu_t)^2}
}
\]

约束：

\[
\sum_x M_s \ge 0.5\,|T|
\]

#### 实现

- 高通保持分掩膜：\(H_v=\mathrm{HP}(z_v,m_v)\)，\(H_t=\mathrm{HP}(z_t,m_t)\)。
- 不再用零填充图直接调用 findTransformECC。
- 整数位移粗搜索 + 掩膜 NCC/ECC。
- 子像素用抛物线插值或带掩膜 Gauss-Newton 精化。
- 精化阶段掩膜固定在最接近的整数位移，避免掩膜插值伪影。

#### 验收

- 合成 nodata 边界：掩膜 ECC 与无 nodata 真值偏差 <0.05 px。
- 无 nodata 边界伪匹配。
- 方向测试：热红外结构向东移 5 个网格像素，输出 \(d=(+0.5,0)\) m。

### 5.2 P0-2 多候选 + 反向一致性

#### 前向/反向候选

\[
C_i^+=\{s_{i1}^+,\dots,s_{iK}^+\}
\]

\[
C_i^-=\{s_{i1}^-,\dots,s_{iK}^-\}
\]

反向 ECC：

\[
ECC_{t\to v}(s)
\]

#### 反向一致性

当前 inverse-warp 约定下：

\[
e_i^{fb}=r\,\|s_i^+ + s_i^-\|
\]

\[
\mathrm{inconsistent}_i=
\left[e_i^{fb}>\tau_{fb}\right]
\]

\[
\tau_{fb}=
\max\left(0.2\text{ m},\ 
3\cdot1.4826\cdot\operatorname{MAD}(e^{fb})\right)
\]

阈值由高置信瓦片分布估计，不用固定经验值。

#### 周期歧义

\[
\Delta_i=ECC(s_{i1}^+)-ECC(s_{i2}^+)
\]

\[
\delta_i=r\,\|s_{i1}^+-s_{i2}^+\|
\]

\[
\mathrm{ambiguous}_i=
[\Delta_i<\tau_\Delta]
\land
[\delta_i\ge \rho/2]
\]

\(\rho\) 由可见光高通瓦片的二维自相关峰估计；无明显周期时，\(\rho\) 取保守最小值。

#### 记录字段

    cc_best, cc_second, margin, separation_m, period_m,
    reverse_cc, fb_error_m, consistent, ambiguous

#### 准入

- inconsistent 或 ambiguous：不进入位移场拟合，只保留诊断。
- 歧义率/不一致率作为站点 QA 指标。
- 任何高 \(cc\) 但反向不成立或近邻同分候选的瓦片，必须被拦截。

#### 验收

- 合成周期阵列：正确候选排第一；错位候选标 ambiguous。
- 合成反向不一致：被标 inconsistent。
- 高 \(cc\) 错位瓦片不进入场拟合。

### 5.3 P0-3 逐瓦片协方差 \(\Sigma_i\)

#### 候选离散度

对前向候选做 softmax：

\[
\pi_{ij}=
\frac{\exp(\beta ECC_{ij})}
{\sum_k\exp(\beta ECC_{ik})}
\]

\[
\bar s_i=\sum_j\pi_{ij}s_{ij}
\]

\[
\Sigma_i^{cand}
=r^2\sum_j\pi_{ij}
(s_{ij}-\bar s_i)(s_{ij}-\bar s_i)^T
\]

#### 反向项与基础噪声

\[
\Sigma_i^{fb}=(e_i^{fb})^2 I
\]

\[
\Sigma_i=
\Sigma_i^{cand}
+\Sigma_i^{fb}
+\sigma_0^2 I
\]

\(\sigma_0\) 由重复观测/合成位移标定；\(\beta\) 由候选分数分布标定。

#### 验收

- \(\Sigma_i\) 正定。
- 合成候选景观中，协方差覆盖真实误差。
- 两个候选相距一个周期且分数接近时，\(\Sigma_i\) 在对应方向明显增大。
- 与重复观测/合成标定一致。

---

## 6. P1：场与诊断

P1 通过条件：空间块留出不退化、无折叠、支撑与目标覆盖可解释。  
P1 不使用独立锚点。

### 6.1 P1-1 带 \(\Sigma_i\) 的加权相似与 RBF

#### 相似变换

未知量：

\[
\theta=(a,h,b_x,b_y)^T
\]

\[
X_i=
\begin{bmatrix}
p_{ix}&-p_{iy}&1&0\\
p_{iy}&p_{ix}&0&1
\end{bmatrix}
\]

\[
t_i\approx X_i\theta
\]

加权最小二乘：

\[
\hat\theta=
\left(\sum_i X_i^T W_i X_i\right)^{-1}
\left(\sum_i X_i^T W_i t_i\right)
\]

\[
W_i=\Sigma_i^{-1}
\]

连续鲁棒 IRLS：

\[
\rho_k(e)=
\begin{cases}
\frac12e^2,&e\le k\\
k(e-\frac12k),&e>k
\end{cases}
\]

#### RBF

标量权重：

\[
w_i=\frac{u_i\,q_i\,s_i}{\sigma_i^2},
\qquad
\sigma_i^2=\frac12\operatorname{tr}\Sigma_i
\]

\[
(K^T W K+\lambda I)c=K^T W r
\]

其中：

- \(u_i\)：唯一性权重，由 \(\Delta_i/\tau_\Delta\) 得到；
- \(q_i=vf_i^{1.5}\)：基础覆盖质量；
- \(s_i\)：间距先验，不宣称是密度补偿；
- \(\Sigma_i\)：观测协方差。

#### 验收

- 空间块留出不退化。
- 鲁棒损失优于二值 IRLS。
- alias 仍由 P0-2 拦截；本项不宣称修复周期歧义。

### 6.2 P1-2 Jacobian + 低频耦合 + rigid 判据

#### Jacobian

\[
J_f(p)=A-\frac1{L^2}\sum_i k_i(p)\,c_i\,(p-p_i)^T
\]

诊断：

\[
\det J_f,\qquad
\sigma_{\min}(J_f),\qquad
\sigma_{\max}(J_f)
\]

- \(\det J_f\le0\)：折叠/反向，告警。
- 局部尺度/旋转相对全局 \(A\) 的偏离超过基线 3×MAD：告警。

#### 低频耦合

归一化坐标 \(q\)，低阶基：

\[
\Phi(p)=[1,q_x,q_y]
\]

投影：

\[
P_{\text{low}}g=
\Phi(\Phi^T\Phi)^{-1}\Phi^T g
\]

\[
\eta=
\frac{\|P_{\text{low}}g\|_F}{\|g\|_F}
\]

\[
\eta_{\text{sim}}=
\frac{\|P_{\text{low}}g\|_F}
{\|(A-I)p+b\|_F}
\]

\(\eta\) 大：RBF 在重新拟合低频。  
\(\eta_{\text{sim}}\) 大：RBF 在替代全局相似项。

先诊断，不立即加惩罚。若确认严重，P2 再做正交约束：

\[
\min_c\|W^{1/2}(Kc-r)\|^2+\lambda\|c\|^2
\quad\text{s.t.}\quad
\Phi^T W Kc=0
\]

#### rigid/similarity 判据

\[
e_k\approx J_\delta(P_0)(P_k-P_0)
\]

\[
\|J_\delta(P_0)\|_2\,\rho\le\tau_{\text{panel}}
\]

满足才用 rigid；超过时标记面板尺度形变显著，再决定是否 similarity，不自动改 pointwise。

#### 验收

- 无 \(\det J_f\le0\)。
- \(\eta,\eta_{\text{sim}}\) 分布可解释。
- rigid 判据可复现。
- 合成“低频被 RBF 吸收”案例能被 \(\eta\) 检出。

### 6.3 P1-3 支撑几何

\[
a_i(p)=W_i k(p,p_i)
\]

\[
N_{\text{eff}}(p)=
\frac{\left(\sum_i a_i(p)\right)^2}
{\sum_i a_i(p)^2}
\]

\[
\bar p(p)=
\frac{\sum_i a_i p_i}{\sum_i a_i}
\]

\[
C_p(p)=
\frac{\sum_i a_i(p_i-\bar p)(p_i-\bar p)^T}
{\sum_i a_i}
\]

有效支撑：

\[
d_{\min}\le D
\ \land\
N_{\text{eff}}\ge3
\ \land\
\lambda_{\min}(C_p)\ge\epsilon
\]

### 6.4 P1-4 面板三态有效性

1. source_valid：原始 5 点是否有热红外数据。
2. field_supported：中心 + 至少 3 角点是否满足 P1-3。
3. target_valid：纠正后 5 点或多边形覆盖率是否足够。

起始策略：

- corrected 5 点中 ≥3 点有效，或 corrected polygon area ≥50%。
- 不满足时输出低置信标记，不静默当作有效。
- 原始任意一点有效不再等同于纠正后可用。

### 6.5 P1-5 不确定性

空间块 Bootstrap：

\[
\Sigma_\delta(p)=\operatorname{Cov}_b(\hat\delta_b(p))
\]

\[
\Sigma_{\text{pixel}}=
B_t^{-1}\Sigma_\delta B_t^{-T}
\]

报告 1σ 椭圆/区间。

注意：周期错位时方差可能很小但结果仍错；不能替代 P3。

---

## 7. P2：模型选择

### 7.1 KRR vs ridge

当前：

\[
(K^T W K+\lambda I)c=K^T W r
\]

标准加权 KRR：

\[
(K+\lambda W^{-1})c=r
\]

或：

\[
(WK+\lambda I)c=Wr
\]

两者 \(\lambda\) 分开调；用空间块留出比较，不用训练残差。

### 7.2 \(L,\lambda\) 空间隔离选择

嵌套空间 CV：

- 外层留整块/整边；
- 内层在训练块内选 \(L,\lambda\)；
- 评价用留出块，不用随机瓦片。

### 7.3 低频正交化

仅当 P1-2 的 \(\eta\) 确认严重时启用：

\[
\Phi^T W Kc=0
\]

或对 \(\|P_{\text{low}}g\|^2\) 加惩罚。

### 7.4 数值与复杂度

- 坐标局部中心化：

\[
f(p)=p_0+A(p-p_0)+t_0+g(p)
\]

- 全流程 float64。
- \(N\sim10^2\) 时 \(O(N^3)\) 可接受；暂不做稀疏核。

---

## 8. P3：封存独立锚定 + V5（最终验收）

### 8.1 原则

- 锚点集在 P0–P2 期间封存，不打开、不调参。
- P3 是发布前最后一步；P3 不通过，不发布。
- 不能在看到 P3 结果后回头调 \(L,\lambda,\tau\)，否则不再是独立验证。

### 8.2 锚点

来源：

- 阵列外边界；
- 建筑角点、道路交点；
- 不规则地物；
- 人工/半自动在可见光与热红外正射上量测。

误差：

\[
E_{\text{geo}}=
\sqrt{\frac1M\sum_{j=1}^M\|f(a_j)-b_j\|^2}
\]

报告 P50/P95，分站点。

### 8.3 V5 热像投影

141/247/290 各跑一次；与基线比较，不退化。

### 8.4 最终判定

- \(E_{\text{geo}}\) P95 不差于基线。
- 有 alias 的站点相对基线改善 ≥20%。
- V5 不退化。
- 任一站点显著变差，整体不通过，回到 P0/P1 找原因。

---

## 9. 实施顺序与依赖

| 顺序 | 模块 | 依赖 | 产物 |
|---|---|---|---|
| 1 | P0-1 掩膜 ECC | 无 | 掩膜 ECC 函数 + nodata/方向测试 |
| 2 | P0-2 多候选 + 反向一致性 | P0-1 | 候选字段、fb_error、ambiguous |
| 3 | P0-3 \(\Sigma_i\) | P0-2 | sigma_e/sigma_n/sigma_en |
| 4 | P1-1 加权相似/RBF | P0-3 | 带 \(\Sigma_i\) 的场 |
| 5 | P1-2 Jacobian + 低频耦合 | P1-1 | 诊断字段/报告 |
| 6 | P1-3/4/5 | P1-1 | 支撑/三态/不确定性 |
| 7 | P2 KRR/\(L,\lambda\) | P1 全部 | 对照报告 |
| 8 | P3 锚定/V5 | P2 | 最终验收 |

P0 未通过，不做 P1。  
P1 未通过，不做 P2。  
P2 未通过，不做 P3。

---

## 10. 验收矩阵

| 阶段 | 验收 | 通过标准 |
|---|---|---|
| P0-1 | 合成 nodata；方向 | 偏差 <0.05 px；5 px 方向正确 |
| P0-2 | 合成周期；反向 | 正确候选第一；错位/反向不一致被拦截 |
| P0-3 | 协方差标定 | 正定；覆盖真实误差；与重复观测一致 |
| P1-1 | 空间块留出 | 不退化；鲁棒优于二值 IRLS |
| P1-2 | Jacobian/低频 | 无 det≤0；eta 可解释；rigid 判据可复现 |
| P1-3 | 支撑 | N_eff/Cp 与单侧/边缘一致 |
| P1-4 | 三态有效性 | 纠正后全 nodata 不再标 valid |
| P1-5 | 不确定性 | 1σ 覆盖合理；不被低方差误导 |
| P2 | KRR/L,lambda | 空间块稳定；无站点退化 |
| P3 | 独立锚点/V5 | E_geo P95 不差；V5 不退化 |

---

## 11. 数据契约

### 11.1 瓦片记录

    row, column,
    east, north,
    dx_m, dy_m,
    cc, ncc_before, ncc_after,
    valid_fraction, quality_weight,
    cc_best, cc_second, margin, separation_m, period_m,
    reverse_cc, fb_error_m, consistent, ambiguous,
    sigma_e, sigma_n, sigma_en,
    accepted, outlier

### 11.2 场拟合输入

- east, north, dx_m, dy_m；
- cc；
- quality_weight；
- sigma_e/sigma_n/sigma_en（或 \(\Sigma_i\)）；
- uniqueness 权重；
- ambiguous/inconsistent 标记。

### 11.3 输出与溯源

- 场参数：\(A,b,c,L,\lambda\)；
- 每个瓦片的歧义/反向/协方差诊断；
- 面板三态有效性；
- 逐面板不确定性；
- P3 锚点误差与 V5 结果。

---

## 12. 风险与约束

1. \(\tau_{fb},\tau_\Delta,\beta,\sigma_0\) 需要基线标定；不能拍脑袋。
2. 反向一致性不能保证唯一性；周期歧义仍需 P0-2 的周期候选。
3. \(\Sigma_i\) 是经验协方差，不是校准概率；需重复/合成标定。
4. 低频耦合阈值需基线分布；先诊断，后惩罚。
5. P3 可能整体否决 P0–P2；这是独立验证的代价，不能用调锚点规避。
6. 3 个站点样本有限；290 边缘只有 2–6 个瓦片，结论需标注样本量。

---

## 13. 决策记录

- **不采用 blanket 双初值**：141 单初值更好（面板 0.027 vs 0.050；edge_half −13.1% vs −24.6%；pca4 +2.5% vs −4.2%）；247 双初值面板更稳但均值相近。17.67 m alias 由位移域处理。
- **保留覆盖权重 \(q=vf^{1.5}\)** 作为当前基础质量权重；P0 后由 \(\Sigma_i\) 与唯一性权重共同决定。
- **不采用全量覆盖权重之外的局部核/多尺度/GCV/B-spline**：留出与稳定性不支持。
- **独立锚定移到 P3**：按用户要求最后做；P0–P2 只用合成/内部指标。
- **不新增 RBF 核族**；不先调 \(L,\lambda\)。

---

## 14. 下一步

从 **P0-1 掩膜 ECC** 开始：

1. 实现候选位移下的联合掩膜 ECC。
2. 加 nodata 边界合成测试。
3. 加 5 px 方向测试。
4. 通过后进入 P0-2。

P0-1 未通过，不修改场模型。
