# 可见光—热像 DSM 平面配准数学模型

实现与运行参数见 [CURRENT_ALGORITHM.md](CURRENT_ALGORITHM.md)。本文档描述生产模型；全原始候选联合实验模型见 [CANDIDATE_FIELD_EM.md](CANDIDATE_FIELD_EM.md)。

## 1. 目标与范围

输入：

- 可见光 DSM；
- 热像 DSM；
- 可见光正射影像；
- 热像正射影像；
- 可见光组件世界模型：每个面板的中心、四角、阵列编号。

输出：

- 可见光 EN 坐标到热像 EN 坐标的连续位移场；
- 每个面板中心与四角的 EN/像素坐标；
- 源、场、目标三态有效性；
- 保守交付标记。

本模型只处理平面位置映射，不包含：

- 绝对地理定位；
- 热相机内外参、畸变和 V5 原始热像投影；
- 高程场或三维表面重建。

## 2. 符号

- \(p=(e,n)\)：可见光 EN 坐标；
- \(d(p)\)：可见光到热像的 EN 位移；
- \(q=p+d(p)\)：热像 EN 坐标；
- \(r\)：公共网格分辨率；
- \(W_t\)：瓦片物理宽度；
- \(W_s\)：瓦片步长；
- \(D_s\)：场支撑半径；
- \(J_v(x),J_t(x)\)：可见/热像有效掩膜；
- \(J(x)=J_v(x)\land J_t(x)\)：联合掩膜；
- \(z_v(x),z_t(x)\)：可见/热像 DSM 高程；
- \(\Sigma_i\)：第 \(i\) 个瓦片的 2×2 观测协方差；
- \(P_i=\Sigma_i^{-1}\)：其精度矩阵。

## 3. 公共网格与预处理

### 3.1 公共网格

两幅 DSM 必须具有一致投影米制 CRS。  
取两幅 DSM 包围盒交集，按分辨率 \(r=0.1\) m 重投影到同一网格。

### 3.2 有效掩膜

\[
J(x)=J_v(x)\land J_t(x)
\]

无效像元不参与相关，也不以零值参与归一化。

## 4. 分块与场支撑

### 4.1 自适应 Tile

当前按图像面积确定 tile：

\[
A_t=0.01\,WH
\]

\[
W_t\approx\sqrt{A_t}
\]

\[
W_s=W_t/2
\]

像素数取偶数，保证精确 50% 重叠。

若第一次尝试 accepted 少于 6，则 tile/step 减半重试，最多 3 次；取 accepted 最多的尝试。

### 4.2 场支撑

当前使用连续高斯支撑置信度：

\[
S(p)=\frac{\sum_i w_i e^{-\|p-p_i\|^2/(2(T/2)^2)}}{\sum_i w_i}
\]

并保留硬保护：

\[
\text{field\_supported}(p)=
[\text{finite }d(p)]
\land[\text{weight}(p)>0]
\land[d_{\min}(p)\le 2T]
\]

不设支撑置信度门限 \(C_{\min}\)。

其中 \(d_{\min}\) 是到最近观测中心的距离。\(D_s\) 是拟合阶段的支撑计划参数；交付硬保护使用 \(2T\)。

## 5. 单瓦片观测

### 5.1 瓦片筛选

对瓦片 \(T_k\)：

\[
vf_k=\frac{1}{|T_k|}\sum_{x\in T_k}J(x)
\]

要求：

\[
vf_k\ge0.5
\]

纹理标准差不足时，该瓦片不产生观测。

### 5.2 粗初值

瓦片降采样一半，归一化后做 ECC，得到粗位移：

\[
s_0=\mathrm{ECC}_{1/2}(z_v,z_t)
\]

还原：

\[
\hat s_0=2s_0
\]

搜索半径：

\[
R_p=\max(8r,2\ \text{m})
\]

即 prior 半径默认 2 m。

### 5.3 掩膜相关

对候选位移 \(s\)：

\[
M_s(x)=J(x)J(x+s)
\]

要求：

\[
|M_s|\ge0.5|M_0|
\]

定义：

\[
C_m(s)=\sum_x M_s(x)
\]

\[
C_v(s)=\sum_x J(x)J(x+s)z_v(x)
\]

\[
C_t(s)=\sum_x J(x)J(x+s)z_t(x+s)
\]

\[
C_{vv}(s)=\sum_x J(x)J(x+s)z_v(x)^2
\]

\[
C_{tt}(s)=\sum_x J(x)J(x+s)z_t(x+s)^2
\]

\[
C_{vt}(s)=\sum_x J(x)J(x+s)z_v(x)z_t(x+s)
\]

归一化相关：

\[
\rho(s)=
\frac{
C_{vt}-\frac{C_vC_t}{C_m}
}{
\sqrt{
\left(C_{vv}-\frac{C_v^2}{C_m}\right)
\left(C_{tt}-\frac{C_t^2}{C_m}\right)
}
}
\]

整数峰再做抛物线亚像素插值。

### 5.4 候选与 alias 冲突

前向只保留 prior 窗口内 top-5：

\[
\|s-\hat s_0\|_2\le R_p
\]

全局 top-1 用于 alias 冲突：

\[
d_{\rm global}>R_p
\quad\land\quad
\rho_{\rm global}>\rho_{\rm best}+0.05
\]

### 5.5 二维候选 mode 质量

前向 top-5 候选为：

\[
C=\{(s_j,\rho_j)\}_{j=1}^{K},\quad K\le5
\]

候选权重：

\[
\pi_j=\mathrm{softmax}\left(50(\rho_j-\max_k\rho_k)\right)
\]

每个候选还携带相关面峰值的归一化逆 Hessian：

\[
C^{\rm px}_{ij}=\frac{(-H_{ij})^{-1}}
{\tfrac12\mathrm{tr}((-H_{ij})^{-1})}
\]

其特征值保持各向异性，均值方差归一为 1 px²。该协方差当前用于诊断/联合 EM 原型，不进入生产歧义门。

可选的 DSM 高程 profile 协方差为：

\[
Q_{ij}=-\nabla_s^2\rho_i(s_{ij}),\qquad
\beta_{ij}^{\rm eff}=\frac{\operatorname{SST}_{v,i}}{\sigma_z^2}\rho_{ij},\qquad
C_{ij}=(\beta_{ij}^{\rm eff}Q_{ij})^{-1}
\]

该路线属于候选级模型：\(\pi_{ij}\) 仍是条件 softmax，\(C_{ij}\) 只进入测量协方差，不乘 \(|C_{ij}|^{1/2}\)。当前生产仍使用归一化逆 Hessian；DSM 协方差为可选/实验路径，未默认启用。

实验性歧义信息：单 tile 的负对数边缘似然曲率为

\[
\mathcal I_i=W_i-D_i,\qquad
W_i=\sum_jq_{ij}V_{ij}^{-1},\qquad
D_i=\operatorname{Cov}_{q_i}(V_{ij}^{-1}(s_{ij}-f_i))
\]

它只用于诊断/不确定性，不替代 M 步的 \(W_i\)。对应的局部场参数后验Hessian为

\[
H_{\rm obs}=\Lambda+\sum_i\Phi_i^T\mathcal I_i\Phi_i,\qquad
\Sigma_f(p)=\Phi(p)H_{\rm obs}^{-1}\Phi(p)^T
\]

\[
\Sigma_d(p)=\Sigma_f(p)+\sigma_f^2I
\]

这是单个后验模式附近的局部 Laplace 不确定性，不覆盖 coherent alias 的全局错配模式。

潜在变量熵拆分为：

\[
H_i^{\rm valid}=-m_i\log m_i-(1-m_i)\log(1-m_i)
\]

\[
H_i^{\rm alias}=-\sum_jw_{ij}\log w_{ij}
\]

\[
H_i^{\rm total}=H_i^{\rm valid}+m_iH_i^{\rm alias}
\]

这样可以区分“确定无效、有效性不明确、确定有效但候选歧义、唯一对应”四种状态。

在候选位移空间做二维 Gaussian mean-shift：

\[
w_j(c)=\exp\left(-\frac{\|s_j-c\|^2}{2h_c^2}\right)
\]

\[
c^{(t+1)}=\frac{\sum_j \pi_j w_j(c^{(t)})s_j}
{\sum_j \pi_j w_j(c^{(t)})}
\]

其中 \(h_c=2\) px。收敛后得到 mode \(m_1,m_2,\dots\)，其质量为：

\[
M_k=\sum_{j\in\mathrm{mode}_k}\pi_j
\]

歧义判定：

\[
\text{ambiguous}=
\left[
M_1<0.73
\land
d(m_1,m_2)\ge\max(1,0.5P)
\right]
\lor
\text{alias\_conflict}
\]

其中 \(d(m_1,m_2)\) 是两个 mode 的米制距离，\(P\) 是可见光 tile 自相关主周期。

0.73 来自旧 margin 阈值的等价点：两个候选时

\[
M_1=\frac{1}{1+e^{-50\Delta\rho}}
\]

当 \(\Delta\rho=0.02\) 时，\(M_1=0.731\)。

若所有候选合并成一个 mode，则 \(M_1=1\)，不判歧义；但“只有一个 mode”仍不等于几何唯一性证明。`margin`、`cc_second`、`separation_m` 继续保留为诊断量。

### 5.6 观测协方差

前向候选 softmax：

\[
\pi_k=\mathrm{softmax}(50(\rho_k-\max\rho))
\]

\[
\mu=\sum_k\pi_k s_k
\]

\[
\Sigma_{\rm px}=
\sum_k\pi_k(s_k-\mu)(s_k-\mu)^T
\]

像素到 EN：

\[
B=\mathrm{diag}(r,-r)
\]

\[
\Sigma_{\rm EN}=B\Sigma_{\rm px}B^T
\]

\[
\Sigma_i=
\Sigma_{\rm EN}
+\sigma_0^2I
\]

\[
\sigma_0=0.05\ \text{m}
\]

该协方差用于 8.4.1 的 RBF 广义加权。

### 5.7 观测记录

每个瓦片产生：

- 中心 \(p_i=(e_i,n_i)\)；
- 位移 \(d_i=(dx_i,dy_i)\)；
- 目标 \(t_i=p_i+d_i\)；
- 相关系数 \(cc_i\)；
- 质量权重 \(qf_i=vf_i^{1.5}\)；
- ambiguous；
- margin/separation/period；
- candidate_count；
- 协方差字段；
- alias_conflict。

## 6. 位移域

### 6.1 二维联合密度

对每个 tile 的位移观测：

\[
d_i=(dx_i,dy_i)
\]

用 Gaussian KDE 估计二维联合密度：

\[
\hat p_H(d)=\frac{1}{N}\sum_i
K_H(d-d_i)
\]

\[
K_H(u)=(2\pi)^{-1}|H|^{-1/2}
\exp\left(-\frac12u^T H^{-1}u\right)
\]

带宽矩阵：

\[
H=h^2S_0
\]

\[
h=0.8N^{-1/6}
\]

其中 \(S_0\) 是逐轴 MAD 的鲁棒初始协方差：

\[
S_0=\mathrm{diag}\left(
(1.4826\,\mathrm{MAD}(dx))^2,
(1.4826\,\mathrm{MAD}(dy))^2
\right)
\]

MAD 尺度下限为 0.15 m。

二维 mode：

\[
\hat c=\arg\max_d \hat p_H(d)
\]

用 mean-shift 迭代求解：

\[
w_i(c)=\exp\left(-\frac12(d_i-c)^T H^{-1}(d_i-c)\right)
\]

\[
c^{(t+1)}=\frac{\sum_i w_i(c^{(t)})d_i}
{\sum_i w_i(c^{(t)})}
\]

为避开局部 mode，使用确定性多起点：中位数点，加上 10%–90% 分位范围内 \(7\times7\) 网格；取 KDE 密度最高的收敛点。

### 6.2 主簇协方差与椭圆

给定 mode \(\hat c\)，用 Mahalanobis 椭圆迭代主簇：

\[
C=\left\{i:(d_i-\hat c)^T S^{-1}(d_i-\hat c)
\le\chi^2_{2,0.995}\right\}
\]

\[
\chi^2_{2,0.995}=10.5966
\]

\[
S=\mathrm{Cov}(C)
\]

对 \(S\) 的特征值施加下限：

\[
\lambda_{\min}(S)\ge0.25\ \mathrm{m}^2
\]

接受域为：

\[
D=\left\{d:(d-\hat c)^T S^{-1}(d-\hat c)
\le\chi^2_{2,0.995}\right\}
\]

即 99.5% 置信椭圆。

若观测少于 8 个，或主簇少于 6 个，回退到二维 median/MAD：

\[
\hat c=\mathrm{median}(d_i)
\]

\[
S=\mathrm{diag}\left(
(1.4826\,\mathrm{MAD}(dx))^2,
(1.4826\,\mathrm{MAD}(dy))^2
\right)
\]

并同样施加协方差下限。

### 6.3 域筛选与统计

保留：

\[
(d_i-\hat c)^T S^{-1}(d_i-\hat c)
\le\chi^2_{2,0.995}
\]

非有限位移不参与域估计，并作为无效测量被拒绝。

输出：

- center_dx_m、center_dy_m；
- covariance_m2；
- semiaxis_m；
- radius；
- confidence；
- estimator；
- rejected。

## 7. 观测集合

场拟合前，只保留：

\[
\text{accepted}=
\text{ambiguous}=0
\]

域筛选在 accepted 之前执行。

## 8. 场模型

### 8.1 模型形式

先区分坐标映射与位移场。

坐标映射：

\[
T(p)=p+f(p)
\]

位移场：

\[
f(p)=B(p-p_c)+t_0+g(p)
\]

其中 \(p_c\) 是场站参考中心，\(x=p-p_c\)。

\[
B=A-I=
\begin{pmatrix}
u & -v\\
v & u
\end{pmatrix}
\]

\(A\) 是真正的相似变换矩阵，\(B=A-I\) 是位移梯度矩阵。\(g\) 是 Gaussian RBF 残差。

零位移状态：

\[
B=0,\qquad t_0=0,\qquad g=0
\]

此时 \(T(p)=p\)，\(f(p)=0\)。

当前代码已按中心化参数保存 \(B,t_0,p_c\)，求值返回 \(B(p-p_c)+t_0+g\)。模型序列化仍保留绝对映射键以兼容旧消费者：

\[
t_{\rm abs}=t_0-Bp_c,\qquad A=I+B
\]

序列化中的 `matrix` 是 \(A\)，`translation` 是 \(t_{\rm abs}\)；内部拟合/求值使用 \(B,t_0,p_c\)。

### 8.2 观测权重

\[
w_i=\frac{cc_i\cdot qf_i}{n_i}\cdot s_i
\]

其中：

\[
qf_i=vf_i^{1.5}
\]

\(n_i\) 是 tile footprint 内平均覆盖数（tile/8 格网离散积分）。

\(s_i\) 是最近邻支撑权重：

\[
\delta_i=\min_{j\ne i}\|p_i-p_j\|_2
\]

\[
\delta_0=\mathrm{median}(\delta_i)
\]

\[
s_i=
\exp\left[
-\frac12
\left(\frac{\delta_i}{\delta_0}-1\right)^2
\right]
\]

少于 3 条观测时不做支撑权重。

### 8.3 相似变换 IRLS

在中心化位移参数下：

\[
\min_{B,t_0}\sum_i w_i
\left\|d_i-\left[B(p_i-p_c)+t_0\right]\right\|^2
\]

最终：

\[
\epsilon_i=
\left\|d_i-\left[B(p_i-p_c)+t_0\right]\right\|
\]

当前代码直接拟合 \(B,t_0\)；序列化时再换算为 \(A,t_{\rm abs}\) 以保持兼容。

\[
\tau=\max(3\cdot1.4826\mathrm{MAD}(\epsilon),0.5)
\]

内点：

\[
\epsilon_i\le\tau
\]

拟合权重：

\[
w_i^{fit}=
\begin{cases}
w_i,&i\in\text{inlier}\\
0.2w_i,&i\notin\text{inlier}
\end{cases}
\]

最多 4 轮。

### 8.4 RBF 残差

\[
r_i=d_i-\left[B(p_i-p_c)+t_0\right]
\]

\[
K_{ij}=
\exp\left(
-\frac{\|p_i-p_j\|^2}{2L^2}
\right)
\]

\[
L=\max(2.5W_t,4\delta_0)
\]

RBF 系数由 8.4.1 的协方差广义加权块系统求解；默认：

\[
\lambda=0.1
\]

### 8.4.1 协方差广义加权

每个瓦片还携带 ECC 候选离散度得到的 2×2 协方差：

\[
\Sigma_i=
\begin{pmatrix}
\sigma_{e,i}^2 & \sigma_{en,i}\\
\sigma_{en,i} & \sigma_{n,i}^2
\end{pmatrix}
\]

记其逆为：

\[
P_i=\Sigma_i^{-1}=
\begin{pmatrix}
a_i & b_i\\
b_i & d_i
\end{pmatrix}
\]

为保持 \(\lambda\) 与协方差尺度无关，令归一化精度

\[
\tilde P_i=\frac{P_i}{\mathrm{median}(\tfrac12(a_i+d_i))}
\]

下文将归一化后的 \(\tilde P_i\) 简记为 \(P_i\)，其元素 \(a_i,b_i,d_i\) 均为归一化后的精度。

令 \(u_i=w_i^{fit}\) 为 8.3 的 IRLS 拟合权重。二维 RBF 残差的广义最小二乘为：

\[
\min_{C}\sum_i
u_i
(r_i-\phi_i C)^T P_i (r_i-\phi_i C)
+\lambda\|C\|_F^2
\]

其中 \(C\) 是 \(N\times2\) 的 RBF 系数矩阵，\(\phi_i\) 是第 \(i\) 行的核向量。对 \(c_e,c_n\) 求导得到 2N×2N 块系统：

\[
\begin{pmatrix}
K^T\operatorname{diag}(u_i a_i)K+\lambda I &
K^T\operatorname{diag}(u_i b_i)K\\
K^T\operatorname{diag}(u_i b_i)K &
K^T\operatorname{diag}(u_i d_i)K+\lambda I
\end{pmatrix}
\begin{pmatrix}c_e\\c_n\end{pmatrix}
=
\begin{pmatrix}
K^T(u_i a_i r_e+u_i b_i r_n)\\
K^T(u_i b_i r_e+u_i d_i r_n)
\end{pmatrix}
\]

当 \(\Sigma_i=\sigma_0^2I\) 时，\(P_i=(1/\sigma_0^2)I\)，GLS 退化为各向同性标量权重；这是模型的一个特例，不是单独的估计路径。

### 8.5 场求值

\[
g(p)=\sum_i c_i
\exp\left(
-\frac{\|p-p_i\|^2}{2L^2}
\right)
\]

\[
f(p)=B(p-p_c)+t_0+g(p)
\]

核密度：

\[
w(p)=\frac1N\sum_i
\exp\left(
-\frac{\|p-p_i\|^2}{2L^2}
\right)
\]

## 9. 面板映射

每个面板取 5 个点：

- center；
- 4 corners。

### 9.1 Rigid 映射

对面板 \(P\)：

\[
\bar\delta_P=
\frac15\sum_{j=1}^5
f(p_j)
\]

\[
q_j=p_j+\bar\delta_P
\]

面板只做整体平移，保持形状。

### 9.2 像素与 alpha

热像仿射逆变换：

\[
(u_j,v_j)=T_t^{-1}(q_j)
\]

在热像 alpha 上按最近像元采样。

### 9.3 三态

\[
S_j=[\alpha_t(u_j^0,v_j^0)>0]
\]

\[
F_j=\text{field\_supported}(p_j)
\]

\[
T_j=[\alpha_t(u_j,v_j)>0]
\]

### 9.4 组件有效性

源覆盖：

\[
C_{\rm source}=\exists j:S_j
\]

场覆盖：

\[
C_{\rm field}=F_{\rm center}
\land
\left(\sum_{\rm corners}F_j\ge3\right)
\]

目标覆盖：

\[
C_{\rm target}=\sum_jT_j\ge3
\]

逐点保守：

\[
V_j=S_j\land F_j\land T_j
\]

组件保守：

\[
C_{\rm conservative}=
V_{\rm center}
\land
\left(\sum_{\rm corners}V_j\ge2\right)
\]

兼容字段：

\[
\text{valid}_j=C_{\rm source}
\]

它不是交付验收口径。

## 10. 输出

面板 payload：

- center、4 corners；
- visible EN/U/V；
- thermal EN/U/V；
- field_shift_px；
- source_valid；
- field_supported；
- target_valid；
- conservative_valid；
- component_conservative_valid；
- low_confidence；
- nearest_tile_m；
- field_weight。

payload 只按：

\[
\text{center.valid}=1
\]

排除原始五点全 nodata 的组件；保守字段供交付侧筛选。

摘要包括：

- panel_points；
- panel_points_valid；
- panel_points_conservative_valid；
- panel_points_corrected_valid；
- panel_points_unsupported；
- panel_points_target_nodata；
- panel_ids_valid；
- panel_ids_conservative_valid；
- panel_ids_all_original_nodata；
- grid 统计；
- field shift 统计。

`attempts[].displacement_domain_m` 包含：

- center_dx_m、center_dy_m；
- covariance_m2；
- semiaxis_m；
- radius、confidence、estimator。

每个 tile 记录还包含：

- candidate_mode_count；
- candidate_top_mass、candidate_second_mass；
- candidate_mode_separation_m；
- candidate_ambiguity。

## 11. 模型参数

| 参数 | 值 |
|---|---:|
| 公共网格 \(r\) | 0.1 m |
| 请求 tile 上限 | 40 m |
| tile 面积目标 | 图像 1% |
| prior 半径 | 2 m |
| 旧歧义 margin（诊断） | 0.02 |
| 候选 mode 质量下限 | 0.73 |
| 候选 mean-shift 带宽 | 2 px |
| \(\sigma_0\) | 0.05 m |
| 位移域 | 二维 KDE + mean-shift |
| 域置信度 | 0.995 |
| KDE 带宽系数 | 0.8 |
| 域最少观测 | 8 |
| 域协方差下限 | 0.25 m² |
| 场支撑 | 高斯置信 + \(2T\) 硬保护 |
| 支撑 \(C_{\min}\) | 无 |
| 覆盖归一化 | tile/8 格网 |
| 相似变换 | q-only 加权 IRLS |
| RBF 权重 | 协方差 GLS |
| RBF 正则 | coefficient ridge |
| 岭参数 \(\lambda\) | 0.1 |
| RBF 核长 | \(L=\max(2.5W_t,4\delta_0)\) |
| 边界衰减 | Gaussian \(\sigma=T\)，范围 \(2T\) |
| 面板映射 | rigid |
| 交付口径 | conservative_valid |

## 12. 控制流与失败处理

主路径只有一套算法，但保留必要的运行控制：

- 瓦片有效比例不足：跳过；
- 纹理不足或 ECC 非有限：跳过；
- accepted 少于 6：缩小 tile 重试；
- 二维模式簇过小：回退二维 median/MAD 椭圆；
- 场支撑不足：标记 unsupported；
- 面板源/场/目标无效：标记 conservative 失败；
- 无可用观测：抛错，不生成空场。

这些不是可选算法分支，而是单一路径内的失败处理和自适应行为。

## 13. 能力边界

必须区分：

\[
\text{内部可预测性}
\ne
\text{独立几何精度}
\]

当前模型：

- 不做绝对锚点校正；
- 不做 V5 原始热像投影；
- 不做 coherent alias 的完全消解；
- 不做外部几何精度认证；
- conservative_valid 只表示源、场、目标三态同时成立，不表示真实误差上界。

在缺少独立锚点时，模型输出应被视为相对配准候选，而不是绝对几何真值。

边界单侧支撑与边界外推问题见 [BOUNDARY_EXTRAPOLATION.md](BOUNDARY_EXTRAPOLATION.md)。
