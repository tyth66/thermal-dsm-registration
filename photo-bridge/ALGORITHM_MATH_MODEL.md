# 可见光—热像 DSM 平面配准数学模型

版本：单一路径生产模型  
代码基线：03c0bb9  
测试：85 passed

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
- \(J(x)=J_v(x)\land J_t(x)\)：联合掩膜。

## 3. 公共网格与预处理

### 3.1 公共网格

两幅 DSM 必须具有一致投影米制 CRS。  
取两幅 DSM 包围盒交集，按分辨率 \(r=0.1\) m 重投影到同一网格。

### 3.2 有效掩膜

\[
J(x)=J_v(x)\land J_t(x)
\]

无效像元不参与相关，也不以零值参与归一化。

### 3.3 Gaussian 高通

对 DSM \(z(x)\)：

\[
h(x)=z(x)-\frac{(G_\sigma*(Jz))(x)}{(G_\sigma*J)(x)}
\]

其中：

\[
G_\sigma(x)=\frac{1}{2\pi\sigma^2}
\exp\left(-\frac{\|x\|^2}{2\sigma^2}\right)
\]

默认：

\[
\sigma=1.5\ \text{m}
\]

高通后无效像元置 0。

## 4. 分块与场支撑

### 4.1 自适应 Tile

请求值：

\[
W_t^{req}=40\ \text{m},\qquad W_s^{req}=20\ \text{m}
\]

短边为：

\[
S=\min(W,H)
\]

实际 tile：

\[
W_t=\min\left(W_t^{req},
\max(0.2S,8r)\right)
\]

步长：

\[
W_s=\min(W_s^{req},W_t/2)
\]

两者量化到整像素。

若第一次尝试 accepted 少于 6，则 tile/step 减半重试，最多 3 次；取 accepted 最多的尝试。

### 4.2 场支撑

\[
D_s=\min\left(30,\max(10,0.30S)\right)
\]

场支撑条件：

\[
\text{field\_supported}(p)=
[\text{finite }d(p)]
\land[\text{weight}(p)>0]
\land[d_{\min}(p)\le D_s]
\]

其中 \(d_{\min}\) 是到最近观测中心的距离。

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

高通纹理标准差不足时，该瓦片不产生观测。

### 5.2 粗初值

瓦片降采样一半，归一化后做 ECC，得到粗位移：

\[
s_0=\mathrm{ECC}_{1/2}(h_v,h_t)
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
C_v(s)=\sum_x J(x)J(x+s)h_v(x)
\]

\[
C_t(s)=\sum_x J(x)J(x+s)h_t(x+s)
\]

\[
C_{vv}(s)=\sum_x J(x)J(x+s)h_v(x)^2
\]

\[
C_{tt}(s)=\sum_x J(x)J(x+s)h_t(x+s)^2
\]

\[
C_{vt}(s)=\sum_x J(x)J(x+s)h_v(x)h_t(x+s)
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

### 5.4 候选与一致性

前向只保留 prior 窗口内 top-5：

\[
\|s-\hat s_0\|_2\le R_p
\]

反向使用：

\[
-\hat s_0
\]

反向一致性：

\[
e_{fb}=r\|s^++s^-\|_2
\]

\[
\text{consistent}=[e_{fb}\le0.20\ \text{m}]
\]

全局 top-1 用于 alias 冲突：

\[
d_{\rm global}>R_p
\quad\land\quad
\rho_{\rm global}>\rho_{\rm best}+0.05
\]

### 5.5 歧义

前向候选：

\[
\Delta\rho=\rho_1-\rho_2
\]

\[
d_s=r\|s_1-s_2\|_2
\]

由高通自相关估计主周期 \(P\)。

歧义：

\[
\text{ambiguous}=
\left[
\Delta\rho<0.02
\land
d_s\ge\max(1,0.5P)
\right]
\lor
\text{alias\_conflict}
\]

当只有一个候选时：

\[
margin=1.0
\]

这只是“没有第二候选可比较”，不是唯一性证明。

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
+e_{fb}^2I
+\sigma_0^2I
\]

\[
\sigma_0=0.05\ \text{m}
\]

该协方差只作为观测诊断保存；当前场拟合不使用 \(\Sigma_i\) 加权。

### 5.7 观测记录

每个瓦片产生：

- 中心 \(p_i=(e_i,n_i)\)；
- 位移 \(d_i=(dx_i,dy_i)\)；
- 目标 \(t_i=p_i+d_i\)；
- 相关系数 \(cc_i\)；
- 质量权重 \(qf_i=vf_i^{1.5}\)；
- consistent/ambiguous；
- margin/separation/period；
- candidate_count；
- 协方差字段；
- alias_conflict。

## 6. 位移域

### 6.1 mode 域

对每个轴，取有限位移值。

若样本少于 8，回退到稳健中位数/MAD。

否则：

1. 取 5% 与 95% 分位；
2. 直方图带宽：

\[
w=\max\left(0.25,\min\left(1,\frac{P_{95}-P_5}{12}\right)\right)
\]

3. 取最密直方图簇，并向两侧扩展一个 bin；
4. 若簇样本少于：

\[
\max(4,0.15N)
\]

回退到稳健中位数/MAD。

否则：

\[
c_j=\mathrm{median}(C_{\rm mode})
\]

\[
s_j=1.4826\cdot\mathrm{MAD}(C_{\rm mode})
\]

\[
b_j=\max(1.5,3s_j)
\]

### 6.2 域筛选

对每轴：

\[
|d_{ij}-c_j|\le b_j
\]

非有限位移不参与域估计，并作为无效测量被拒绝。

### 6.3 域统计

输出：

- center_dx、bound_dx；
- center_dy、bound_dy；
- rejected。

## 7. 观测集合

场拟合前，只保留：

\[
\text{accepted}=
\text{consistent}=1
\land
\text{ambiguous}=0
\]

域筛选在 accepted 之前执行。

## 8. 场模型

### 8.1 模型形式

\[
f(p)=Ap+b+g(p)
\]

\[
A=
\begin{pmatrix}
a & -b\\
b & a
\end{pmatrix}
\]

\(g\) 是 Gaussian RBF 残差。

位移：

\[
d(p)=f(p)-p
\]

\[
q=p+d(p)
\]

### 8.2 观测权重

\[
w_i=cc_i\cdot qf_i\cdot s_i
\]

其中：

\[
qf_i=vf_i^{1.5}
\]

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

\[
\min_{A,b}\sum_i w_i
\|t_i-(Ap_i+b)\|^2
\]

最终：

\[
\epsilon_i=\|t_i-(Ap_i+b)\|
\]

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
r_i=t_i-(Ap_i+b)
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

\[
(K^TW^{fit}K+\lambda I)c
=
K^TW^{fit}r
\]

默认：

\[
\lambda=0.1
\]

### 8.5 场求值

\[
g(p)=\sum_i c_i
\exp\left(
-\frac{\|p-p_i\|^2}{2L^2}
\right)
\]

\[
d(p)=Ap+b-p+g(p)
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
\left(f(p_j)-p_j\right)
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

## 11. 当前默认参数

| 参数 | 值 |
|---|---:|
| 公共网格 \(r\) | 0.1 m |
| 请求 tile/step | 40 / 20 m |
| 自适应 tile | 开启 |
| 高通 \(\sigma\) | 1.5 m |
| prior 半径 | 2 m |
| FB 容差 | 0.20 m |
| 歧义 margin | 0.02 |
| \(\sigma_0\) | 0.05 m |
| 位移域 | mode |
| 场支撑 | 自适应，上限 30 m |
| 相似变换 | q-only 加权 IRLS |
| RBF | coefficient ridge |
| 岭参数 \(\lambda\) | 0.1 |
| RBF 核长 \(L\) | 自动 |
| 面板映射 | rigid |
| 交付口径 | conservative_valid |

## 12. 控制流与失败处理

主路径只有一套算法，但保留必要的运行控制：

- 瓦片有效比例不足：跳过；
- 纹理不足或 ECC 非有限：跳过；
- accepted 少于 6：缩小 tile 重试；
- mode 簇过小：回退稳健中位数/MAD；
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
