# 全原始候选联合模型（研究原型）

状态：**实验性研究模型，尚未进入生产**。正式生产算法见 [CURRENT_ALGORITHM.md](CURRENT_ALGORITHM.md)。

更新时间：2026-11-25

测试：`113 passed`

## 1. 目标

输入是可见光 DSM 与热像 DSM 的 tile 级 masked ECC 候选。每个 tile 保留全部原始候选，不先做硬筛选：

\[
S_i=\{(s_{ij},\rho_{ij},C_{ij})\}_{j=1}^{K_i}
\]

模型同时估计：

- 连续位移场 \(f_\theta(p)\)；
- tile 是否为 outlier；
- tile 内哪个候选是真实对应；
- 场模型误差 \(\sigma_f\)；
- 候选歧义造成的后验信息损失。

## 2. 四层生成模型

### Layer 1：场先验

\[
\theta\sim N(\theta_0,\Lambda^{-1})
\]

中心化位移场：

\[
f_\theta(p)=B(p-p_c)+t_0+\sum_k c_k\phi_k(p)
\]

\[
B=A-I=\begin{pmatrix}u&-v\\v&u\end{pmatrix}
\]

坐标映射：

\[
T_\theta(p)=p+f_\theta(p)
\]

零位移状态为 \(B=0,t_0=0,c_k=0\)。

### Layer 2：真实位移

\[
d_i\mid\theta\sim N(f_\theta(p_i),\sigma_f^2I)
\]

### Layer 3：潜在对应

\[
P(z_i=0)=a_{i0},\qquad
P(z_i=j)=(1-a_{i0})\pi_{ij}
\]

### Layer 4：候选集合观测

outlier tile：

\[
p(S_i\mid z_i=0,d_i)=\prod_k u_i(s_{ik})
\]

候选 \(j\) 为真实对应：

\[
p(S_i\mid z_i=j,d_i)=
N(s_{ij};d_i,C_{ij})
\prod_{k\ne j}u_i(s_{ik})
\]

## 3. 边缘化

对 \(d_i\) 做高斯卷积：

\[
V_{ij}=C_{ij}+\sigma_f^2I
\]

\[
\int N(s_{ij};d,C_{ij})N(d;f_i,\sigma_f^2I)dd
=N(s_{ij};f_i,V_{ij})
\]

对 \(z_i\) 求和并提取公共背景因子：

\[
p(S_i\mid\theta)=
\left[\prod_k u_i(s_{ik})\right]
\left[a_{i0}+(1-a_{i0})\sum_j\pi_{ij}B_{ij}\right]
\]

\[
B_{ij}(\theta)=
\frac{N(s_{ij};f_\theta(p_i),V_{ij})}{u_i(s_{ij})}
\]

全原始联合后验：

\[
p(\theta\mid\{S_i\})\propto p(\theta)
\prod_i\left[\prod_k u_i(s_{ik})\right]
\left[a_{i0}+(1-a_{i0})\sum_j\pi_{ij}B_{ij}\right]
\]

公共因子与 \(\theta\) 无关，MAP 时可略去。

## 4. MAP 目标

\[
\mathcal L(\theta)=
\frac12(\theta-\theta_0)^T\Lambda(\theta-\theta_0)
-\sum_i\log\left[a_{i0}+(1-a_{i0})\sum_j\pi_{ij}B_{ij}\right]
\]

## 5. 两层 E 步

定义 tile 整体证据：

\[
E_i=\sum_j\pi_{ij}B_{ij}
\]

第一层 tile 有效性：

\[
m_i=\frac{(1-a_{i0})E_i}{a_{i0}+(1-a_{i0})E_i}
\]

\[
\log\frac{m_i}{1-m_i}=
\log\frac{1-a_{i0}}{a_{i0}}+\log E_i
\]

第二层候选竞争：

\[
w_{ij}=\frac{\pi_{ij}B_{ij}}{E_i}
\]

\[
q_{ij}=m_iw_{ij},\qquad q_{i0}=1-m_i
\]

## 6. 熵分解

\[
H_i^{\rm valid}=-m_i\log m_i-(1-m_i)\log(1-m_i)
\]

\[
H_i^{\rm alias}=-\sum_jw_{ij}\log w_{ij}
\]

\[
H_i^{\rm total}=H_i^{\rm valid}+m_iH_i^{\rm alias}
\]

## 7. 精确 M 步

\[
W_i=\sum_jq_{ij}V_{ij}^{-1}
\]

\[
h_i=\sum_jq_{ij}V_{ij}^{-1}s_{ij}
\]

\[
f_\theta(p_i)=\Phi_i\theta
\]

\[
\Phi_i=\begin{pmatrix}
x_i&-y_i&1&0&\phi_{i1}&0&\cdots\\
y_i&x_i&0&1&0&\phi_{i1}&\cdots
\end{pmatrix}
\]

\[
H=\sum_i\Phi_i^TW_i\Phi_i+\Lambda,\qquad
b=\sum_i\Phi_i^Th_i+\Lambda\theta_0
\]

\[
H\theta=b
\]

正式求解器不再使用 Moment Matching；Moment Matching 仅保留为诊断。\(W_i\) 已经包含 \(m_i\)，不能再额外乘一次。

## 8. \(\sigma_f\) 的 M 步

\[
\mathcal L_\sigma(t)=
\frac12\sum_{i,j}q_{ij}
\left[\log|V_{ij}(t)|+r_{ij}^TV_{ij}(t)^{-1}r_{ij}\right]
\]

\[
V_{ij}(t)=C_{ij}+tI,\qquad t=\sigma_f^2
\]

在 \([t_{\min},t_{\max}]\) 内做有界 MAP 优化。各向同性时：

\[
\hat\sigma_f^2=\max\left(0,
\frac{\sum_{i,j}q_{ij}\|r_{ij}\|^2}{2\sum_i m_i}-c\right)
\]

有效质量不足时冻结为先验值，不强行求解。

## 9. 确定性退火

退火只作用于 E 步后验：

\[
q_{iz}^{(\tau)}\propto\exp(\ell_{iz}/\tau)
\]

\[
\ell_{i0}=\log a_{i0}
\]

\[
\ell_{ij}=\log(1-a_{i0})+\log\pi_{ij}+\log B_{ij}
\]

最终 \(\tau\to1\)，再执行标准 MAP-EM。\(\sigma_f\) 与退火温度分离。

## 10. 后验曲率

\[
v_{ij}=V_{ij}^{-1}(s_{ij}-f_i),\qquad v_{i0}=0
\]

\[
\mathcal I_i=W_i-D_i
\]

\[
D_i=\sum_jq_{ij}v_{ij}v_{ij}^T
-\left(\sum_jq_{ij}v_{ij}\right)
\left(\sum_jq_{ij}v_{ij}\right)^T
\]

\[
H_{\rm obs}=\Lambda+\sum_i\Phi_i^T\mathcal I_i\Phi_i
\]

\[
\Sigma_f(p)=\Phi(p)H_{\rm obs}^{-1}\Phi(p)^T
\]

\[
\Sigma_d(p)=\Sigma_f(p)+\sigma_f^2I
\]

这是单个后验模式附近的局部 Laplace 不确定性，不覆盖 coherent alias 的全局错配模式。

## 11. 候选协方差

当前生产/默认路径使用归一化逆 Hessian。可选 DSM profile 协方差：

\[
Q_{ij}=-\nabla_s^2\rho_i(s_{ij})
\]

\[
\beta_{ij}^{\rm eff}=
\frac{\operatorname{SST}_{v,i}}{\sigma_z^2}\rho_{ij}
\]

\[
C_{ij}=(\beta_{ij}^{\rm eff}Q_{ij})^{-1}
\]

候选级路线保持 \(\pi_{ij}\) 独立，\(C_{ij}\) 只进入 \(V_{ij}\)；不把 \(|C_{ij}|^{1/2}\) 重复加进 \(\pi_{ij}\)。完整相关面 Laplace 路线尚未启用。

## 12. 算法结构

```text
初始化场（top1 候选）
重复：
  E 步：B_ij -> E_i -> m_i, w_ij, q_ij
  可选温度 τ
  精确 M 步：W_i, h_i -> Hθ=b
  更新 σ_f：有界 MAP-MLE
收敛
输出 f, Σ_f, support, ambiguity, entropy
```

当前只从 top1 初始化，没有多假设模式比较；coherent alias 不能由单模式局部后验解决。

## 12.1 端到端调用

研究包现在提供与生产 `estimate_planimetric` 同级的直接入口：

```python
import json
from registration_algorithms.research import (
    ResearchPlanimetricInputs, estimate_research_planimetric,
)

world = json.load(open("world_model.json"))
result = estimate_research_planimetric(ResearchPlanimetricInputs(
    visible_dsm="visible_dsm.tif",
    thermal_dsm="thermal_dsm.tif",
    visible_world=world,
    tile_m=40.0,
    step_m=20.0,
    max_iterations=6,
    dsm_covariance=False,
))

result.field
result.history
result.responsibilities
result.components
result.sigma_f_m
```

该入口内部完成：

```text
DSM 读取 / 公共网格
→ 自适应 tiling
→ register(..., apply_domain=False) 收集全部原始候选
→ fit_candidate_field_em
→ 最终 sigma/责任
→ panel components
```

`dsm_covariance=True` 时使用 `research.dsm_covariance.dsm_candidate_covariance` 回调；生产入口不导入该包。

### 完整内存映射

需要继续做 panel/grid 映射时，使用：

```python
from registration_algorithms.research import (
    ResearchPipelineInputs, run_research_pipeline,
)

pipeline = run_research_pipeline(ResearchPipelineInputs(
    planimetric=inputs,
    visible_ortho="visible_ortho.tif",
    thermal_ortho="thermal_ortho.tif",
    grid_step=16,
))

pipeline.planimetric.field
pipeline.mapping.panel_rows
pipeline.mapping.grid
pipeline.mapping.summary
```

`run_research_pipeline` 不写 JSON，全部在内存中返回。

## 13. 当前数值状态

以下结果是研究原型，不是生产验收：

```text
296 全 tile，精确 M 步 + 归一化协方差：
  field mean 0.1085 m
  panel mean 0.1092 m

296 含 boundary，精确 M 步：
  panel mean 0.1206 m

141，精确 M 步：
  field mean 0.1472 m

Moment Matching 诊断（296 boundary）：
  panel mean 0.0463 m
```

精确 M 步降低了 Q 目标，但相对当前 accepted-only 场的 panel 差异更大；这不是求解器符号错误，而是精确候选协方差/ridge 的标定问题。

\(\sigma_f\) MAP-MLE 在 296 上收缩到接近下界：

```text
归一化协方差：0.0541 m
DSM 协方差：0.0883 m
```

说明 \(\sigma_f\) 与 outlier 责任弱可辨识，需要更强先验、有效质量下限或外部标定。

DSM profile 协方差的原始 \(\beta_{\rm eff}\) 中位数约 \(7\times10^5\)，协方差 trace 中位数约 \(0.003\ \text{px}^2\)，说明 i.i.d. 高程噪声假设过度自信，尚未做空间自相关/有效样本数修正。

## 14. 测试与代码

- 测试：`113 passed`
- [research/candidate_field.py](registration_algorithms/research/candidate_field.py)
- [research/dsm_covariance.py](registration_algorithms/research/dsm_covariance.py)
- [masked_ecc.py](registration_algorithms/masked_ecc.py)
- [residual_model.py](registration_algorithms/residual_model.py)
- [tile_outlier_em_validation.json](outputs/domain_validation/tile_outlier_em_validation.json)

## 15. 能力边界

- 候选集合条件独立；重叠 tile 不满足严格独立联合，需要加权复合似然；
- \(\pi_{ij}\) 是条件候选权重，不是完整 ECC 峰生成过程；
- \(u_i\)、\(a_{i0}\)、\(\beta\)、\(\sigma_f\)、\(\Lambda\) 仍需外部标定；
- 局部 Laplace 不覆盖 coherent alias；
- 边界支撑不足时，\(\Sigma_f\) 小可能只是先验强；
- 当前未切换生产。