# V3-B 收官：可辨识性、可比较性与评价几何

状态：V3-B 结束；V2 默认继续冻结。  
代码基线：79e5242  
测试：129 passed

## 1. 最终状态

| 模块 | 状态 | 结论 |
|---|---|---|
| V2 | 冻结 | 默认行为不变 |
| V3-A | 第一轮完成 | 已建立三站几何指纹与无量纲指标 |
| V3-B 参数敏感性 | 完成 | 未找到可替换默认的稳定优势 |
| Prior=1 归因 | 完成 | accepted 增益部分来自搜索窗口截断 |
| 单候选唯一性 | 完成 | margin=1.0 不等于唯一性 |
| 严格足迹评价 | 不可执行 | 当前几何条件与支撑半径冲突 |
| V3-C | 暂停 | 不启动 L/lambda 自适应 |

## 2. 三个正式结论

### 结论一：站点特征有差异，但没有参数映射

141/247/290 在覆盖、GSD、周期、纹理、观测数量上明显不同。  
但 Tile/Prior/Grid/Sigma 的受控实验没有产生同时满足误差与覆盖的稳定候选。

因此不能从：

\[
F_s
\]

直接推出：

\[
\theta_s^\star=\mathcal A(F_s)
\]

当前没有足够证据建立这个映射。

### 结论二：参数变化改变的是可观测性，不只是误差

Prior=1 m：

- accepted 39→56；
- 新增 20 条中 19 条单候选；
- 其中 8 条在 prior=2 m records 中有 2–4 个候选，且全部 ambiguous；
- 8/8 在 prior=1 下 margin=1.0；
- 12 条未匹配，无法分类。

这说明：

\[
\text{缩小搜索窗口}
\Rightarrow
\text{竞争峰不可见}
\Rightarrow
\text{歧义未被检出}
\]

默认 prior=2 的 39 条 accepted 中也有 23 条单候选，说明单候选问题不是 prior=1 独有。

因此必须区分：

\[
\text{没有发现竞争峰}
\quad\neq\quad
\text{经过充分搜索后唯一}
\]

### 结论三：严格评价协议受几何条件约束

同尺寸轴对齐方形瓦片、缓冲 b、支撑半径 D_s：

\[
\boxed{
D_s < W_t+2b
\Longrightarrow
\text{严格足迹隔离后不可能有受支持评价点}
}
\]

三站 b=6 m：

| 站 | W_t | W_t+2b | D_s | 结果 |
|---|---:|---:|---:|---|
| 141 | 40.0 | 52.0 | 30.0 | 不可能 |
| 247 | 10.4 | 22.4 | 15.66 | 不可能 |
| 290 | 35.5 | 47.5 | 30.0 | 不可能 |

这不是观测稀疏的偶然现象。  
141/290 即使 b=0 也已不满足；247 的最大正缓冲为 2.63 m。

## 3. 因此得到的三类边界

\[
\boxed{
\begin{gathered}
\text{观测是否具有可辨识性}\\
\text{参数收益是否具有可比较性}\\
\text{评价协议是否具有数学可行性}
\end{gathered}
}
\]

这三个问题优先于继续降低内部误差。

## 4. 不声称的结论

- 不声称 prior=2 m 在真实几何上一定更准；
- 不声称 prior=1 m 一定错误；
- 不声称三站永远无法评价；
- 不声称当前场模型无效；
- 不声称内部误差下降等于几何精度提升。

## 5. 冻结默认

\[
\boxed{
r=0.1,\quad W_t=10.4,\quad R_p=2,\quad \sigma=1.5
}
\]

- 场：q-only similarity + RBF + coefficient ridge；
- 面板：rigid；
- 交付：conservative_valid；
- V3-C 不启动。

## 6. 重新启动条件

只有在以下证据出现时，才值得重启参数自适应：

1. 新站点或新数据显著提高独立观测密度；
2. 新的评价设计能在不扩大 D_s 的情况下形成受支持留出；
3. 有独立几何锚点或合成已知位移验证；
4. 新的参数证据能同时满足覆盖约束和可比较性。

## 7. 相关文档

- V3A_SITE_GEOMETRY_PROFILER.md
- V3B_247_OBSERVATION_PARAMETERS.md
- V3B_FIXED_EVALUATION_247.md
- V3B_FOOTPRINT_EVALUATION.md
- V3B_PRIOR_UNIQUENESS_DIAGNOSTIC.md
- V3B_FEASIBILITY_REPORT.md
- VERSION_FREEZE.md
