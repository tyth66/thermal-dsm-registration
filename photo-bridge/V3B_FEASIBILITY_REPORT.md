# V3-B：三站严格足迹评价可行性报告

状态：只读评价；默认 V2 未改。  
代码基线：79e5242  
测试：129 passed

## 1. 几何必要条件

两个同尺寸轴对齐方形瓦片，宽度 W_t，中心差 (dx,dy)。  
不相交条件：

\[
|dx|\ge W_t+2b \quad\lor\quad |dy|\ge W_t+2b
\]

因此中心距离：

\[
d_{\min}\ge W_t+2b
\]

场支撑要求：

\[
d_{\min}\le D_s
\]

所以严格隔离评价点仍可能被支持的必要条件：

\[
\boxed{D_s\ge W_t+2b}
\]

若：

\[
D_s<W_t+2b
\]

则无论观测如何分布，严格隔离后的评价点都不可能获得场支撑。

## 2. 三站代入

b=6 m：

| 站 | W_t | W_t+2b | D_s | 几何相容 |
|---|---:|---:|---:|---|
| 141 | 40.0 | 52.0 | 30.0 | 否 |
| 247 | 10.4 | 22.4 | 15.66 | 否 |
| 290 | 35.5 | 47.5 | 30.0 | 否 |

即使 b=0：

| 站 | W_t | D_s | 几何相容 |
|---|---:|---:|---|
| 141 | 40.0 | 30.0 | 否 |
| 247 | 10.4 | 15.66 | 是 |
| 290 | 35.5 | 30.0 | 否 |

247 允许正缓冲的最大值：

\[
b_{\max}=\frac{15.66-10.4}{2}=2.63\ \text{m}
\]

因此：

- 141/290 在 b=0 时已经几何不相容；
- 247 在 b>2.63 m 后几何不相容；
- 三站 6 m 缓冲出现 0 支持点是确定性结果，不是观测稀疏的偶然现象。

## 3. 可行性报告

| 站 | reference | 每折训练数 | 支持点 | 有效折 | reason |
|---|---:|---|---:|---:|---|
| 141 | 89 | 40 / 56 / 37 / 48 | 0 | 0 | geometric_incompatibility |
| 247 | 39 | 6 / 26 / 6 / 32 | 0 | 0 | geometric_incompatibility |
| 290 | 31 | 15 / 13 / 11 / 31 | 0 | 0 | geometric_incompatibility |

字段：

- reference_count；
- footprint_disjoint_count；
- buffered_supported_count；
- valid_fold_count；
- common_supported_count；
- reference_source；
- buffer_m；
- feasibility_reason；
- evaluation_feasible。

## 4. 预检查

新增 geometry_precheck(support_m, tile_m, buffer_m)：

- 返回 required_min_distance_m；
- 返回 geometrically_compatible；
- evaluation_feasibility 在训练前即可返回 geometric_incompatibility；
- 不需要运行 DSM 配准即可判定。

## 5. 解释边界

- 这个条件只适用于当前同尺寸、轴对齐方形瓦片、中心距离支撑规则；
- 若未来采用不同窗口形状、不同支撑定义或外推评价，必须重新推导；
- 这不是“三站永远无法评价”，而是“当前协议不可执行”；
- 不能为了 CV 能算而临时扩大 D_s，否则评价对象变为外推。

## 6. 当前裁决

\[
\boxed{\text{严格评价不可执行}\Rightarrow\text{保留 V2}}
\]

- V2 继续冻结；
- 不再扩大 Tile/Prior/Grid/Sigma 搜索；
- V3-C 不启动；
- 需要新数据或新评价设计才重启参数自适应研究。

## 7. 实现与测试

- evaluation.py：geometry_precheck，feasibility_reason；
- 测试：129 passed；
- 纯数学回归用例：
  - 247 b=6：required 22.4 > 15.66，不可行；
  - 边界 22.4 = 22.4，可行；
  - 247 b=0/2 可行，b=6 不可行。
