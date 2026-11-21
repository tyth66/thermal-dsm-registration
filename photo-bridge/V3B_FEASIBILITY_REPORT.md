# V3-B：三站严格足迹评价可行性报告

状态：只读评价；默认 V2 未改。  
代码基线：f65aea1  
测试：127 passed

## 1. 协议

- 固定评价点：各站当前默认 accepted；
- 固定四象限；
- 方形瓦片足迹重叠排除；
- 高通影响缓冲 b=6 m，约 4 sigma；
- 训练候选必须位于其余三象限，且与留出评价点足迹不相交；
- 评价点是留出象限中仍有 field_supported 的点。

可行性字段：

- reference_count；
- footprint_disjoint_count：留出折中剩余训练观测的最小值；
- buffered_supported_count；
- valid_fold_count；
- common_supported_count；
- reference_source；
- evaluation_feasible。

feasibility 规则：

\[
\text{feasible}=
[\text{reference}>0]
\land[\text{footprint}>0]
\land[\text{supported}>0]
\land[\text{valid\_folds}=\text{folds}]
\land[\text{common}>0]
\]

这是最低执行条件，不是统计功效证明。

## 2. 结果

| 站 | accepted/reference | tile | 每折训练数 | 支持点 | 有效折 | feasible |
|---|---:|---:|---|---:|---:|---|
| 141 | 89 | 40.0 m | 40 / 56 / 37 / 48 | 0 | 0 | false |
| 247 | 39 | 10.4 m | 6 / 26 / 6 / 32 | 0 | 0 | false |
| 290 | 31 | 35.5 m | 15 / 13 / 11 / 31 | 0 | 0 | false |

290 的第 4 折留出点为 0，因此实际有效折更少。

reference_source：V2 默认 accepted ECC 观测。  
buffer_m：6.0。

## 3. 解释

- 6 m 缓冲后，三站都没有任何留出点仍被场支持；
- 247 是最稀疏的，但 141/290 也失败，说明这不是 247 个例；
- 这是当前“tile/step + 场支撑 + 4 sigma 缓冲”协议下的结论，不是“这些站永远无法评价”；
- 不能为了让评价能算而临时扩大 D_s，否则评价对象变成外推；
- 当前严格足迹协议不可执行，不能用于参数选择。

## 4. 对 V3 的含义

V3 必须把“参数能否被可靠比较”作为输出：

\[
\text{data}\rightarrow\text{evaluation feasibility}\rightarrow
\begin{cases}
\text{allow adaptive choice}\\
\text{keep frozen V2}
\end{cases}
\]

对这三站，当前应输出：

\[
\boxed{\text{参数选择证据不足}\Rightarrow\text{保留 V2}}
\]

## 5. 下一步

1. 若继续严格评价，需要提高观测密度或改变评价单元；
2. 141/290 可先做 buffer 0/2/6 敏感性，确认是否只是缓冲过严；
3. 用合成已知位移验证变换等变性，但不能替代真实几何；
4. 不再扩大 Tile/Prior/Grid/Sigma 搜索；
5. V3-C 不启动。

## 6. 实现

- 评价器：registration_algorithms/evaluation.py
- 可行性报告辅助：evaluation_feasibility
- 测试：127 passed
