# V3-B：统一搜索范围下的 Prior 唯一性诊断

状态：只读诊断；默认 V2 未改。  
代码基线：f65aea1  
测试：127 passed

## 1. 目的

Prior=1 m 的 accepted 从 39 增加到 56，其中 19/20 是单候选。  
本诊断不改变匹配结果，只检查：这些单候选在较大的 2 m 搜索范围内是否仍然唯一。

## 2. 方法

- 分别运行 prior=1 m 与 prior=2 m；
- 两个配置使用同一 tile/step/grid；
- 按瓦片中心匹配 prior1 的 accepted 与 prior2 的 records；
- 比较 candidate_count、margin、ambiguous、dx/dy。

注意：prior2 的 records 已经过 mode 域筛选；未匹配不代表 prior2 一定不接受。

## 3. 结果

prior1 新增 20 条 accepted：

| 指标 | 值 |
|---|---:|
| 新增 accepted | 20 |
| prior1 单候选 | 19 |
| 在 prior2 records 中匹配到 | 8 |
| 未匹配 | 12 |
| 匹配且 prior2 候选≥2 | 8 |
| 匹配且 prior2 ambiguous | 8 |
| 匹配且 prior2 candidate_count=1 | 0 |

结论：

- 8 条新增观测在 prior=2 下都有 2–4 个候选，并触发 ambiguous；
- 它们在 prior=1 下全部显示 margin=1.0；
- 这 8 条的高 margin 是**搜索窗口截断伪影**；
- 另外 12 条未匹配，可能是被 prior2 domain 过滤或没有记录，不能据此判断唯一性。

## 4. 示例

| 中心 | prior1 候选 | prior1 margin | prior2 候选 | prior2 margin | prior2 dx/dy | prior1 dx/dy |
|---|---:|---:|---:|---:|---|---|
| 594783.89, 3231405.81 | 1 | 1.000 | 3 | 0.012 | 0.968, −0.884 | −0.292, −0.917 |
| 594747.49, 3231400.61 | 1 | 1.000 | 4 | 0.003 | −1.129, 0.220 | 0.187, 0.242 |
| 594773.49, 3231405.81 | 1 | 1.000 | 2 | 0.004 | −0.811, −0.502 | −0.811, −0.502 |

前两条不仅候选被隐藏，prior1 还选择了与 prior2 不同的峰。  
这说明缩小窗口不仅“看不到竞争峰”，还可能改变最终峰选择。

## 5. 当前判断

- prior=1 的 accepted 增加，至少部分来自搜索窗口截断；
- 19/20 单候选不能解释为唯一性提高；
- 8/20 在 2 m 范围内明确歧义；
- 12/20 尚未有足够信息分类；
- 没有理由替换 prior=2 m。

## 6. 后续

1. 若继续研究，优先对未匹配的 12 条做独立重算，而不是继续调 prior；
2. 评价时同时报告 candidate_count 与 diagnostic-window 候选；
3. 不把 margin=1.0 当成唯一性证据；
4. 默认继续冻结。
