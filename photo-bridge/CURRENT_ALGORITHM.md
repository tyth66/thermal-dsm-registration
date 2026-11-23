# 当前正式算法（run 296）

更新时间：2026-11-23  
测试：89 passed

## 1. 算法链路

可见光/热像 DSM
→ 公共米制网格
→ 1% 面积 tile / 50% 重叠
→ masked ECC
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
| 高通 | 关闭（0） |

完整 tile 使用完整窗口；边界 tile 由有效 data 轮廓向内生成，只使用 mask 有效区域。

## 3. 核心场

f(p)=Ap+b+g(p)

d_c(p)=f(p)-p

- A：旋转 + 尺度相似变换；
- b：平移；
- g：Gaussian RBF 残差；
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

edge_confidence 仍计算并输出，但不再参与 conservative 判定。

conservative = source_valid AND field_supported AND target_valid

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
| edge low confidence panels | 3190 |

## 8. 正式文件

outputs/run296/final/296_mapping_result.json  
outputs/run296/final/296_panel_mapping.csv  
outputs/run296/final/296_summary.json  
outputs/run296/visualizations_final_v2/296-final-v2_full_arrays.png

## 9. 已知限制

1. 边界残差仍是核心场样本内残差，尚未做 grouped cross-fit；
2. 顶部/底部修正后仍有残余：top +1.7 cm，bottom +5.5 cm；
3. 边界局部模型仍是 6 系数、最少 4 观测、ridge 0.01，可辨识性偏弱；
4. 当前 edge_confidence 不参与交付，只保留诊断。
