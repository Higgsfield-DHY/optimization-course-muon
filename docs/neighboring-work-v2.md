**相邻工作与本研究的比较对象**

当前核心问题：更新RMS和辅助优化一致时，有限NS候选的谱误差、梯度对齐及方向曲率如何排序下降收益；当次目标反馈如何影响深度选择。

| 工作 | 已建立的对象 | 本研究采用的比较方式 |
|---|---|---|
| [Convergence of Muon with Newton–Schulz](https://arxiv.org/abs/2601.19156) | 有限NS下的驻点收敛及近似常数 | 使用其有限矩阵变换视角，测量给定状态上的目标响应 |
| [Convergence of Practical Muon](https://arxiv.org/abs/2609.33152) | 经验系数、动量与解耦衰减的分析 | 保留实际多项式与完整更新，提出同动量曲率置换实验 |
| [Why Muon Outperforms Adam: A Curvature Perspective](https://arxiv.org/abs/2606.04662) | 一阶下降与二阶曲率代价、归一化方向锐度 | 对有限NS候选进行完整HVP测量，并用独立数据检验排序 |
| [Spectral Scaling Laws of Muon](https://arxiv.org/abs/2606.04058v2) | 跨层、跨模型规模的动量谱规律 | 记录实际动量谱与有限深度误差，加入两个规模的训练 |
| [Cost-free Spectral Estimation for Adaptive NS](https://arxiv.org/abs/2609.33047) | 从NS内生矩估计谱分布，依谱选择矩阵函数程序 | 将谱误差排序与目标对齐、曲率和完整损失响应并置 |
| [Muon with Finite NS: The Smoothing Benefit](https://arxiv.org/abs/2608.26288) | 有限深度谱变换的平滑作用和非光滑优化分析 | 以实际误差测量区分经验深度及近似质量，配套经典NS参考 |
| [Spectrally Targeted Muon](https://arxiv.org/abs/2610.10965) | 针对特定奇异方向的干预和训练分析 | 固定更新能量，以同状态完整步长比较有限谱变换 |
| [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) | 更新尺度、衰减与大规模训练 | 采用共同RMS及标准尺度消融，统一辅助参数设置 |

本研究的证据组织为：精确曲率反转与相图；GPU实际候选的冻结数据；独立批次的一阶/二阶排序；固定深度、目标反馈与阶段频率匹配重放的完整训练。主要判断来自这些对照的结果。
