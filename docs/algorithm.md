# 算法与实现约定

## Muon

来源：KellerJordan/Muon，MIT，提交 `f98f1cacc0263b04290753e32be8d498c1efc806`。

https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py

本项目采用单设备实现，使用 float32 进行 NS 运算；原作者该提交采用 bfloat16。未使用分布式同步或 torch.compile。相关代码按 MIT 保留许可，见 `third_party/Muon-LICENSE.txt`。

更新约定：

1. `B = momentum * B + (1-momentum) * gradient`，初值为零。
2. Nesterov 模式使用 `(1-momentum) * gradient + momentum * B`，不原地修改输入梯度。
3. 输入矩阵按 Frobenius 范数加 `ns_eps` 归一化；必要时转置以缩小 Gram 矩阵。
4. 经验多项式系数 `(3.4445, -4.7750, 2.0315)`，默认 5 步。
5. 可选更新尺度 `sqrt(max(1, rows/cols))`，语言实验默认开启，矩阵实验默认关闭。
6. 解耦衰减先执行 `W *= 1-lr*weight_decay`，再减去学习率乘更新方向。

`method="svd"` 是小矩阵用的参照变体，用紧致数值秩的极因子；零矩阵返回零，不人为补全零奇异值方向。它与经验系数有限步 NS 的实现不同，须单独命名与报告。

语言模型中，隐藏线性层的二维权重使用 Muon。所有 embedding、输出 head 及一维权重/偏置使用辅助 AdamW，输出 head 与 token embedding 共享权重，每个参数只归一个优化器。辅助 AdamW 的二维参数采用配置中的衰减系数，一维参数不衰减。纯 AdamW 基线采用相同的二维/一维衰减规则。

## AdamW

直接使用锁定版本的 `torch.optim.AdamW`，保留其偏差修正和根号外 epsilon。当前 betas 为 `(0.9, 0.95)`，不是声明这些参数满足某个收敛定理。

## 理论边界

当前工程不宣称已证明代码中的完整训练算法收敛。文献可能采用精确极因子、标准 NS 多项式、不同归一化稳定项或特定的动量/衰减/步长安排。后续主定理选定后，应逐一给出对应关系。

经验系数迭代的标量映射不以 1 为不动点，增加迭代次数并不必然单调逼近精确极因子。不能把标准 NS 的逼近定理直接套到该实现上。
