# Muon 课程论文与实验复现

本项目研究 Muon 的矩阵更新机制，以 AdamW 为实验基线。正式协议固定于2026-11-04；理论分析重构 arXiv:2609.33152v1，工程实验的学习率调度、裁剪、矩形尺度与辅助 AdamW 单独报告。

## 环境和默认预览

在 PowerShell 中执行：

```powershell
Set-Location E:\Optimization_Course
.\scripts\setup.ps1
.\.venv\Scripts\python.exe -m pytest -q
.\scripts\run_study.ps1
```

最后一条只输出 NOT_RUN 计划，不下载数据或训练。环境锁定 Python3.12、PyTorch2.7.1+cu128、NumPy1.26.4、Matplotlib3.9.4。普通测试跳过真实训练；显式端到端恢复检查使用 `python -m pytest -q --run-training`，其中训练仅在临时合成语料和微型 CPU 模型上进行，不作为论文实验结果。

## 正式实验

首次显式准备固定提交的 Tiny Shakespeare：

```powershell
.\scripts\prepare_data.ps1 -DownloadShakespeare
.\scripts\run_study.ps1 -Execute
```

数据准备拒绝覆盖已有目录。再次运行 study 会复用配置与身份一致的完成任务；不要为了重跑而修改原清单或覆盖数据。

正式结果存于 `runs/course_v1/`：`study_state.json` 保存预期任务、配置身份、状态与选择；`gpu_budget.json` 是跨阶段共享的28800秒账本；`configs/` 是运行前冻结的有效配置；`matrix/`、`language/` 保存原始记录和检查点；`reports/` 保存对照图和统计。失败和中断消耗也计入账本；异常退出未关闭的会话按经过的墙钟时间保守扣除，不能通过重启清零。

可用 `-Stage matrix/pilot/tuning/final/evaluate/report` 分阶段执行。依赖尚未完成时命令会停止；语言阶段的数据指纹必须保持一致。正式模型全部训练完后才运行测试集评估，缓存评估绑定数据、配置和检查点校验值。

## 复现论文的固定配置

标准study流程会根据机器速度重新校准并筛选；如果要复现本文的2000步正式设置，准备好相同语料后使用：

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_frozen.py
.\.venv\Scripts\python.exe scripts/reproduce_frozen.py --execute
```

第一条只预览且不创建结果目录。第二条在新的 `runs/reproduction_v1` 下直接重放论文的45个矩阵与9个语言任务、评估和报告，复用已冻结学习率，不再调参或按机器速度缩短长度；同样执行28800秒上限，较慢机器可能在预算内无法完成，状态会如实保留。再次恢复可用 `optcourse study --root runs/reproduction_v1 --stage final --execute` 等现有入口，不要覆盖清单。

原研究完成后，`python scripts/audit_results.py` 从原始记录核对54个正式运行和9个测试结果；`python scripts/write_paper_results.py` 生成论文结果段、表和图。公开仓库附原始指标和评估摘要，检查点只在本机保留；完整重新计算测试损失需重放训练。

| 阶段 | 数量与设置 |
|---|---|
| 矩阵筛选 | 3条件数×2方法×3学习率，种子101，各300步，18次 |
| 矩阵正式 | 3条件数×5方法×3种子，各1000步，45次 |
| 语言校准 | AdamW/NS5/NS3，种子99，各100步 |
| 语言筛选 | AdamW和NS5各3学习率，种子101，各300步，6次 |
| 语言正式 | AdamW/NS5/NS3×种子0/1/2，9次 |

矩阵为32×16、条件数1/10/100、无噪声无衰减、CPU单线程；Muon各变体共享NS5筛选出的学习率。语言为4823552参数、6层、宽256、4头、上下文128，每更新4096字节。用校准实测时间乘1.3，从2000/1000/500选择使调参与正式训练预计不超过6小时的最大长度，其余预算留给校准与评估。正式语言任务按种子轮转方法顺序，避免固定顺序影响。

验证与测试连续分块，每个可预测目标计分一次，尾块按目标数加权、块边界重置上下文。每100步完整验证，验证选最佳模型。图表只读取正式清单，报告均值、样本标准差、完成分母和失败记录，不把校准、调参与恢复测试混入正式样本。

## 论文与证据

`paper/main.tex` 引用 `paper/theory.tex` 的完整推导，以及根据正式记录生成的结果表和图。编译：

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build -cd paper/main.tex
```

理论保证针对明确假设和参数安排下的单矩阵算法；小模型结果只支持所列语料、模型和预算。表格时间为任务各尝试累计墙钟时间，包含构建、验证、保存及重跑。时间曲线使用可恢复的优化轨迹时钟，不包含构建、末次保存与被回滚工作，不能与表格总时间混用。显存为PyTorch峰值已分配量；每次GPU任务记录启动时整卡利用率和空闲显存，不把有桌面负载的时间解释为隔离环境性能基准。

单次评估入口默认预览；显式执行用 `python -m optcourse evaluate --run 运行目录 --execute`。CUDA正式任务（含直接run/batch入口）必须使用所属study的28800秒账本，缓存测试结果也必须附已关闭会话的记账来源。验证中断会保存已完成更新及待评估标记，恢复先完成该次验证，再继续下一次更新。

Muon代码改写自KellerJordan/Muon的MIT许可版本，保留 `third_party/Muon-LICENSE.txt`。公开发布通过干净导出创建，不包含原始课程PPT、下载文献、缓存、虚拟环境或模型检查点。
