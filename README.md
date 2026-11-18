# Muon有限步正交化的下降几何

论文研究有限 Newton–Schulz 深度对目标下降的作用。曲率置换保持梯度、动量和谱误差一致，直接改变候选方向的优化收益。语言模型实验比较了等幅度的 NS1、NS3、NS5、基于损失的深度选择和基于谱误差的深度选择。

## 文件

| 文件 | 内容 |
|---|---|
| `paper/v2_main.tex` | 论文主稿及各节源文件 |
| `paper/figures_v2/` | 矩阵相图、冻结状态测量、训练曲线和计时图 |
| `src/optcourse/research/` | 矩阵方向、优化器、训练、探测与报告代码 |
| `results/research_v2/` | 发布包内的冻结配置、身份记录、逐种子指标和各图数据 |
| `scripts/audit_research_v2.py` | 核对配对控制、训练端点、选模、目标计数、样本完整性和记账 |

## 环境和数据

本文使用 Windows、Python 3.12、PyTorch 2.7.1+cu128 与 RTX 3070 Ti Laptop GPU。依赖由 `uv.lock` 固定。

```powershell
uv sync --locked --extra research --extra dev
.\.venv\Scripts\python.exe scripts/prepare_research_v2.py
.\.venv\Scripts\python.exe scripts/prepare_research_v2.py --execute
```

准备脚本按修订版本下载 WikiText103 和 TinyStories 的五个源文件，校验长度和 SHA256，再用 GPT-2 BPE 打包。语料、分割、编码版本、源文件校验值和输出校验值均记录在数据清单中。固定训练前缀各含 50M token。

## 运行研究

以下命令从代码仓库根目录执行。默认研究入口显示配置。执行阶段接收 `--execute`。原始训练目录的名称为 `runs/research_v2`。

```powershell
.\.venv\Scripts\python.exe -m optcourse.research.protocol
.\.venv\Scripts\python.exe -m optcourse.research.matrix --execute
.\.venv\Scripts\python.exe -m optcourse.research.matrix --extended --output runs/research_v2/synthetic_extended --execute
.\.venv\Scripts\python.exe -m optcourse.research.report --synthetic
.\.venv\Scripts\python.exe -m optcourse.research.protocol --stage pilot --execute
.\.venv\Scripts\python.exe -m optcourse.research.protocol --stage freeze --execute
.\.venv\Scripts\python.exe -m optcourse.research.protocol --stage tuning --execute
.\.venv\Scripts\python.exe -m optcourse.research.protocol --stage formal --execute
.\.venv\Scripts\python.exe -m optcourse.research.probes --execute
.\.venv\Scripts\python.exe -m optcourse.research.protocol --stage test --execute
.\.venv\Scripts\python.exe -m optcourse.research.protocol --stage profile --execute
.\.venv\Scripts\python.exe -m optcourse.research.protocol --stage schedule-control --execute
.\.venv\Scripts\python.exe scripts/audit_research_v2.py
.\.venv\Scripts\python.exe -m optcourse.research.publication
latexmk -cd -xelatex -interaction=nonstopmode -halt-on-error -outdir=build_v2 paper/v2_main.tex
```

复现论文的既定参数时，将归档的 `frozen_protocol.json` 和 `selections.json` 复制到新结果目录，再从 `formal` 阶段执行，使用 `--root` 指定目录。学习率筛选和正式种子在配置中分别登记。训练按 `latest.pt` 恢复，完整测试从验证集选定的 `best.pt` 读取。冻结测量直接使用更新提交前保存的权重、动量和三组完整候选。CPU 探测可以用 `--wait` 在训练进行时逐个消费就绪锚点。

## 用归档指标重建图表

```powershell
.\.venv\Scripts\python.exe scripts/audit_research_v2.py --root results/research_v2 --archived-metrics
.\.venv\Scripts\python.exe -m optcourse.research.publication --root results/research_v2
```

归档包含每次正式训练的配置、损失曲线、验证选模点和测试目标数，以及每个冻结状态的候选损失、梯度对齐、曲率与谱统计。原始训练产生的模型和全量候选张量存放在运行目录，执行训练后可重建。所有 CUDA 阶段共用一个账本。账本沿用课程第一版已消耗的 3507.46 秒，并记录研究版自身的 CUDA 时长。
