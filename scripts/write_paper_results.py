"""Populate paper from completed, audited study artifacts; no model execution."""
from pathlib import Path
import csv,json,math,re,shutil,statistics,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from optcourse.comparison import compare_study

study=ROOT/'runs/course_v1'
state=json.loads((study/'study_state.json').read_text(encoding='utf-8'))
budget=json.loads((study/'gpu_budget.json').read_text(encoding='utf-8'))
assert state['status']=='EXPERIMENTS_COMPLETE'
assert budget['active'] is None and budget['used_seconds']<=28800
report=compare_study(study/'study_state.json',study/'reports')
assert report['expected_final_runs']==report['completed_final_runs']==54
assert all(row['n']==3 for row in report['groups'])
assert sum(row['test_n'] for row in report['groups'])==9
paper=ROOT/'paper';figures=paper/'figures';figures.mkdir(exist_ok=True)
for name in ('matrix_objective','matrix_gradient','language_bytes','language_time'):
    for suffix in ('.pdf','.png'):
        shutil.copy2(study/'reports'/f'{name}{suffix}',figures/f'{name}{suffix}')
labels=dict(adamw='AdamW',svd='Muon-SVD',ns1='Muon-NS1',ns3='Muon-NS3',ns5='Muon-NS5')
language={row['variant']:row for row in report['groups'] if row['task']=='language'}
matrix={(row['condition_number'],row['variant']):row for row in report['groups'] if row['task']=='matrix'}
def number(value):
    text=f'{value:.3g}'
    if 'e' in text:
        base,power=text.split('e');return base+r'\times10^{'+str(int(power))+'}'
    return text
def pm(mean,std,digits=None):
    if digits is None:return number(mean)+r'\pm '+number(std)
    return f'{mean:.{digits}f}'+r'\pm '+f'{std:.{digits}f}'
matrix_rows=[]
for condition in (1,10,100):
    for variant in ('adamw','svd','ns1','ns3','ns5'):
        row=matrix[condition,variant]
        matrix_rows.append(f"{condition} & {labels[variant]} & ${pm(row['score_mean'],row['score_std'])}$ & {row['elapsed_mean']:.3f} "+r'\\')
language_rows=[]
for variant in ('adamw','ns5','ns3'):
    row=language[variant]
    language_rows.append(f"{labels[variant]} & ${pm(row['score_mean'],row['score_std'],4)}$ & ${pm(row['test_loss_mean'],row['test_loss_std'],4)}$ & {row['elapsed_mean']:.1f} & {row['peak_memory_MiB_mean']:.1f} "+r'\\')
diagnostics=json.loads((study/'matrix_diagnostics.json').read_text())['rows']
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.rcParams.update({'pdf.fonttype':42,'font.size':13})
fig,axes=plt.subplots(1,3,figsize=(10,2.7),layout='constrained')
for axis,condition in zip(axes,(1,10,100)):
    values=[[row['relative_error_F'] for row in diagnostics if row['condition_number']==condition and row['variant']==variant] for variant in ('ns1','ns3','ns5')]
    axis.errorbar([1,3,5],[statistics.mean(x) for x in values],yerr=[statistics.stdev(x) for x in values],marker='o',capsize=3)
    axis.set(xlabel='NS steps',ylabel='Relative polar error',title=f'Condition number {condition}',xticks=[1,3,5])
    axis.grid(alpha=.2)
fig.savefig(figures/'ns_diagnostics.pdf',bbox_inches='tight');fig.savefig(figures/'ns_diagnostics.png',dpi=180,bbox_inches='tight');plt.close(fig)
final_jobs=[row for row in state['jobs'].values() if row['task']=='language' and row['phase']=='final']
paired={v:{row['seed']:row['test_evaluation']['loss'] for row in final_jobs if row['variant']==v} for v in ('adamw','ns5','ns3')}
differences=[paired['ns5'][seed]-paired['adamw'][seed] for seed in (0,1,2)]
delta=statistics.mean(differences)
baseline=[];start_memory=[]
for row in final_jobs:
    meta=json.loads((Path(row['run_path'])/'metadata.json').read_text(encoding='utf-8'))
    baseline.append(int(meta['gpu_baseline_utilization_percent_and_free_MiB'].split(',')[0]))
    start_memory.append(meta['gpu_process_start_allocated_MiB'])
recoveries=[]
for row in state['jobs'].values():
    if row.get('run_path'):
        meta=json.loads((Path(row['run_path'])/'metadata.json').read_text(encoding='utf-8'))
        recoveries.extend(meta.get('attempts',[]))
text=r'''\subsection{运行完整性与固定参数}
实验在Windows11、Intel Core i9-12900H和RTX3070Ti Laptop GPU（8192MiB）上运行，CPU线程数为1。
环境为Python3.12.4、PyTorch2.7.1+cu128、CUDA12.8，显卡驱动591.74。
正式清单中45次矩阵运行、9次语言模型运行全部完成，每组均为三个独立种子。校准和学习率筛选未计入正式样本。
GPU共享账本累计@GPU_MIN@分钟，包含校准、筛选、训练、测试评估，以及一次测量维护重启的保守记账；预算上限为480分钟。
重启发生在调参阶段，检查点与中断原因保留，正式配置和目标步数未因此改变。
依据预先规定的1.3倍耗时预测规则，正式长度固定为@STEPS@步，即每次处理@BYTES@字节。
语言模型选定的学习率为AdamW @LR_A@、Muon @LR_M@；辅助AdamW为$3\times10^{-4}$。
三种正式方法均使用微批@BATCH@、累积@ACCUM@。

\subsection{矩阵优化结果}
表\ref{tab:matrix}给出1000步后的目标值，已知最优值为零。$\pm$表示样本标准差，时间列为三次任务累计墙钟时间的均值。
在本次固定步长和搜索网格下，三个条件数中AdamW的终点目标值均更低。
例如条件数100时，AdamW为@MATRIX_A@，NS5为@MATRIX_M@。
这并不与驻点定理矛盾：本实验采用固定经验步长，而定理按总时域设置参数，且没有给出算法之间的无条件优越性。
\begin{table}[htbp]\centering\small
\caption{矩阵正式实验终点目标值，$n=3$}\label{tab:matrix}
\begin{tabular}{rlrr}\toprule 条件数&方法&目标值（均值$\pm$标准差）&时间/s\\\midrule
@MATRIX_ROWS@
\bottomrule\end{tabular}\end{table}

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/matrix_objective.pdf}
\caption{矩阵目标值轨迹。每条线对应一个种子；纵轴为对数坐标。}\label{fig:matrix}
\end{figure}
\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/matrix_gradient.pdf}
\caption{真实梯度Frobenius范数。该指标从解析梯度计算，未用优化器内部动量替代。}
\end{figure}
图\ref{fig:matrix}显示了下降与终点邻域的行为。极因子类方向改变梯度幅值信息，固定步长在接近解时仍可能产生非消失更新；这是解释终点差异的一种机制，本文没有额外实验把它识别为唯一原因。
两个方法的矩阵学习率都在搜索网格上界处选中，因而本次筛选没有定位各自全局最优学习率。

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/ns_diagnostics.pdf}
\caption{同一初始梯度输入下，NS输出相对紧致SVD极因子的Frobenius误差。误差条为三个输入种子的样本标准差。}\label{fig:ns}
\end{figure}
图\ref{fig:ns}将近似误差与完整优化轨迹分开。误差与最终损失不是同一指标；有限步经验多项式的正向对齐也不要求极因子误差逐步趋零。

\subsection{小型语言模型结果}
表\ref{tab:language}中验证损失来自相同总训练字节预算的终点，测试损失来自各次运行由验证集选出的最佳检查点。
二者可能对应不同步数，不能把测试结果解释为终点模型的测试损失。所有损失单位均为nats/byte。
\begin{table}[htbp]\centering\small
\caption{语言模型正式实验，$n=3$；时间为任务累计墙钟均值，显存为峰值已分配量均值}\label{tab:language}
\resizebox{\linewidth}{!}{\begin{tabular}{lrrrr}\toprule
方法&终点验证损失&最佳模型测试损失&时间/s&显存/MiB\\\midrule
@LANGUAGE_ROWS@
\bottomrule\end{tabular}}\end{table}

\begin{figure}[htbp]\centering
\includegraphics[width=.78\linewidth]{figures/language_bytes.pdf}
\caption{完整验证损失随训练字节数变化。实线为三种子的均值，阴影为样本标准差。}\label{fig:bytes}
\end{figure}
\begin{figure}[htbp]\centering
\includegraphics[width=.78\linewidth]{figures/language_time.pdf}
\caption{可恢复优化轨迹时钟下的验证损失，每条线为一个种子。此时钟不包含模型构建、末次保存或被回滚的工作，不能与表格累计任务时间混用。}\label{fig:time}
\end{figure}
NS5减去AdamW的三个配对测试损失差为@PAIRS@，平均差为@DELTA@。
这说明在所列语料、模型和筛选预算下，NS5的平均测试损失@DIRECTION@；三个种子不足以支持跨模型规模或一般统计显著性的结论。
NS3复用了NS5的学习率，因而该比较考察固定设置下改变NS步数的效果，不代表为NS3单独调参后的最好表现。

\subsection{计算开销与适用范围}
正式GPU任务启动前整卡利用率范围为@UTIL_MIN@\%至@UTIL_MAX@\%，存在桌面图形负载。
任务间做垃圾回收与未使用CUDA缓存清理，记录的进程起始已分配量范围为@START_MIN@至@START_MAX@MiB。
表格和时间曲线是本机当次环境的观测，不用于宣称隔离环境的内核加速比。方法顺序按种子轮转，仍不能消除所有时变负载。
Tiny Shakespeare规模有限，且连续划分可能存在文本分布差异；本文采用字节编码与短上下文，不评价通用生成能力。
有限学习率网格、仅三个正式种子、辅助AdamW的固定学习率和精度设置都限制了结论范围。
完整理论假设未在该Transformer上验证；实际损失表现与单矩阵收敛证明应分别理解。
'''
values={'GPU_MIN':f"{budget['used_seconds']/60:.2f}",'STEPS':str(state['profile']['steps']),
 'BYTES':f"{state['profile']['steps']*4096:,}",'LR_A':str(state['language_selection']['adamw']),
 'LR_M':str(state['language_selection']['muon']),'BATCH':str(state['profile']['micro_batch']),
 'ACCUM':str(32//state['profile']['micro_batch']),'MATRIX_A':r'$'+pm(matrix[100,'adamw']['score_mean'],matrix[100,'adamw']['score_std'])+'$',
 'MATRIX_M':r'$'+pm(matrix[100,'ns5']['score_mean'],matrix[100,'ns5']['score_std'])+'$',
 'MATRIX_ROWS':'\n'.join(matrix_rows),'LANGUAGE_ROWS':'\n'.join(language_rows),
 'PAIRS':', '.join(f'{x:+.4f}' for x in differences),'DELTA':f'{delta:+.4f}',
 'DIRECTION':'较低' if delta<0 else '较高','UTIL_MIN':str(min(baseline)),'UTIL_MAX':str(max(baseline)),
 'START_MIN':f'{min(start_memory):.2f}','START_MAX':f'{max(start_memory):.2f}'}
for key,value in values.items():text=text.replace('@'+key+'@',value)
assert not re.search(r'@[A-Z_]+@',text)
(paper/'results.tex').write_text(text,encoding='utf-8')
conclusion=f'''本文从谱范数约束下的更新方向出发，完整重构了五步经验NS、动量与权重衰减下的有限步驻点界。
证明通过方向对齐而非单调正交化误差控制下降，保留初始化、稳定项与参数安排的作用。
正式实验完成45次矩阵优化和9次小型语言模型训练。
固定步长的矩阵问题中，AdamW在所列网格下终点目标值更低；语言模型中，NS5的平均测试损失为{language['ns5']['test_loss_mean']:.4f}，AdamW为{language['adamw']['test_loss_mean']:.4f}。
这种跨任务差异要求结合问题结构和参数安排理解优化器，不能由一个任务推断普遍优劣。
进一步扩大数据和模型规模、增加种子及调参预算，才有条件研究结论的稳定性。\n'''
(paper/'conclusion.tex').write_text(conclusion,encoding='utf-8')
abstract=f'''\\begin{{abstract}}
本文研究Muon的矩阵更新机制，区分精确极因子、经验Newton--Schulz迭代与实际神经网络训练设置。
从谱范数约束的线性化目标解释更新方向，并重构五步经验NS、动量和解耦权重衰减下的有限步收敛界，保留初始化与归一化误差。
实验包含45次可控矩阵优化及9次约482万参数Transformer训练，校准、调参与正式种子分离，完整验证选模、测试集最终评估。
矩阵问题中AdamW在所列固定步长网格下终点目标更低；语言模型中AdamW与NS5的测试损失分别为{language['adamw']['test_loss_mean']:.4f}和{language['ns5']['test_loss_mean']:.4f}nats/byte。
GPU账本累计{budget['used_seconds']/60:.2f}分钟。结果限定于所列任务、预算与实现变体，理论保证不直接覆盖完整Transformer流程。
\\end{{abstract}}'''
main=(paper/'main.tex').read_text(encoding='utf-8')
main=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:abstract,main,flags=re.S)
(paper/'main.tex').write_text(main,encoding='utf-8')
evidence=dict(final_runs=54,test_runs=9,gpu_seconds=budget['used_seconds'],profile=state['profile'],
 dataset_fingerprint=state['dataset_fingerprint'],matrix_selection=state['matrix_selection'],
 hardware=dict(cpu='Intel Core i9-12900H',gpu='RTX 3070 Ti Laptop GPU',memory_MiB=8192,driver='591.74',torch='2.7.1+cu128',cuda='12.8',cpu_threads=1),
 language_selection=state['language_selection'],language_groups=language,paired_test_differences=differences,
 matrix_groups=report['groups'][:15],maintenance_recovery_records=recoveries,
 gpu_baseline_utilization_range=[min(baseline),max(baseline)],gpu_start_allocated_MiB_range=[min(start_memory),max(start_memory)])
(paper/'numeric-evidence.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'gpu_minutes':budget['used_seconds']/60,'language':language,'paired_test_differences':differences},indent=2))
