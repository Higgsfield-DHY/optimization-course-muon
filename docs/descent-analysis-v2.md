**Muon有限步谱变换的下降几何分析**

令g为当前位置的目标梯度，M为Nesterov动量矩阵。候选谱变换为D_k=R·Q_k(M)/||Q_k(M)||_F；R在同一个参数矩阵的候选之间固定。实际网络更新还包含共享辅助参数和衰减项U。下述局部分析使用完整候选步长S_k=eta D_k+U，参数更新为w←w−S_k。

**一、完整步长的可测下降分解。**

设f的Hessian在所考察线段上为rho-Lipschitz。定义a_k=<grad f(w),S_k>，b_k=<S_k,H(w)S_k>。则

    f(w)−f(w−S_k)=a_k−b_k/2+r_k,
    |r_k|≤rho ||S_k||^3/6.

证明沿t↦f(w−tS_k)应用带积分余项的二阶Taylor公式。对二次目标余项为零。完整S_k使Hessian的跨层项、辅助参数项及其与矩阵更新的交叉项进入同一个预测量。神经实验在批次B计算a_k、b_k，在独立批次C测量实际损失下降。

对正定二次目标及一个未乘标量的方向D，设a=<g,D>>0，b=<D,HD>>0。沿该方向的最优正步长为eta*=a/b，最大下降量为J(D)=a²/(2b)。D乘任意正常数时J保持不变。由此可在调节更新幅度和优化标量学习率后研究方向几何。

**二、相同动量下的曲率反转。**

设两个单位Frobenius方向u、v线性无关，且a_u=<g,u>>0、a_v=<g,v>>0。给定0<epsilon<1，定义

    H_u[X]=X−(1−epsilon)<X,u>u,
    H_v[X]=X−(1−epsilon)<X,v>v.

这两个算子的最大特征值均为1，最小特征值均为epsilon。目标

    f_u(X)=<g,X>+<X,H_u[X]>/2,
    f_v(X)=<g,X>+<X,H_v[X]>/2

在共同起点X=0具有相同梯度g；给定的动量M和所有候选谱误差也相同。令c=<u,v>，则

    J_u(u)/J_u(v)
      =(a_u²/a_v²)[epsilon+(1−epsilon)(1−c²)]/epsilon,
    J_v(v)/J_v(u)
      =(a_v²/a_u²)[epsilon+(1−epsilon)(1−c²)]/epsilon.

因为1−c²>0，两比值在epsilon→0时均趋于无穷。于是足够小的同一epsilon使u在f_u上更好、v在f_v上更好。更新RMS相等和每个方向单独优化标量步长均保留该排序反转。

证明：H_u在u方向上的特征值为epsilon，在其正交补上的特征值为1，故<u,H_u u>=epsilon，<v,H_u v>=epsilon+(1−epsilon)(1−c²)。将其代入J；交换u、v得到第二式。

该构造将曲率作为单独干预变量：谱相同的更新候选在两个目标上需要相反选择。实验用同一g、M和NS候选实现这组目标置换，逐项验证两个比值。

**三、二维谱方向的精确分界。**

令g=(1,rho)，H=diag(1,kappa)，候选方向的坐标比例为r>0。幅度归一化后的最优下降量为

    J(r)=(1+rho r)^2/[2(1+kappa r²)].

给定r>s>0，其差的符号由

    (r−s)[2rho+(rho²−kappa)(r+s)−2rho kappa r s]

决定。分界为

    kappa*=[2rho+rho²(r+s)]/[r+s+2rho r s].

在kappa<kappa*时J(r)>J(s)；在kappa>kappa*时顺序反转。J的连续最优比例为rho/kappa。

证明将J(r)−J(s)通分并因式分解。共同RMS和共同Hessian缩放不改变分界。对经验NS多项式，各r_k由实际变换后的奇异值比例计算；经典三次NS与精确极因子提供收敛参考族。

**四、有限候选的独立探测选择。**

在训练历史和梯度批次A给定后，候选完整步长S_1,...,S_K固定。取m个独立训练探测样本，记单样本下降Z_k=ell(w;z)−ell(w−S_k;z)，且Z_k∈[−B,B]。以样本平均下降最大的候选为k_hat。以至少1−delta的概率，

    E[Z_k_hat] ≥ max_k E[Z_k] − 2B sqrt(2 log(2K/delta)/m).

证明：对每个候选使用Hoeffding界并对K个候选取并集，得到所有经验均值与期望偏差同时小于B sqrt(2log(2K/delta)/m)。把经验最优候选与期望最优候选的两个偏差相加即得。

这里的候选是在同一当前位置比较的完整参数更新，K=3。模型训练以固定周期重新取训练探测批次，stage-matched replay保留各训练阶段内的深度频率和探测计算，用于检验当次目标反馈的贡献。
