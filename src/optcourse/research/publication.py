"""Generate the research paper's figures and numeric tables from verified records."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from .analysis import assemble, read
from .report import COLORS, LABELS, save, style, synthetic_figures
from .training import atomic_json


def train_curves(evidence, output):
    curves = evidence["training"]["curves"]
    panels = [
        ("main/wikitext103/small", ("adamw","momentum","ns1","ns3","ns5","adaptive","replay","spectral"), "(a) WikiText103, 26.5M"),
        ("scale/wikitext103/medium", ("adamw","ns5","adaptive"), "(b) WikiText103, 51.1M"),
        ("transfer/tinystories/small", ("adamw","ns5","adaptive"), "(c) TinyStories, 26.5M"),
        ("long/wikitext103/small", ("ns3","ns5"), "(d) Extended WikiText103 training"),
    ]
    fig,axes=plt.subplots(2,2,figsize=(7.3,4.8),layout="constrained")
    for ax,(key,methods,title) in zip(axes.flat,panels):
        for method in methods:
            source = "criterion/wikitext103/small" if method=="spectral" else key
            members = curves[source][method]
            steps = [row["tokens"]/1e6 for row in next(iter(members.values()))]
            losses = np.array([[row["validation_loss"] for row in run] for run in members.values()])
            avg = losses.mean(axis=0)
            spread = losses.std(axis=0,ddof=1)
            ax.plot(steps,avg,label=LABELS[method],color=COLORS[method],marker=".",markersize=2.4)
            ax.fill_between(steps,avg-spread,avg+spread,color=COLORS[method],alpha=.11,linewidth=0)
        ax.set(xlabel="Training tokens (millions)",ylabel="Validation loss (nats/token)",title=title)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=4,fontsize=8.5)
    save(fig,output,"v2_training")


def probe_figures(probe, output):
    summaries = probe["ranking"]
    names = {"polar_error":"Polar error","alignment":"Alignment","quadratic":"Quadratic","batch_b_step":"Probe loss"}
    colors = {"polar_error":"#58857d","alignment":"#3a91aa","quadratic":"#cb6a3b","batch_b_step":"#9c4874"}
    fig,axes = plt.subplots(1,2,figsize=(7.2,2.65),layout="constrained")
    for predictor,label in names.items():
        rows=sorted((row for row in summaries if row["predictor"]==predictor),key=lambda row:row["multiplier"])
        for ax,metric,factor in zip(axes,("concordance","regret"),(100,1000)):
            xs=[row["multiplier"] for row in rows]
            ys=[factor*row[metric]["mean"] for row in rows]
            spread=[factor*(row[metric]["sd"] or 0) for row in rows]
            errors=[np.minimum(spread,ys),spread] if metric=="regret" else spread
            ax.errorbar(xs,ys,yerr=errors,marker="o",markersize=3.7,capsize=2.4,color=colors[predictor],label=label)
    axes[0].axhline(50,ls="--",color="#777777",lw=.8)
    axes[0].set(ylabel="Pairwise agreement (%)",title="(a) Independent-batch ranking")
    axes[1].set(ylabel="Selection regret ($10^{-3}$ nats/token)",title="(b) Loss from the selected depth")
    axes[1].set_yscale("symlog",linthresh=.01)
    axes[1].set_yticks([0,.01,.1,1,10,100])
    axes[1].yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    for ax in axes:
        ax.set(xlabel="Step multiplier",xticks=[.5,1,2])
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=4,fontsize=8.5)
    save(fig,output,"v2_probe_ranking")

    rows=probe["calibration_rows"]
    actual=np.array([r["same_batch_gain"] for r in rows])*1e3
    limit=max(abs(actual).max(),1e-4)
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.55),layout="constrained",sharex=True,sharey=True)
    for ax,name,title in ((axes[0],"linear","(a) Alignment"),(axes[1],"quadratic","(b) Alignment and curvature")):
        predicted=np.array([r["linear"]+(r["curvature"] if name=="quadratic" else 0)
                            for r in rows])*1e3
        ax.scatter(actual,predicted,color=colors["quadratic" if name=="quadratic" else "alignment"],
                   s=11,alpha=.56,edgecolors="none")
        ax.plot([-limit,limit],[-limit,limit],ls="--",color="#444444",lw=.75)
        ax.set(xlabel="Observed gain gap on B ($10^{-3}$)",title=title)
    axes[0].set_ylabel("Predicted gain gap ($10^{-3}$)")
    save(fig,output,"v2_curvature_calibration")


def profile_figure(profile,output):
    methods=("adamw","momentum","ns1","ns3","ns5","canonical5")
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.45),layout="constrained",sharey=True)
    for ax,scale in zip(axes,("small","medium")):
        measurements=profile[scale]
        ticks=np.arange(len(methods))
        model=np.array([measurements[method]["model_seconds"] for method in methods])*1000
        clip=np.array([measurements[method]["clip_seconds"] for method in methods])*1000
        optim=np.array([measurements[method]["optimizer_seconds"] for method in methods])*1000
        ax.bar(ticks,model,color="#758ca3",label="Forward/backward")
        ax.bar(ticks,clip,bottom=model,color="#9ebd9b",label="Clipping")
        ax.bar(ticks,optim,bottom=model+clip,color="#cb6a3b",label="Optimizer")
        ax.set_xticks(ticks,[LABELS[k].replace("Canonical ","Std. ") for k in methods],rotation=30,ha="right")
        ax.set_title("26.5M parameters" if scale=="small" else "51.1M parameters")
    axes[0].set_ylabel("Median elapsed time (ms)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=3,fontsize=9)
    save(fig,output,"v2_profile")


def spectra_figure(rows,output):
    layers=("QKV","MLP expansion","Attention output","MLP contraction")
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.7),layout="constrained",sharey=True)
    for ax,scale in zip(axes,("small","medium")):
        for index,variant in enumerate(("ns1","ns3","ns5")):
            means=[]
            spreads=[]
            for layer in layers:
                seeds=defaultdict(list)
                for row in rows:
                    if row["corpus"]=="wikitext103" and row["scale"]==scale and row["layer"]==layer and row["variant"]==variant:
                        seeds[row["seed"]].append(row["stable_rank_fraction"])
                values=[mean(group) for group in seeds.values()]
                means.append(mean(values))
                spreads.append(stdev(values))
            centers=np.arange(len(layers))+(index-1)*.22
            ax.bar(centers,means,yerr=spreads,capsize=2,width=.2,color=COLORS[variant],label=LABELS[variant])
        ax.set_xticks(np.arange(len(layers)),["QKV","MLP up","Attention out","MLP down"],rotation=23,ha="right")
        ax.set_title("26.5M parameters" if scale=="small" else "51.1M parameters")
    axes[0].set_ylabel("Stable rank / matrix rank limit")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=3,fontsize=9)
    save(fig,output,"v2_update_spectra")


def pm(row,digits=4):
    return f'${row["mean"]:.{digits}f}\\pm{row["sd"]:.{digits}f}$'


def write_tables(evidence,paper):
    training=evidence["training"]
    groups=training["groups"]
    main=groups["main/wikitext103/small"]|groups["ablation/wikitext103/small"]|groups["criterion/wikitext103/small"]
    order=("adamw","momentum","ns1","ns3","ns5","adaptive","replay","spectral","adam_rms","canonical5")
    roll=r" \\"
    def table(rows,style):
        return "\n".join([r"\begin{tabular}{"+style+"}",r"\toprule"]+rows+[r"\bottomrule",r"\end{tabular}"])+"\n"

    rows=[r"方法 & 矩阵峰值学习率 & 终点验证损失 & 测试损失"+roll+r" \midrule"]
    for method in order:
        group=main[method]
        rows.append(f"{LABELS[method]} & {group['seeds'][0]['lr']:.4f} & {pm(group['terminal_validation'])} & {pm(group['test_loss'])}"+roll)
    (paper/"v2_table_main.tex").write_text(table(rows,"lrrr"),encoding="utf-8")

    rows=[r"实验设置 & 方法 & 终点验证损失 & 测试损失"+roll+r" \midrule"]
    for group_key,label in (("scale/wikitext103/medium","WikiText103，51.1M"),
                            ("transfer/tinystories/small","TinyStories，26.5M"),
                            ("long/wikitext103/small","WikiText103，20.48M token")):
        methods=groups[group_key]
        for index,(method,values) in enumerate(methods.items()):
            rows.append(f"{label if index==0 else ''} & {LABELS[method]} & {pm(values['terminal_validation'])} & {pm(values['test_loss'])}"+roll)
        if not group_key.startswith("long/"):
            rows.append(r"\addlinespace")
    (paper/"v2_table_transfer.tex").write_text(table(rows,"llrr"),encoding="utf-8")

    rows=[r"种子 & Probe-NS深度 & 重放深度 & 改动步数"+roll+r" \midrule"]
    for row in training["replay"]:
        a=",".join(map(str,row["adaptive"]))
        r=",".join(map(str,row["replay"]))
        rows.append(f"{row['seed']} & {a} & {r} & {row['changed_updates']}"+roll)
    (paper/"v2_table_replay.tex").write_text(table(rows,"cllr"),encoding="utf-8")

    fixed=evidence["synthetic"]
    rows=[r"对数谱跨度 & 非并列状态 & 误差最小候选收益次优 & 收益并列"+roll+r" \midrule"]
    for item in fixed["error_by_spectrum"]:
        rows.append(f"{item['exponent']:g} & {item['states']} & {item['error_winner_suboptimal']} & {item['tied_states']}"+roll)
    (paper/"v2_table_matrix.tex").write_text(table(rows,"rrrr"),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",default="runs/research_v2")
    parser.add_argument("--paper",default="paper")
    args=parser.parse_args()
    root,paper=Path(args.root),Path(args.paper)
    evidence=assemble(root)
    expected=read(root/"frozen_protocol.json")["core_formal_runs"]
    if evidence["training"]["training_status"]!={"COMPLETED":expected} or evidence["training"]["test_completed"]!=expected:
        raise ValueError("Publication tables require the registered complete training and test set")
    if evidence["probes"]["completed_anchors"]!=evidence["probes"]["expected_anchors"] or evidence["probes"]["expected_anchors"]!=36:
        raise ValueError("Publication figures require 36 verified frozen anchors")
    if set(evidence["profile"])!={"small","medium"}:
        raise ValueError("Missing synchronized performance measurements")
    output=paper/"figures_v2"
    style()
    synthetic_figures(root,output)
    train_curves(evidence,output)
    probe_figures(evidence["probes"],output)
    profile_figure(evidence["profile"],output)
    spectra_figure(evidence["update_spectra"],output)
    write_tables(evidence,paper)
    atomic_json(paper/"v2_numeric_evidence.json",evidence)


if __name__=="__main__":
    main()
