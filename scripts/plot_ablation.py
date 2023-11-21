import logging
import os
import os.path as osp
import re
import sys
from datetime import datetime
from typing import List, Tuple

import colorcet as cc
import matplotlib.patches as mpatches
# import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
# import sci_palettes
import seaborn as sns

sys.path.append("/".join(osp.abspath(__file__).split("/")[:-2]))
from src.utils import read_json

metric_mapping = {
    "bacc": "Balanced Accuracy",
    "acc": "Accuracy",
    "auc": "Area Under ROC curver",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
}


def fn2dt(fn: str) -> datetime:
    if re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", fn):
        return datetime.strptime(fn, "%Y-%m-%d_%H-%M-%S")
    return None


def get_test_scores(
    run_items: List[Tuple],
    root: str = "/mnt/data1/tiantan/results",
) -> pd.DataFrame:
    test_scores = []
    for run_item in run_items:
        for i in range(5):
            fn = osp.join(
                root, run_item["dir"], "fold%d" % i, "test_scores.json"
            )
            if not osp.exists(fn):
                continue
            test_scores_i = read_json(fn)
            test_scores_i["fold"] = i
            for k, v in run_item.items():
                if k == "dir":
                    continue
                test_scores_i[k] = v
            test_scores.append(test_scores_i)
    test_scores = pd.DataFrame.from_records(test_scores)
    test_scores.rename(columns=metric_mapping, inplace=True)
    return test_scores


def main():
    logging.basicConfig(level=logging.INFO)

    root = "/mnt/data1/tiantan/results"
    plt.rcParams["font.family"] = "Times New Roman"
    palette = cc.glasbey_category10

    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1, 1.5])
    fig_wkl = subfigs[0]
    subsubfigs = subfigs[1].subfigures(2, 1, hspace=0.07, height_ratios=[1, 2])
    fig_att, fig_focal = subsubfigs[0], subsubfigs[1]

    # -- 1. w_kl_satt
    # -- (1) filter runs
    paths = []
    for fni in os.listdir(root):
        datei = fn2dt(fni)
        if datei is None:
            continue
        if (
            datei >= fn2dt("2023-11-19_18-41-56")
            and datei <= fn2dt("2023-11-20_01-18-08")
        ) or (
            datei >= fn2dt("2023-11-21_09-15-02")
            and datei <= fn2dt("2023-11-21_14-00-38")
        ):
            paths.append(fni)
    # -- (2) load args
    run_items_wklsatt = []
    for fni in paths:
        args = read_json(osp.join(root, fni, "fold0", "args.json"))
        run_items_wklsatt.append(
            {
                "dir": fni,
                "Weight": args["w_kl_satt"],
                "loss_func": args["loss_func"],
            }
        )
    run_items_wklsatt.append(
        {"dir": "2023-11-19_17-40-37", "Weight": 0.0, "loss_func": "focal"}
    )
    run_items_wklsatt.append(
        {"dir": "2023-11-21_22-18-35", "Weight": 0.0, "loss_func": "ce"}
    )
    # -- (3) load test scores
    test_scores = get_test_scores(run_items_wklsatt, root)
    test_scores.replace(
        {"loss_func": {"focal": "Focal Loss", "ce": "CrossEnropy Loss"}},
        inplace=True,
    )
    # -- (4) plotting
    axs = fig_wkl.subplots(nrows=len(metric_mapping), sharey=True)
    for i, metrici in enumerate(metric_mapping.values()):
        ax = axs[i]
        sns.boxplot(
            data=test_scores,
            x="Weight",
            y=metrici,
            hue="loss_func",
            ax=ax,
        )
        ax.set_xlabel(
            "Weight of Spatial Attention Constraint"
            if i == (len(metric_mapping) - 1)
            else ""
        )
        ax.set_ylabel("")
        ax.set_title(metrici, fontsize="medium")
        if i < (len(metric_mapping) - 1):
            ax.set_xticklabels([])
        ax.spines[["right", "top"]].set_visible(False)
    handles, labels = axs[0].get_legend_handles_labels()
    fig_wkl.legend(
        handles,
        labels,
        loc="outside upper center",
        ncols=2,
        frameon=False,
        fancybox=False,
    )
    for ax in axs:
        ax.get_legend().remove()

    # -- 2. without satt, iatt, satt & iatt
    run_items_att = [
        {"method": "Proposed Method", "dir": "2023-11-19_21-00-03"},
        {"method": "w/o Spatial Attention", "dir": "2023-11-19_15-25-53"},
        {"method": "w/o Instance Attention", "dir": "2023-11-19_15-58-19"},
        {"method": "w/o All Attentions", "dir": "2023-11-19_16-30-42"},
    ]
    test_scores = get_test_scores(run_items_att, root)
    # gs = fig_att.add_gridspec(ncols=6, nrows=2)
    # axs = [
    #     fig_att.add_subplot(gs[0, :2]),
    #     fig_att.add_subplot(gs[0, 2:4]),
    #     fig_att.add_subplot(gs[0, 4:]),
    #     fig_att.add_subplot(gs[1, 1:3]),
    #     fig_att.add_subplot(gs[1, 3:5]),
    # ]
    axs = fig_att.subplots(nrows=1, ncols=5, sharey=False)
    axs = axs.flatten()
    for i, metrici in enumerate(metric_mapping.values()):
        ax = axs[i]
        sns.boxplot(
            data=test_scores,
            x="method",
            y=metrici,
            hue="method",
            ax=ax,
            dodge=False,
            palette=palette,
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(metrici, fontsize="medium")
        ax.set_xticklabels([])
        ax.xaxis.set_ticks([])
        ax.spines[["right", "top"]].set_visible(False)

    # axs[-1].spines[["top", "right", "bottom", "left"]].set_visible(False)
    # axs[-1].get_xaxis().set_ticks([])
    # axs[-1].get_yaxis().set_ticks([])
    handles, labels = [], []
    for i, label in enumerate(test_scores["method"].unique()):
        handles.append(mpatches.Patch(color=palette[i], label=label))
        labels.append(label)
    fig_att.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=2,
        frameon=False,
        fancybox=False,
    )

    # -- 3. focal loss parameters
    paths = []
    for fni in os.listdir(root):
        datei = fn2dt(fni)
        if datei is None:
            continue
        if (
            datei >= fn2dt("2023-11-21_15-05-11")
            and datei <= fn2dt("2023-11-21_20-40-49")
        ):
            paths.append(fni)
    run_items_focal = []
    for fni in paths:
        args = read_json(osp.join(root, fni, "fold0", "args.json"))
        run_items_focal.append(
            {
                "dir": fni,
                "Gamma": args["focal_gamma"],
                "Alpha": "alpha=%.1f" % args["focal_alpha"],
            }
        )
    test_scores = get_test_scores(run_items_focal, root)

    gs = fig_focal.add_gridspec(ncols=6, nrows=2)
    axs = [
        fig_focal.add_subplot(gs[0, :2]),
        fig_focal.add_subplot(gs[0, 2:4]),
        fig_focal.add_subplot(gs[0, 4:]),
        fig_focal.add_subplot(gs[1, 1:3]),
        fig_focal.add_subplot(gs[1, 3:5]),
    ]

    for i, metrici in enumerate(metric_mapping.values()):
        ax = axs[i]
        sns.boxplot(
            data=test_scores,
            x="Gamma",
            y=metrici,
            hue="Alpha",
            ax=ax,
            hue_order=["alpha=0.5", "alpha=0.8", "alpha=1.0"],
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(metrici, fontsize="medium")
        # if i < (len(metric_mapping) - 1):
        #     ax.set_xticklabels([])
        ax.spines[["right", "top"]].set_visible(False)
    handles, labels = axs[0].get_legend_handles_labels()
    fig_focal.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=3,
        frameon=False,
        fancybox=False,
        # title="alpha",
    )
    for ax in axs:
        ax.get_legend().remove()

    # 5. saving
    fig.savefig(osp.join(root, "plot_ablation_weight.png"))


if __name__ == "__main__":
    main()
