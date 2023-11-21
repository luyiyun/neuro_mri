import logging
import os.path as osp
import sys
from typing import Dict

import colorcet as cc
# import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def get_test_scores(
    run_dirs: Dict, var_name: str, root: str = "/mnt/data1/tiantan/results"
) -> pd.DataFrame:
    test_scores = []
    for k, run_dir in run_dirs.items():
        for i in range(5):
            fn = osp.join(root, run_dir, "fold%d" % i, "test_scores.json")
            test_scores_i = read_json(fn)
            test_scores_i["fold"] = i
            test_scores_i[var_name] = k
            test_scores.append(test_scores_i)
    test_scores = pd.DataFrame.from_records(test_scores)
    test_scores.rename(columns=metric_mapping, inplace=True)
    return test_scores


def main():
    logging.basicConfig(level=logging.INFO)

    root = "/mnt/data1/tiantan/results"

    # 4. plot the metrics
    plt.rcParams["font.family"] = "Times New Roman"
    palette = cc.glasbey_category10

    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    # -- w_kl_satt
    run_dirs_wklsatt = {
        0.0: "2023-11-19_17-40-37",
        0.001: "2023-11-19_18-41-56",
        0.01: "2023-11-19_19-17-36",
        0.05: "2023-11-19_19-54-24",
        0.1: "2023-11-19_20-31-56",
        0.2: "2023-11-19_21-00-03",
        0.3: "2023-11-19_23-14-13",
        0.4: "2023-11-19_23-50-25",
        0.5: "2023-11-20_00-21-05",
        0.7: "2023-11-20_00-47-25",
        1.0: "2023-11-20_01-18-08",
    }
    test_scores = get_test_scores(run_dirs_wklsatt, "Weight", root)
    axs = subfigs[0].subplots(nrows=len(metric_mapping), sharey=True)
    for i, metrici in enumerate(metric_mapping.values()):
        ax = axs[i]
        sns.lineplot(
            data=test_scores,
            x="Weight",
            y=metrici,
            err_style="bars",
            errorbar=("se", 1),
            err_kws={"capsize": 5.0},
            ax=ax,
            # palette=palette
        )
        ax.set_xlabel(
            "Weight of Spatial Attention Constraint"
            if i == (len(metric_mapping) - 1)
            else ""
        )
        ax.set_ylabel("")
        ax.set_title(metrici)
        if i < (len(metric_mapping) - 1):
            ax.set_xticklabels([])
        ax.spines[["right", "top"]].set_visible(False)
    # fig.tight_layout()

    # -- without satt, iatt, satt & iatt
    run_dirs_att = {
        "Proposed Method": "2023-11-19_21-00-03",
        "w/o Spatial Attention": "2023-11-19_15-25-53",
        "w/o Instance Attention": "2023-11-19_15-58-19",
        "w/o Spatial and Instance Attention": "2023-11-19_16-30-42",
    }
    test_scores = get_test_scores(run_dirs_att, "method", root)
    axs = subfigs[1].subplots(nrows=3, ncols=2, sharey=False)
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
        ax.set_title(metrici)
        ax.set_xticklabels([])
        ax.spines[["right", "top"]].set_visible(False)

    axs[-1].spines[["top", "right", "bottom", "left"]].set_visible(False)
    axs[-1].get_xaxis().set_ticks([])
    axs[-1].get_yaxis().set_ticks([])
    handles, labels = [], []
    for i, label in enumerate(test_scores["method"].unique()):
        handles.append(mpatches.Patch(color=palette[i], label=label))
        labels.append(label)
    axs[-1].legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        fancybox=False,
    )
    # ax.legend(
    #     loc="best",
    #     ncols=3,
    #     frameon=False,
    #     fancybox=False,
    # )

    # fig = plt.figure(constrained_layout=True)
    # gs = fig.add_gridspec(ncols=6, nrows=2)
    # axs = [
    #     fig.add_subplot(gs[0, :2]),
    #     fig.add_subplot(gs[0, 2:4]),
    #     fig.add_subplot(gs[0, 4:]),
    #     fig.add_subplot(gs[1, 1:3]),
    #     fig.add_subplot(gs[1, 3:5]),
    # ]
    # ncol, nrow = 3, 2
    # fig, axs = plt.subplots(
    #     nrows=nrow,
    #     ncols=ncol,
    #     figsize=(3 * ncol, 3 * nrow),  # add space for legend
    #     squeeze=False,
    #     layout="constrained",
    # )
    # axs = axs.flatten()
    # for i, metrici in enumerate(metric_mapping.values()):
    #     ax = axs[i]
    #     sns.boxplot(
    #         data=test_scores,
    #         x="method",
    #         y=metrici,
    #         hue="method",
    #         ax=ax,
    #         dodge=False,
    #         palette=palette,
    #     )
    #     ax.set_ylabel("")
    #     ax.set_xlabel("")
    #     ax.set_title(metrici)
    #     ax.set_xticklabels([])
    #     ax.spines[["right", "top"]].set_visible(False)

    # handles, labels = [], []
    # for i, label in enumerate(test_scores["method"].unique()):
    #     handles.append(mpatches.Patch(color=palette[i], label=label))
    #     labels.append(label)
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="outside lower center",
    #     ncols=3,
    #     frameon=False,
    #     fancybox=False,
    # )

    # 5. saving
    fig.savefig(osp.join(root, "plot_ablation_weight.png"))


if __name__ == "__main__":
    main()
