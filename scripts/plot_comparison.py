import logging
import os.path as osp
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
# import sci_palettes
import seaborn as sns
import colorcet as cc

sys.path.append("/".join(osp.abspath(__file__).split("/")[:-2]))
from src.utils import read_json


def main():
    logging.basicConfig(level=logging.INFO)

    root = "/mnt/data1/tiantan/results"
    run_dirs = {
        "DualAttCNN (Focal Loss, Proposed)": "2023-11-19_21-00-03",
        "DualAttCNN (Cross Entropy)": "2023-11-19_18-06-19",
        "3D-CNN (Focal Loss)": "2023-11-20_15-36-37",
        "3D-CNN (Cross Entropy)": "2023-11-21_01-14-01",
        "2D-CNN (Focal Loss)": "2023-11-19_16-30-42",
        "2D-CNN (Cross Entropy)": "2023-11-21_01-41-11",
        "SIFT & HOG + SVM": "2023-11-20_23-22-28",
        "SIFT & HOG + RF": "2023-11-21_00-26-24",
    }

    # 3. load the test scores
    test_scores = []
    for k, run_dir in run_dirs.items():
        for i in range(5):
            fn = osp.join(root, run_dir, "fold%d" % i, "test_scores.json")
            test_scores_i = read_json(fn)
            test_scores_i["fold"] = i
            test_scores_i["method"] = k
            test_scores.append(test_scores_i)
    test_scores = pd.DataFrame.from_records(test_scores)

    metric_mapping = {
        "bacc": "Balanced Accuracy",
        "acc": "Accuracy",
        "auc": "Area Under ROC Curver",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
    }
    test_scores.rename(columns=metric_mapping, inplace=True)

    # 4. plot the metrics
    plt.rcParams["font.family"] = "Times New Roman"
    nmethods = len(run_dirs)
    # sci_palettes.register_cmap()
    # palette = sns.color_palette("npg_nrc")[:nmethods]
    palette = cc.glasbey_bw[:nmethods]

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(ncols=6, nrows=2)
    axs = [
        fig.add_subplot(gs[0, :2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]
    # ncol, nrow = 3, 2
    # fig, axs = plt.subplots(
    #     nrows=nrow,
    #     ncols=ncol,
    #     figsize=(3 * ncol, 3 * nrow),  # add space for legend
    #     squeeze=False,
    #     layout="constrained",
    # )
    # axs = axs.flatten()
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

    handles, labels = [], []
    for i, label in enumerate(test_scores["method"].unique()):
        handles.append(mpatches.Patch(color=palette[i], label=label))
        labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=3,
        frameon=False,
        fancybox=False,
    )

    # 5. saving
    fig.savefig(osp.join(root, "plot_comparison.png"), dpi=300)
    fig.savefig(osp.join(root, "plot_comparison.pdf"))


if __name__ == "__main__":
    main()
