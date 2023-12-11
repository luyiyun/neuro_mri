import logging
import os.path as osp

import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import precision_recall_curve, roc_curve

# sys.path.append("/".join(osp.abspath(__file__).split("/")[:-2]))


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
    test_preds = []
    for k, run_dir in run_dirs.items():
        for i in range(5):
            fn = osp.join(root, run_dir, "fold%d" % i, "test_pred.csv")
            test_pred_i = pd.read_csv(fn, index_col=0)
            test_pred_i["fold"] = i
            test_pred_i["method"] = k
            test_preds.append(test_pred_i)
    test_preds = pd.concat(test_preds, axis=0)
    test_preds[["MS", "CSVD"]] = softmax(test_preds[["MS", "CSVD"]].values, 1)
    test_preds["label"] = test_preds["label"].map(
        lambda x: ["MS", "CSVD"].index(x)
    )

    # 4. plot the metrics
    plt.rcParams["font.family"] = "Times New Roman"
    nmethods = len(run_dirs)
    palette = cc.glasbey_bw[:nmethods]

    fig, axs = plt.subplots(
        ncols=2,
        figsize=(8.5, 5),
        constrained_layout=True,
        # sharex=True,
        # sharey=True,
    )
    axs[0].text(
        -0.1,
        1.05,
        "A",
        fontweight="bold",
        fontsize=12,
        va="center",
        ha="center",
    )
    axs[1].text(
        -0.1,
        1.01,
        "B",
        fontweight="bold",
        fontsize=12,
        va="center",
        ha="center",
    )

    for i, methodi in enumerate(run_dirs.keys()):
        dfi = test_preds.query("method == '%s'" % methodi)

        fpr, tpr, _ = roc_curve(dfi["label"], dfi["CSVD"])
        axs[0].plot(fpr, tpr, "-", label=methodi, color=palette[i])
        axs[0].set_xlabel("False Positive Rate")
        axs[0].set_ylabel("True Positive Rate")

        prec, recall, _ = precision_recall_curve(dfi["label"], dfi["CSVD"])
        axs[1].plot(recall, prec, "-", label=methodi, color=palette[i])
        axs[1].set_xlabel("Recall")
        axs[1].set_ylabel("Precision")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=3,
        frameon=False,
        fancybox=False,
    )

    # 5. saving
    fig.savefig(osp.join(root, "plot_roc.png"), dpi=300)
    fig.savefig(osp.join(root, "plot_roc.pdf"))


if __name__ == "__main__":
    main()
