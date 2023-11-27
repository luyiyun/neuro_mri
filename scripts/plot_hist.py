import logging
import os.path as osp

# import sci_palettes
import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# from scipy.interpolate import splrep, BSpline
from scipy.signal import savgol_filter


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
    }
    nmethods = len(run_dirs)

    # 2. load the hist records
    hists = []
    for k, run_dir in run_dirs.items():
        for i in range(5):
            fn = osp.join(root, run_dir, "fold%d" % i, "hist.csv")
            hist_i = pd.read_csv(fn, index_col=0).reset_index(names="epoch")
            hist_i["fold"] = i
            hist_i["method"] = k
            hists.append(hist_i)
    hists = pd.concat(hists)

    metric_mapping = {
        "main": "Total Loss",
        "bacc": "Balanced Accuracy",
        "acc": "Accuracy",
        "auc": "Area Under ROC Curver",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
    }
    hists.rename(columns=metric_mapping, inplace=True)

    # 3. smooth the curve
    def smooth_df(df):
        # x = df["epoch"].values
        for k in metric_mapping.values():
            y = df[k].values
            df[k] = savgol_filter(y, 10, 2)
            # tck = splrep(x, y, s=0)
            # df[k] = BSpline(*tck)(x)
        return df

    hists = hists.groupby(["phase", "fold", "method"]).apply(smooth_df)
    hists = (
        hists.reset_index(drop=True)
        .groupby(["phase", "method", "epoch"])
        .mean()
    )
    hists = hists.reset_index()

    # 4. plot the losses
    plt.rcParams["font.family"] = "Times New Roman"
    # sci_palettes.register_cmap()
    # palette = sns.color_palette("npg_nrc")[1:(nmethods + 1)]
    palette = cc.glasbey_category10[:nmethods]

    fig = plt.figure(constrained_layout=True, figsize=(10, 8))
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    full_axs = []
    for i, phasei in enumerate(["train", "valid"]):
        subfigs[i].text(
            0.03,
            0.98,
            "AB"[i],
            fontweight="bold",
            fontsize=12,
            va="center",
            ha="center",
        )
        subfigs[i].suptitle(phasei.capitalize(), fontsize="x-large")
        axs = subfigs[i].subplots(ncols=2, nrows=3)
        axs = axs.flatten()
        full_axs += axs.tolist()

        sub_data = hists.query("phase == '%s'" % phasei)
        for j, metrici in enumerate(metric_mapping.values()):
            ax = axs[j]
            sns.lineplot(
                data=sub_data,
                x="epoch",
                y=metrici,
                hue="method",
                # units="fold",
                estimator=None,
                ax=ax,
                palette=palette,
            )
            ax.set_ylabel("")
            ax.set_xlabel("" if j < 3 else "Epoch")
            ax.set_title(metrici)
            ax.spines[["right", "top"]].set_visible(False)

    handles, labels = full_axs[0].get_legend_handles_labels()
    for ax in full_axs:
        ax.get_legend().remove()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=3,
        frameon=False,
        fancybox=False,
    )

    fig.savefig(osp.join(root, "plot_hist.png"), dpi=300)
    fig.savefig(osp.join(root, "plot_hist.pdf"))


if __name__ == "__main__":
    main()
