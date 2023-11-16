import sys
import argparse
import logging
import os
import os.path as osp
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append("/".join(osp.abspath(__file__).split("/")[:-2]))
from src.utils import read_json, get_config_from_args, filter_runs


def main():
    logging.basicConfig(level=logging.INFO)

    # 0. argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/mnt/data1/tiantan/results/", type=str
    )
    parser.add_argument("--start", default="2023-11-14", type=str)
    parser.add_argument("--end", default=None, type=str)
    parser.add_argument("--run_dirs", default=None, type=str, nargs="+")
    parser.add_argument("--save_fn", default=None, type=str)
    parser.add_argument(
        "--phase", default="all", choices=["valid", "train", "all"]
    )
    parser.add_argument("--metric", default=None, type=str, nargs="+")
    parser.add_argument("--config", default=None, nargs="+", type=str)
    args = parser.parse_args()

    # 1. select the runs
    if args.run_dirs is not None:
        run_dirs = [
            {"fdir": osp.join(args.root, runi), "dir": runi}
            for runi in args.run_dirs
        ]
    else:
        run_dirs = filter_runs(args.root, args.start, args.end)

    # 2. load the training histories
    hists, config_mappings = [], {}
    for runi in run_dirs:
        fdir = runi["fdir"]
        if "fold0" not in os.listdir(fdir):
            if args.config is not None:
                configs = read_json(osp.join(fdir, "args.json"))
                config_mappings[runi["dir"]] = configs
            hist_i = pd.read_csv(osp.join(fdir, "hist.csv"), index_col=0)
            hist_i["run"] = runi["dir"]
            hists.append(hist_i)
        else:
            if args.config is not None:
                configs = read_json(osp.join(fdir, "fold0", "args.json"))
                config_mappings[runi["dir"]] = configs
            for subdir in os.listdir(fdir):
                match_res = re.search(r"fold(\d+?)", subdir)
                if match_res:
                    hist_i = pd.read_csv(
                        osp.join(fdir, subdir, "hist.csv"), index_col=0
                    )
                    hist_i["run"] = runi["dir"]
                    hist_i["fold"] = int(match_res.group(1))
                    hists.append(hist_i)
    hists = pd.concat(hists, axis=0)
    hists.reset_index(names="epoch", inplace=True)  # the epoch is the index

    # 3. plot the losses
    metric_mapping = {
        "main": "Total Loss",
        "bacc": "Balanced Accuracy",
        "acc": "Accuracy",
        "auc": "Area Under ROC curver",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
    }
    phases = ["train", "valid"] if args.phase == "all" else [args.phase]
    metrics = metric_mapping.keys() if args.metric is None else args.metric
    ncol = len(phases)
    nrow = len(metrics)

    fig, axs = plt.subplots(
        nrows=nrow,
        ncols=ncol,
        figsize=(3.5 * ncol + 3, 3 * nrow),  # add space for legend
        squeeze=False,
        layout="constrained",
    )
    all_handles, all_labels = [], []
    for j, phase in enumerate(phases):
        sub_data = hists.query("phase == '%s'" % phase)
        for i, metric in enumerate(metrics):
            ax = axs[i, j]
            sns.lineplot(
                data=sub_data,
                x="epoch",
                y=metric,
                hue="run",
                units="fold",
                estimator=None,
                ax=ax,
                # legend=False,
            )
            ax.set_ylabel("")
            ax.set_xlabel("Epoch" if i == (nrow - 1) else "")
            ax.set_title(
                "%s %s" % (phase.capitalize(), metric_mapping[metric])
            )

            for hi, li in zip(*ax.get_legend_handles_labels()):
                if li not in all_labels:
                    all_handles.append(hi)
                    all_labels.append(li)

            ax.get_legend().remove()

    if args.config is not None:
        new_all_labels = []
        for li in all_labels:
            configi = config_mappings[li]
            new_li = ",".join(
                [
                    "%s=%.3f" % (k, get_config_from_args(configi, k))
                    if isinstance(get_config_from_args(configi, k), float)
                    else "%s=%s" % (k, str(get_config_from_args(configi, k)))
                    for k in args.config
                ]
            )
            new_all_labels.append(new_li)
        all_labels = new_all_labels
    fig.legend(
        all_handles,
        all_labels,
        loc="outside right center",
    )

    # 4. saving
    if args.save_fn is None:
        save_fn = osp.join(args.root, "plot_hist_%s.png" % args.phase)
    else:
        save_fn = args.save_fn
    logging.info("the losses figure is saved in %s" % save_fn)
    fig.savefig(save_fn)


if __name__ == "__main__":
    main()