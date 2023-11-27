import logging
import os
import os.path as osp
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from monai.transforms import Resize
from scipy import ndimage

sys.path.append("/".join(osp.abspath(__file__).split("/")[:-2]))
from src.dataset import dfs2loaders
from src.utils import read_json


class Plotter:
    fn_col: str = "img"
    label_col: str = "label"
    depth: str = 12

    def __init__(self, run_dir: str) -> None:
        self._resize_func_12 = Resize((256, 256, 12), mode="trilinear")
        self._resize_func_24 = Resize((256, 256, 24), mode="trilinear")

        self._test_pred_df = pd.read_csv(
            osp.join(run_dir, "test_pred.csv"), index_col=0
        )
        self._sscores = torch.load(
            osp.join(run_dir, "sscores.pth"), map_location="cpu"
        ).numpy()
        self._iscores = torch.load(
            osp.join(run_dir, "iscores.pth"), map_location="cpu"
        ).numpy()
        trained_args = SimpleNamespace(
            **read_json(osp.join(run_dir, "args.json"))
        )
        classes = self._test_pred_df.columns.values[-2:]
        self._dataset = dfs2loaders(
            {"test": self._test_pred_df},
            self.fn_col,
            self.label_col,
            batch_size=16,
            num_workers=4,
            drop_last=False,
            classes=classes,
            slice_index=(
                trained_args.slice_index
                if "slice_index" in trained_args.__dict__
                else (None, None)
            ),
        )["test"].dataset

        self._slice_index = slice(None, None)
        if "slice_index" in trained_args.__dict__:
            slice_index_arg = trained_args.slice_index
            self._slice_index = (
                slice(None, None)
                if slice_index_arg is None
                else slice(slice_index_arg[1], -slice_index_arg[0])
            )

    def get_plot_metas(self, root: str) -> List[Dict]:
        plot_metas = []
        for subdir, _, fns in os.walk(root):
            original_fn = None
            for fni in fns:
                if fni.endswith(".nii"):
                    original_fn = fni
                    break
            label_fn = None
            for fni in fns:
                if fni.endswith(".nii.gz"):
                    label_fn = fni
                    break
            if original_fn is None or label_fn is None:
                continue

            # find the corresponding sscore
            original_name = original_fn[:-5]
            mask = list(
                map(
                    lambda x: original_name in x,
                    self._test_pred_df[self.fn_col].values,
                )
            )
            assert sum(mask) == 1
            ind = np.nonzero(mask)[0][0]
            resample_fn = self._test_pred_df[self.fn_col].iloc[ind]
            sscorei = self._resize_func_12(self._sscores[[ind]])[0]

            plot_metas.append(
                {
                    "ori": osp.join(subdir, original_fn),
                    "lesion": osp.join(subdir, label_fn),
                    "resample": resample_fn,
                    "sscore": sscorei,
                    "iscore": self._iscores[ind],
                    "label": self._test_pred_df[self.label_col].iloc[ind],
                    "MS_prob": self._test_pred_df["MS"].iloc[ind],
                    "CSVD_prob": self._test_pred_df["CSVD"].iloc[ind],
                    "array": self._dataset[ind]["img"][0].numpy(),
                }
            )

        return plot_metas

    def plot_instance_spatial_scores(
        self,
        plot_meta: Dict,
        figure: Optional[Figure] = None,
        index_str: Optional[Tuple[str, str, str]] = None,
        subtitles: Optional[Tuple[str, str, str]] = None,
        **fig_kws
    ):
        if figure is None:
            figure = plt.figure(constrained_layout=True, **fig_kws)
        figs = figure.subfigures(ncols=1, nrows=3, hspace=0.0)
        fig_iscore, fig_sscore, fig_true = figs[0], figs[1], figs[2]
        if index_str is not None:
            fig_iscore.text(
                0.01,
                0.95,
                index_str[0],
                fontweight="bold",
                fontsize=12,
                va="center",
                ha="center",
            )
            fig_sscore.text(
                0.01,
                0.95,
                index_str[1],
                fontweight="bold",
                fontsize=12,
                va="center",
                ha="center",
            )
            fig_true.text(
                0.01,
                0.95,
                index_str[2],
                fontweight="bold",
                fontsize=12,
                va="center",
                ha="center",
            )

        if subtitles is not None:
            for figi, titlei in zip(figs, subtitles):
                figi.suptitle(titlei)

        depth = self.depth
        item = plot_meta
        print(item["ori"])

        imgi = item["array"]
        ssi = item["sscore"]
        isi = item["iscore"]
        ssi_ = ssi * isi
        ssi_min, ssi_max = ssi_.min(), ssi_.max()
        ssi_ = (ssi_ - ssi_min) / ssi_max

        ax = fig_iscore.subplots()
        bar_iscore = ax.bar(x=np.arange(1, depth + 1), height=isi, width=0.5)
        ax.bar_label(bar_iscore, padding=3, fmt="%.4f")
        ax.set_xlim(0.42, 12.5)
        ax.xaxis.set_ticks(np.arange(1, depth + 1))
        ax.xaxis.set_ticklabels(np.arange(1, depth + 1))
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.spines[["right", "top"]].set_visible(False)

        axs = fig_sscore.subplots(ncols=depth)
        for i in range(depth):
            ax = axs[i]
            imgii = imgi[..., i]
            ssii_ = ssi_[..., i]
            imgii, ssii_ = ndimage.rotate(imgii, 90), ndimage.rotate(ssii_, 90)
            ax.imshow(imgii, cmap="gray")
            ax.imshow(ssii_, cmap="jet", alpha=0.3, vmax=1.0, vmin=0.0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
        fig_sscore.colorbar(
            ScalarMappable(Normalize(ssi_min, ssi_max), "jet"),
            ax=axs[0],
            location="left",
        )

        lesion = nib.load(item["lesion"]).get_fdata()
        lesion = (lesion - lesion.min()) / lesion.max()
        lesion = self._resize_func_24(np.expand_dims(lesion, 0))[
            0, ..., self._slice_index
        ]
        # set a empty subfigure as placeholder
        fig_true = fig_true.subfigures(
            nrows=1, ncols=2, width_ratios=[1, 50]
        )[1]
        axs = fig_true.subplots(ncols=depth)
        for i in range(depth):
            ax = axs[i]
            lesioni = ndimage.rotate(lesion[..., i], 90)
            ax.imshow(lesioni, cmap="gray")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])


def main():
    logging.basicConfig(level=logging.INFO)

    root = "/mnt/data1/tiantan/results"
    plotter = Plotter(
        run_dir="/mnt/data1/tiantan/results/2023-11-19_12-08-17/fold0/"
    )

    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(constrained_layout=True, figsize=(20, 12))
    subfigs = fig.subfigures(ncols=1, nrows=2, hspace=0.01)
    subtitles = [
        "Instance Attention Scores",
        "Spatial Attention Scores",
        "True Lesions",
    ]
    subfigs[0].suptitle("MS case")
    subfigs[1].suptitle("CSVD case")

    csvd_metas = plotter.get_plot_metas("/mnt/data1/tiantan/labeling/CSVD/")
    plotter.plot_instance_spatial_scores(
        csvd_metas[3], subfigs[0], ("A", "B", "C"), subtitles
    )  # 0 or 3

    ms_metas = plotter.get_plot_metas("/mnt/data1/tiantan/labeling/MS/")
    plotter.plot_instance_spatial_scores(
        ms_metas[0], subfigs[1], ("D", "E", "F"), subtitles
    )  #

    # 5. saving
    fig.savefig(osp.join(root, "plot_heatmap.png"))


if __name__ == "__main__":
    main()
