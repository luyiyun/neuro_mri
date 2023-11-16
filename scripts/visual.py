import sys
import argparse
import logging
import math
import os
import os.path as osp
import re
from datetime import datetime
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as ssp
import torch
from monai.transforms import Resize
from tqdm import tqdm

sys.path.append("/".join(osp.abspath(__file__).split("/")[:-2]))
from src.dataset import dfs2loaders
from src.model import CNN2dATT
from src.train import pred_model
from src.utils import read_json


def process_one_run(one_run_dir: str, device: str):
    trained_args = SimpleNamespace(
        **read_json(osp.join(one_run_dir, "args.json"))
    )
    fn_col, label_col = "img", "label"

    # 1. dataset
    logging.info("load test dataset.")
    test_pred_df = pd.read_csv(
        osp.join(one_run_dir, "test_pred.csv"), index_col=0
    )
    classes = test_pred_df.columns.values[-2:]
    loader = dfs2loaders(
        {"test": test_pred_df},
        fn_col,
        label_col,
        batch_size=16,
        num_workers=4,
        drop_last=False,
        classes=classes,
    )["test"]

    # 2. model
    logging.info("load test dataset.")
    model = CNN2dATT(
        backbone_pretrained=not trained_args.no_backbone_pretrained,
        backbone_freeze=trained_args.backbone_freeze,
        spatial_attention=not trained_args.no_satt,
        spatt_hiddens=trained_args.satt_hiddens,
        spatt_activations=trained_args.satt_acts,
        spatt_bn=not trained_args.no_satt_bn,
        spatt_dp=trained_args.satt_dp,
        instance_attention=not trained_args.no_iatt,
        inatt_hidden=trained_args.iatt_hidden,
        inatt_bias=trained_args.iatt_bias,
        inatt_temperature=trained_args.iatt_temperature,
        loss_func=trained_args.loss_func,
    )
    model.load_state_dict(
        torch.load(osp.join(one_run_dir, "model.pth"), map_location=device)
    )

    # 3. predict
    preds, sscores, iscores = pred_model(model, loader, device=device)
    if sscores is not None:
        torch.save(sscores, osp.join(one_run_dir, "sscores.pth"))
    if iscores is not None:
        torch.save(iscores, osp.join(one_run_dir, "iscores.pth"))
    # 本次预测与之前训练时的预测是一样的
    assert np.allclose(preds, test_pred_df.iloc[:, -2:].values)

    # 4. visual
    visual_path = osp.join(one_run_dir, "visual")
    for ci in classes:
        os.makedirs(osp.join(visual_path, ci), exist_ok=True)
    preds_proba = ssp.softmax(preds, axis=1)
    sscores = Resize((256, 256, 24), mode="trilinear")(sscores)

    sscores = sscores.detach().cpu().numpy()
    iscores = iscores.detach().cpu().numpy()
    for i, sample in tqdm(
        enumerate(loader.dataset),
        desc="Visual(test): ",
        total=len(loader.dataset),
    ):
        img = sample["img"][0].cpu().numpy()  # w,h,d
        label = test_pred_df[label_col].iloc[i]

        sscorei = sscores[i]  # w,h,d
        iscorei = iscores[i]  # d
        predi = preds_proba[i]

        ncol = math.ceil(math.sqrt(img.shape[-1]))
        nrow = math.ceil(img.shape[-1] / ncol)
        fig, axs = plt.subplots(
            nrows=nrow, ncols=ncol, figsize=(ncol * 3, nrow * 3)
        )
        axs = axs.flatten()
        for j in range(img.shape[-1]):
            axs[j].imshow(img[..., j], cmap="gray")
            axs[j].imshow(sscorei[..., j], cmap="jet", alpha=0.3)
            axs[j].set_title("%.4f" % (iscorei[j]))
        title = "True=%s, Prob(%s)=%.2f, Prob(%s)=%.2f" % (
            label,
            classes[0],
            predi[0],
            classes[1],
            predi[1],
        )
        fig.suptitle(title)
        fig.tight_layout()

        fname = osp.basename(test_pred_df[fn_col].iloc[i])[:-4]
        visual_name = osp.join(visual_path, label, fname + ".png")
        fig.savefig(visual_name)
        plt.close()


def main():
    logging.basicConfig(level=logging.INFO)

    # 0. argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/mnt/data1/tiantan/results/", type=str
    )
    parser.add_argument("--run_dir", default=None, type=str)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    # select the running results
    if args.run_dir is None:
        runs = os.listdir(args.root)
        runs_df = [
            {
                "date": datetime.strptime(runi, "%Y-%m-%d_%H-%M-%S"),
                "path": runi,
            }
            for runi in runs
            if re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", runi)
        ]
        runs_df = pd.DataFrame.from_records(runs_df).sort_values(
            "date", ascending=False
        )
        last_run = runs_df["path"].iloc[0]
        logging.info("select the results of last run: %s" % last_run)
        run_dir = osp.join(args.root, last_run)
    else:
        run_dir = osp.join(args.root, args.run_dir)

    if "fold0" in os.listdir(run_dir):
        res_dir_subs = [
            osp.join(run_dir, fn)
            for fn in os.listdir(run_dir)
            if fn.startswith("fold")
        ]
        res_dir_subs = sorted(res_dir_subs)
    else:
        res_dir_subs = [run_dir]

    for res_dir_sub in res_dir_subs:
        logging.info("")
        logging.info("process run %s" % res_dir_sub)
        process_one_run(res_dir_sub, args.device)


if __name__ == "__main__":
    main()
