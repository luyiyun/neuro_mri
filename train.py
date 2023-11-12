import argparse
import json
import logging
import os
import os.path as osp
from datetime import datetime
from typing import Dict

import pandas as pd
import torch

from src.dataset import get_loaders
from src.model import CNN2dATT
from src.train import test_model, train_model


def save_json(obj: Dict, fn: str) -> None:
    with open(fn, "w") as f:
        json.dump(obj, f)


def main():
    logging.basicConfig(level=logging.INFO)

    # 0. argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_size", default=0.1, type=float)
    parser.add_argument("--test_size", default=0.2, type=float)
    parser.add_argument("--seed", default=2022, type=int)

    parser.add_argument("--no_backbone_pretrained", action="store_true")
    parser.add_argument("--backbone_freeze", action="store_true")
    parser.add_argument(
        "--loss_func", choices=["ce", "focal"], default="focal"
    )

    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--nepoches", default=10, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--save_root", default=None, type=str)
    parser.add_argument("--no_modelcheckpoint", action="store_true")
    parser.add_argument("--no_early_stop", action="store_true")
    parser.add_argument("--early_stop_patience", default=10, type=int)
    parser.add_argument(
        "--monitor_metric", choices=["bacc", "acc", "auc"], default="bacc"
    )
    parser.add_argument("--no_show_message", action="store_true")
    args = parser.parse_args()

    # 1. dataset
    df = pd.read_csv("/mnt/data1/tiantan/fn_rm_skull_mv.csv", index_col=0)
    df = df.loc[df.v1_filter2, :]
    loaders, classes = get_loaders(
        df,
        "rm_skull_fm",
        "label",
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
        return_classes_codes=True,
    )

    # 2. model
    model = CNN2dATT(
        backbone_pretrained=not args.no_backbone_pretrained,
        backbone_freeze=args.backbone_freeze,
        loss_func=args.loss_func,
    )

    # 3. train
    hist = train_model(
        model,
        loaders["train"],
        loaders["valid"],
        device=args.device,
        nepoches=args.nepoches,
        learning_rate=args.learning_rate,
        model_checkpoint=not args.no_modelcheckpoint,
        early_stop=not args.no_early_stop,
        early_stop_patience=args.early_stop_patience,
        monitor_metric=args.monitor_metric,
        show_message=not args.no_show_message,
    )
    test_scores, test_pred = test_model(
        model, loaders["test"], device=args.device, return_predict=True
    )
    logging.info(
        "Test: "
        + ", ".join(["%s:%.4f" % (k, vs[-1]) for k, vs in test_scores.items()])
    )

    # 4. saving
    if args.save_root is None:
        save_root = osp.join(
            "/mnt/data1/tiantan/results/%s"
            % datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        )
    else:
        save_root = args.save_root
    logging.info("the results saved in %s" % save_root)
    os.makedirs(save_root)

    save_json(args.__dict__, osp.join(save_root, "args.json"))

    hist_df = pd.DataFrame(hist["train"])
    hist_df["phase"] = "train"
    if "valid" in hist:
        hist_df_valid = pd.DataFrame(hist["valid"])
        hist_df_valid["phase"] = "valid"
        hist_df = pd.concat([hist_df, hist_df_valid])
    hist_df.to_csv(osp.join(save_root, "hist.csv"))

    save_json(test_scores, osp.join(save_root, "test_scores.json"))

    test_df = pd.DataFrame.from_records(loaders["test"].dataset.data)
    test_df = pd.concat(
        [test_df, pd.DataFrame(test_pred, columns=classes)], axis=1
    )
    test_df.to_csv(osp.join(save_root, "test_pred.csv"))

    torch.save(model.state_dict(), osp.join(save_root, "model.pth"))


if __name__ == "__main__":
    main()
