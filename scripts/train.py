import argparse
import logging
import os
import os.path as osp
import sys
from datetime import datetime

import pandas as pd
import torch

sys.path.append("/".join(osp.abspath(__file__).split("/")[:-2]))
from src.dataset import get_loaders
from src.model import CNN2dATT
from src.train import test_model, train_model
from src.utils import save_json, set_seed


def main():
    logging.basicConfig(level=logging.INFO)

    # 0. argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", default=None, type=int)
    parser.add_argument("--valid_size", default=0.1, type=float)
    parser.add_argument("--test_size", default=0.2, type=float)
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--slice_index", default=None, nargs=2, type=int)

    parser.add_argument("--backbone", default="resnet34", type=str)
    parser.add_argument("--no_backbone_pretrained", action="store_true")
    parser.add_argument("--backbone_feature_index", default=None, type=int)
    parser.add_argument("--backbone_freeze", action="store_true")
    parser.add_argument("--no_satt", action="store_true")
    parser.add_argument("--satt_hiddens", nargs="*", default=[256], type=int)
    parser.add_argument("--satt_acts", nargs="*", default=["tanh"], type=str)
    parser.add_argument("--no_satt_bn", action="store_true")
    parser.add_argument("--satt_dp", default=None, type=float)
    parser.add_argument("--no_iatt", action="store_true")
    parser.add_argument("--iatt_hidden", default=256, type=int)
    parser.add_argument("--iatt_bias", action="store_true")
    parser.add_argument("--iatt_temperature", default=1.0, type=float)
    parser.add_argument(
        "--w_kl_satt",
        default=0.05,
        type=lambda x: None if x == "None" else float(x),
    )
    parser.add_argument("--mlp_hiddens", default=[], nargs="*", type=int)
    parser.add_argument("--mlp_act", default="relu", type=str)
    parser.add_argument("--no_mlp_bn", action="store_true")
    parser.add_argument("--mlp_dp", default=None, type=float)
    parser.add_argument(
        "--loss_func", choices=["ce", "focal"], default="focal"
    )

    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--nepoches", default=50, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--save_root", default=None, type=str)
    parser.add_argument("--no_modelcheckpoint", action="store_true")
    parser.add_argument("--no_early_stop", action="store_true")
    parser.add_argument("--early_stop_patience", default=10, type=int)
    parser.add_argument("--lr_schedual", action="store_true")
    parser.add_argument("--lr_sch_factor", default=0.1, type=float)
    parser.add_argument("--lr_sch_patience", default=5, type=int)
    parser.add_argument(
        "--monitor_metric", choices=["bacc", "acc", "auc"], default="bacc"
    )
    parser.add_argument(
        "--message_level",
        default=1,
        type=int,
        help=(
            "2 means all messages, 1 means all messages "
            "but epoch print, 0 means no message."
        ),
    )
    args = parser.parse_args()

    if args.save_root is None:
        save_root = osp.join(
            "/mnt/data1/tiantan/results/%s"
            % datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        )
    else:
        save_root = args.save_root
    logging.info("the results saved in %s" % save_root)

    set_seed(args.seed)

    # 1. dataset
    df = pd.read_csv("/mnt/data1/tiantan/fn_rm_skull_mv.csv", index_col=0)
    df = df.loc[df.v1_filter2, :]
    dataloaders_iter = get_loaders(
        df,
        "rm_skull_fm",
        "label",
        cv=args.cv,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
        return_classes_codes=True,
        slice_index=(
            (None, None) if args.slice_index is None else args.slice_index
        ),
    )
    if args.cv is None:
        dataloaders_iter = [dataloaders_iter]

    all_test_scores = []
    for foldi, (loaders, classes) in enumerate(dataloaders_iter):
        if args.cv is not None:
            logging.info("")
            logging.info("Fold = %d" % foldi)

        # 2. model
        model = CNN2dATT(
            backbone=args.backbone,
            backbone_pretrained=not args.no_backbone_pretrained,
            backbone_feature_index=args.backbone_feature_index,
            backbone_freeze=args.backbone_freeze,
            spatial_attention=not args.no_satt,
            spatt_hiddens=args.satt_hiddens,
            spatt_activations=args.satt_acts,
            spatt_bn=not args.no_satt_bn,
            spatt_dp=args.satt_dp,
            instance_attention=not args.no_iatt,
            inatt_hidden=args.iatt_hidden,
            inatt_bias=args.iatt_bias,
            inatt_temperature=args.iatt_temperature,
            mlp_hiddens=args.mlp_hiddens,
            mlp_act=args.mlp_act,
            mlp_bn=not args.no_mlp_bn,
            mlp_dp=args.mlp_dp,
            loss_func=args.loss_func,
            weight_kl_satt=args.w_kl_satt,
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
            lr_schedual=args.lr_schedual,
            lr_sch_factor=args.lr_sch_factor,
            lr_sch_patience=args.lr_sch_patience,
            monitor_metric=args.monitor_metric,
            message_level=args.message_level,
        )
        if "test" in loaders:
            test_scores, test_pred = test_model(
                model, loaders["test"], device=args.device, return_predict=True
            )
            logging.info(
                "Test: "
                + ", ".join(
                    ["%s:%.4f" % (k, v) for k, v in test_scores.items()]
                )
            )

        # 4. saving
        if args.cv is None:
            save_root_i = save_root
        else:
            save_root_i = osp.join(save_root, "fold%d" % foldi)
        os.makedirs(save_root_i)

        save_json(args.__dict__, osp.join(save_root_i, "args.json"))

        torch.save(model.state_dict(), osp.join(save_root_i, "model.pth"))

        hist_df = pd.DataFrame(hist["train"])
        hist_df["phase"] = "train"
        if "valid" in hist:
            hist_df_valid = pd.DataFrame(hist["valid"])
            hist_df_valid["phase"] = "valid"
            hist_df = pd.concat([hist_df, hist_df_valid])
        hist_df.to_csv(osp.join(save_root_i, "hist.csv"))

        if "test" in loaders:
            save_json(test_scores, osp.join(save_root_i, "test_scores.json"))

            test_df = pd.DataFrame.from_records(loaders["test"].dataset.data)
            test_df = pd.concat(
                [test_df, pd.DataFrame(test_pred, columns=classes)], axis=1
            )
            test_df.to_csv(osp.join(save_root_i, "test_pred.csv"))

            test_scores["fold"] = foldi
            all_test_scores.append(test_scores)

    if all_test_scores:
        all_test_scores_df = pd.DataFrame.from_records(all_test_scores)
        print(all_test_scores_df)
        all_test_scores_df.to_csv(osp.join(save_root, "all_test_scores.csv"))


if __name__ == "__main__":
    main()
