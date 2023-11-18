import argparse
import logging
# import os
import os.path as osp
import sys

import numpy as np
import optuna
import pandas as pd
from torch._dynamo.utils import import_submodule

# from datetime import datetime


# import torch

sys.path.append("/".join(osp.abspath(__file__).split("/")[:-2]))
from src.dataset import get_loaders
from src.model import CNN2dATT
from src.train import test_model, train_model
from src.utils import set_seed


def main():
    logging.basicConfig(level=logging.INFO)

    # 0. argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", default=5, type=int)
    parser.add_argument("--valid_size", default=0.1, type=float)
    parser.add_argument("--test_size", default=0.2, type=float)
    parser.add_argument("--seed", default=0, type=int)

    # parser.add_argument("--backbone", default="resnet34", type=str)
    # parser.add_argument("--no_backbone_pretrained", action="store_true")
    # parser.add_argument("--backbone_feature_index", default=None, type=int)
    # parser.add_argument("--backbone_freeze", action="store_true")
    # parser.add_argument("--no_satt", action="store_true")
    # parser.add_argument(
    #     "--satt_hiddens", nargs="*", default=[256], type=int
    # )
    # parser.add_argument(
    #     "--satt_acts", nargs="*", default=["tanh"], type=str
    # )
    # parser.add_argument("--no_satt_bn", action="store_true")
    # parser.add_argument("--satt_dp", default=None, type=float)
    # parser.add_argument("--no_iatt", action="store_true")
    # parser.add_argument("--iatt_hidden", default=256, type=int)
    # parser.add_argument("--iatt_bias", action="store_true")
    # parser.add_argument("--iatt_temperature", default=1.0, type=float)
    # parser.add_argument("--w_kl_satt", default=0.05, type=float)
    # parser.add_argument("--mlp_hiddens", default=[], nargs="*", type=int)
    # parser.add_argument("--mlp_act", default="relu", type=str)
    # parser.add_argument("--no_mlp_bn", action="store_true")
    # parser.add_argument("--mlp_dp", default=None, type=float)
    # parser.add_argument(
    #     "--loss_func", choices=["ce", "focal"], default="focal"
    # )
    #
    parser.add_argument("--device", default="cuda:1", type=str)
    parser.add_argument("--timeout", default=None, type=float, help="hours")
    parser.add_argument("--n_trials", default=None, type=int)
    # parser.add_argument("--nepoches", default=50, type=int)
    # parser.add_argument("--learning_rate", default=5e-4, type=float)
    # parser.add_argument("--save_root", default=None, type=str)
    # parser.add_argument("--no_modelcheckpoint", action="store_true")
    # parser.add_argument("--no_early_stop", action="store_true")
    # parser.add_argument("--early_stop_patience", default=10, type=int)
    # parser.add_argument("--lr_schedual", action="store_true")
    # parser.add_argument("--lr_sch_factor", default=0.1, type=float)
    # parser.add_argument("--lr_sch_patience", default=5, type=int)
    # parser.add_argument(
    #     "--monitor_metric", choices=["bacc", "acc", "auc"], default="bacc"
    # )
    # parser.add_argument(
    #     "--message_level",
    #     default=1,
    #     type=int,
    #     help=(
    #         "2 means all messages, 1 means all messages "
    #         "but epoch print, 0 means no message."
    #     ),
    # )
    args = parser.parse_args()

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
    )
    if args.cv is None:
        dataloaders_iter = [dataloaders_iter]
    else:
        # 生成器只能用一次，每次必须重新生成
        # 这里将其保存成一个list，就可以多次使用了
        dataloaders_iter = list(dataloaders_iter)

    def objective(trial: optuna.trial.Trial) -> float:
        # 1. sample the hparams
        n_satt_hiddens = trial.suggest_int("n_satt_hiddens", 0, 3)
        satt_act = trial.suggest_categorical(
            "satt_act", ["tanh", "relu", "selu", "gelu"]
        )
        satt_bn = trial.suggest_categorical("satt_bn", [True, False])
        satt_dp = trial.suggest_float("satt_dp", 0.0, 0.5)
        iatt = trial.suggest_categorical("iatt", [True, False])
        n_mlp_hiddens = trial.suggest_int("n_mlp_hiddens", 0, 3)
        mlp_act = trial.suggest_categorical(
            "mlp_act", ["tanh", "relu", "selu", "gelu"]
        )
        mlp_bn = trial.suggest_categorical("mlp_bn", [True, False])
        mlp_dp = trial.suggest_float("mlp_dp", 0.0, 0.5)
        w_kl_satt = trial.suggest_float("w_kl_satt", 0.0001, 0.1, log=True)
        lr = trial.suggest_float("lr", 0.0001, 0.001, log=True)

        set_seed(args.seed)

        all_bacc = []
        for loaders, _ in dataloaders_iter:
            # 2. model
            model = CNN2dATT(
                backbone="resnet34",
                backbone_pretrained=True,
                backbone_feature_index=None,
                backbone_freeze=False,
                spatial_attention=True,
                spatt_hiddens=[256] * n_satt_hiddens,
                spatt_activations=[satt_act] * n_satt_hiddens,
                spatt_bn=satt_bn,
                spatt_dp=satt_dp,
                instance_attention=iatt,
                inatt_hidden=256,
                inatt_bias=False,
                inatt_temperature=1.0,
                mlp_hiddens=[256] * n_mlp_hiddens,
                mlp_act=mlp_act,
                mlp_bn=mlp_bn,
                mlp_dp=mlp_dp,
                loss_func="focal",
                weight_kl_satt=w_kl_satt,
            )

            # 3. train
            train_model(
                model,
                loaders["train"],
                loaders["valid"],
                device=args.device,
                nepoches=50,
                learning_rate=lr,
                model_checkpoint=True,
                early_stop=True,
                early_stop_patience=10,
                lr_schedual=False,
                lr_sch_factor=0.1,
                lr_sch_patience=5,
                monitor_metric="bacc",
                message_level=0,
            )
            test_scores = test_model(
                model,
                loaders["test"],
                device=args.device,
                return_predict=False,
                progress_bar=False,
            )
            all_bacc.append(test_scores["bacc"])

        logging.info("all bacc: " + ",".join("%.4f" % v for v in all_bacc))
        return np.mean(all_bacc)

    study_name = "hparam_search"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=None if args.timeout is None else args.timeout * 3600,
    )


if __name__ == "__main__":
    main()
