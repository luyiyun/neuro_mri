import logging
import warnings
from collections import defaultdict
from copy import deepcopy
from math import inf
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm


class BinaryBalancedAccuracy(Metric):
    def __init__(self, adjusted: bool = True):
        super().__init__()
        self._adjusted = adjusted
        self.add_state(
            "confusion_matrix",
            default=torch.zeros((2, 2)),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        cm = tm.functional.confusion_matrix(preds, target, task="binary")
        self.confusion_matrix += cm

    def compute(self) -> torch.Tensor:
        cm = self.confusion_matrix
        per_class = cm.diagonal() / cm.sum(dim=1)
        score = per_class.mean()
        if self._adjusted:
            n_classes = per_class.size(0)
            chance = 1 / n_classes
            score -= chance
            score /= 1 - chance
        return score


def get_metrics(device: torch.device) -> Dict[str, Metric]:
    return {
        "bacc": BinaryBalancedAccuracy().to(device),
        "acc": tm.classification.Accuracy(task="binary").to(device),
        "auc": tm.classification.AUROC(task="binary").to(device),
        "sensitivity": tm.classification.Recall(task="binary").to(device),
        "specificity": tm.classification.Specificity(task="binary").to(device),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: Optional[DataLoader] = None,
    device: str = "cpu",
    nepoches: int = 100,
    learning_rate: float = 5e-4,
    model_checkpoint: bool = True,
    early_stop: bool = True,
    early_stop_patience: int = 5,
    monitor_metric: Literal["bacc", "acc", "auc"] = "bacc",
    show_message: bool = True,
) -> Dict:
    if model_checkpoint and (valid_loader is None):
        warnings.warn(
            "model_checkpoint only works when valid_loader is available."
        )
    if early_stop and (valid_loader is None):
        warnings.warn("early_stop only works when valid_loader is available.")

    device = torch.device(device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    metrics = get_metrics(device)
    scores = {"train": defaultdict(list)}
    if valid_loader is not None:
        scores["valid"] = defaultdict(list)

    if model_checkpoint or early_stop:
        best_epoch = -1
        best_score = -inf
    if model_checkpoint:
        best_model = deepcopy(model.state_dict())
    if early_stop:
        n_no_improve = 0

    for e in tqdm(range(nepoches), desc="Epoch: "):
        # train loop
        model.train()
        for mi in metrics.values():
            mi.reset()
        total_loss, cnt = 0.0, 0
        with torch.enable_grad():
            for batch in tqdm(
                train_loader, desc="Batch(train): ", leave=False
            ):
                optimizer.zero_grad()
                x = batch["img"].to(device)
                y = batch["label"].to(device)
                loss, pred = model.step(x, y)
                loss.backward()
                optimizer.step()

                # loss and metrics
                total_loss += loss.item() * x.size(0)
                cnt += x.size(0)
                for mi in metrics.values():
                    mi(pred[:, 1], y)
        scores["train"]["loss"].append(total_loss / cnt)
        for k, mi in metrics.items():
            scores["train"][k].append(mi.compute().item())
        if show_message:
            tqdm.write(
                "Train: "
                + ", ".join(
                    [
                        "%s:%.4f" % (k, vs[-1])
                        for k, vs in scores["train"].items()
                    ]
                )
            )

        if valid_loader is None:
            continue

        # valid loop
        model.eval()
        for mi in metrics.values():
            mi.reset()
        total_loss, cnt = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(
                valid_loader, desc="Batch(valid): ", leave=False
            ):
                x = batch["img"].to(device)
                y = batch["label"].to(device)
                loss, pred = model.step(x, y)

                # loss and metrics
                total_loss += loss.item() * x.size(0)
                cnt += x.size(0)
                for mi in metrics.values():
                    mi(pred[:, 1], y)
        scores["valid"]["loss"].append(total_loss / cnt)
        for k, mi in metrics.items():
            scores["valid"][k].append(mi.compute().item())
        if show_message:
            tqdm.write(
                "Valid: "
                + ", ".join(
                    [
                        "%s:%.4f" % (k, vs[-1])
                        for k, vs in scores["valid"].items()
                    ]
                )
            )

        if model_checkpoint or early_stop:
            now_score = scores["valid"][monitor_metric][-1]
            flag_improve = now_score > best_score
            if flag_improve:
                best_score = now_score
        if model_checkpoint:
            if flag_improve:
                best_model = deepcopy(model.state_dict())
                best_epoch = e
        if early_stop:
            if flag_improve:
                n_no_improve = 0
            else:
                n_no_improve += 1
                if n_no_improve > early_stop_patience:
                    logging.info(
                        "%s has %d epoches that do not improve, early stop."
                        % (monitor_metric, early_stop_patience)
                    )
                    break

    if valid_loader is not None and model_checkpoint:
        model.load_state_dict(best_model)
        logging.info(
            "best model at epoch %d, whose valid %s is %.4f"
            % (best_epoch, monitor_metric, best_score)
        )

    return scores


def test_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    return_predict: bool = True,
) -> Union[Dict, Tuple[Dict, np.ndarray]]:
    device = torch.device(device)
    model.to(device)

    metrics = get_metrics(device)
    scores = defaultdict(list)

    model.eval()
    for mi in metrics.values():
        mi.reset()

    total_loss, cnt = 0.0, 0
    total_predict = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Batch(test): "):
            x = batch["img"].to(device)
            y = batch["label"].to(device)
            loss, pred = model.step(x, y)

            # loss and metrics
            total_loss += loss.item() * x.size(0)
            cnt += x.size(0)
            for mi in metrics.values():
                mi(pred[:, 1], y)

            if return_predict:
                total_predict.append(pred)

    scores["loss"].append(total_loss / cnt)
    for k, mi in metrics.items():
        scores[k].append(mi.compute().item())

    if not return_predict:
        return scores

    total_predict = torch.cat(total_predict).detach().cpu().numpy()
    return scores, total_predict


def pred_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    return_ndarray: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device(device)
    model.to(device)

    preds, sscores, iscores = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Batch(test): "):
            x = batch["img"].to(device)
            # y = batch["label"].to(device)
            pred, sscore, iscore = model(x)
            preds.append(pred)
            sscores.append(sscore)
            iscores.append(iscore)

    preds = torch.cat(preds).detach().cpu().numpy()
    if torch.is_tensor(sscores[0]):
        sscores = torch.cat(sscores)
        if return_ndarray:
            sscores = sscores.detach().cpu().numpy()
    else:
        sscores = None
    if torch.is_tensor(iscores[0]):
        iscores = torch.cat(iscores)
        if return_ndarray:
            iscores = iscores.detach().cpu().numpy()
    else:
        iscores = None

    return preds, sscores, iscores
