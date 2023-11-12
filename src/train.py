import logging
import warnings
from collections import defaultdict
from copy import deepcopy
from math import inf
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: Optional[DataLoader] = None,
    device: str = "cpu",
    nepoches: int = 100,
    learning_rate: float = 5e-4,
    model_checkpoint: bool = True,
    show_message: bool = True,
) -> Dict:
    if model_checkpoint and (valid_loader is None):
        warnings.warn(
            "model_checkpoint only works when valid_loader is available."
        )

    device = torch.device(device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # if self._cfgt.lrsch:
    #     stones = [
    #         int(0.3 * self._cfgt.max_epoch),
    #         int(0.6 * self._cfgt.max_epoch)
    #     ]
    #     lrsch = torch.optim.lr_scheduler.MultiStepLR(
    #         optimizer, stones, 0.5
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": lrsch
    #     }
    #
    # return optimizer

    metrics = {
        "acc": tm.classification.Accuracy(task="binary").to(device),
        "auc": tm.classification.AUROC(task="binary").to(device),
        "sensitivity": tm.classification.Recall(task="binary").to(device),
        "specificity": tm.classification.Specificity(task="binary").to(device),
    }
    scores = {"train": defaultdict(list)}
    if valid_loader is not None:
        scores["valid"] = defaultdict(list)
    if model_checkpoint:
        best_model = deepcopy(model.state_dict())
        best_epoch = -1
        best_score = -inf

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
            scores["train"][k].append(mi.compute())
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
            scores["valid"][k].append(mi.compute())
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

        if model_checkpoint:
            now_score = scores["valid"]["auc"][-1]
            if now_score > best_score:
                best_model = deepcopy(model.state_dict())
                best_epoch = e
                best_score = now_score

    if valid_loader is not None and model_checkpoint:
        model.load_state_dict(best_model)
        logging.info(
            "best model at epoch %d, whose valid auc is %.4f"
            % (best_epoch, best_score)
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

    metrics = {
        "acc": tm.classification.Accuracy(task="binary").to(device),
        "auc": tm.classification.AUROC(task="binary").to(device),
        "sensitivity": tm.classification.Recall(task="binary").to(device),
        "specificity": tm.classification.Specificity(task="binary").to(device),
    }
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
        scores[k].append(mi.compute())

    if not return_predict:
        return scores

    total_predict = torch.cat(total_predict).detach().cpu().numpy()
    return scores, total_predict
