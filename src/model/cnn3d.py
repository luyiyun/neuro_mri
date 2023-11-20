from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from monai.networks.layers import Act

from .utils import MLP, FocalLoss


class CNN3d(nn.Module):
    def __init__(
        self,
        in_channel: int = 1,
        nclasses: int = 2,
        hiddens: List[int] = [8, 32, 128],
        conv_ksize: Tuple[int, int, int] = (3, 3, 2),
        conv_stride: Tuple[int, int, int] = (1, 1, 1),
        conv_padding: Tuple[int, int, int] = (1, 1, 1),
        pool_ksize: Tuple[int, int, int] = (2, 2, 2),
        pool_stride: Tuple[int, int, int] = (2, 2, 2),
        norm: Literal["instance", "batch", "none"] = "instance",
        activition: str = "relu",
        global_pool: Literal["avg", "max"] = "max",
        mlp_hiddens: List[int] = [],
        mlp_act: str = "relu",
        mlp_bn: bool = True,
        mlp_dp: Optional[float] = None,
        loss_func: Literal["ce", "focal"] = "ce",
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        # 1. CNN backbone
        backbone = []
        for ind, (i, o) in enumerate(
            zip([in_channel] + hiddens[:-1], hiddens)
        ):
            backbone.append(
                nn.Conv3d(i, o, conv_ksize, conv_stride, conv_padding)
            )
            if norm == "instance":
                backbone.append(nn.InstanceNorm3d(o))
            elif norm == "batch":
                backbone.append(nn.BatchNorm3d(o))
            backbone.append(Act[activition]())
            if ind < (len(hiddens) - 1):
                backbone.append(nn.MaxPool3d(pool_ksize, pool_stride))
        self.backbone = nn.Sequential(*backbone)

        # 2. global pooling
        self.gpool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1))
            if global_pool == "avg"
            else nn.AdaptiveMaxPool3d((1, 1, 1)),
            nn.Flatten(),
        )

        # 3. MLP predictor
        self.predictor = MLP(
            hiddens[-1], nclasses, mlp_hiddens, mlp_act, mlp_bn, mlp_dp
        )

        # # ------ loss function ------
        if loss_func == "ce":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_func == "focal":
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, x):
        return self.predictor(self.gpool(self.backbone(x)))

    def step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = self.forward(x)
        loss = self.criterion(pred, y)
        return {"main": loss}, pred
