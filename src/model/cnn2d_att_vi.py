from itertools import chain
from typing import List, Literal, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from monai.networks.layers import Act
from torch.distributions import RelaxedBernoulli

from .cnn2d_att import get_resnet_bone, MLP, FocalLoss


class SpatialAttentionVI(nn.Module):
    def __init__(
        self,
        in_channel: int,
        nclasses: int,
        hiddens: List[int],
        activations: List[str],
        bn: bool = False,
        dp: Optional[float] = None,
    ) -> None:
        super().__init__()
        layers = []
        for i, o, ai in zip(
            [in_channel + nclasses] + hiddens[:-1], hiddens, activations
        ):
            layers.append(nn.Conv2d(i, o, 1))
            if bn:
                layers.append(nn.BatchNorm2d(o))
            layers.append(Act[ai]())
            if dp is not None:
                layers.append(nn.Dropout2d(dp))
        layers.append(nn.Conv2d(hiddens[-1], 1, 1))
        self._net = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        sscore = self._net(x)  # (bz,1,x,y)
        return RelaxedBernoulli(temperature, logits=sscore)


class InstanceAttention(nn.Module):
    def __init__(
        self,
        in_channel: int,
        hidden: int,
        bias: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self._temp = temperature
        self._fc1 = nn.Sequential(
            nn.Linear(in_channel, hidden, bias=bias), nn.Tanh()
        )
        self._fc2 = nn.Sequential(
            nn.Linear(in_channel, hidden, bias=bias), nn.Sigmoid()
        )
        self._fc3 = nn.Linear(hidden, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key = self._fc1(x)
        value = self._fc2(x)
        score = self._fc3(key * value)  # (b,z,1)
        score = torch.softmax(score / self._temp, dim=1)
        return score


class CNN2dATTVI(nn.Module):
    def __init__(
        self,
        nchannels: int = 1,
        nclasses: int = 2,
        backbone: str = "resnet18",
        backbone_pretrained: bool = True,
        backbone_freeze: bool = False,
        spatial_attention: bool = True,
        spatt_hiddens: List[int] = [256, 256],
        spatt_activations: List[str] = ["tanh", "tanh"],
        spatt_bn: bool = False,
        spatt_dp: Optional[float] = None,
        instance_attention: bool = True,
        inatt_hidden: int = 256,
        inatt_bias: bool = False,
        inatt_temperature: float = 1.0,
        mlp_hiddens: List[int] = [],
        mlp_act: str = "relu",
        mlp_bn: bool = True,
        mlp_dp: Optional[float] = None,
        loss_func: Literal["ce", "focal"] = "ce",
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
    ):
        assert loss_func in ["ce", "focal"]
        assert focal_alpha >= 0.0 and focal_alpha <= 1.0

        self._bb_freeze = backbone_freeze

        super().__init__()

        # 1. CNN backbone
        if backbone.startswith(backbone):
            self.backbone, backbone_outdims = get_resnet_bone(
                backbone, backbone_pretrained, nchannels
            )
        else:
            raise NotImplementedError

        # 2. Spatial Attention
        if spatial_attention:
            self.satt = SpatialAttention(
                backbone_outdims,
                spatt_hiddens,
                spatt_activations,
                spatt_bn,
                spatt_dp,
            )
        else:
            self.satt = None

        # 3. Instance Attention
        if instance_attention:
            self.iatt = InstanceAttention(
                backbone_outdims, inatt_hidden, inatt_bias, inatt_temperature
            )
        else:
            self.iatt = None

        # 4. MLP predictor
        self.predictor = MLP(
            backbone_outdims, nclasses, mlp_hiddens, mlp_act, mlp_bn, mlp_dp
        )

        # # ------ loss function ------
        if loss_func == "ce":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_func == "focal":
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        nb, c, w, h, _ = x.shape
        x = x.permute(0, 4, 1, 2, 3)  # (b,d,c,w,h)
        x = x.reshape(-1, c, w, h)  # (bd,c,w,h)
        x = self.backbone(x)[0]  # (bd,c',w',h')

        if self.satt is not None:
            sscore = self.satt(x)  # (bd,1,w',h')
            x = (x * sscore).sum(dim=(2, 3))  # (bd,c')
            sscore = (
                sscore.squeeze(1)
                .reshape(nb, -1, *sscore.shape[-2:])
                .permute(0, 2, 3, 1)
            )  # (b,w',h',d)
        else:
            x = x.mean(dim=(2, 3))
            sscore = None

        x = x.reshape(nb, -1, x.size(-1))  # (b,d,c')

        if self.iatt is not None:
            iscore = self.iatt(x)  # (b,d,1)
            xg = (x * iscore).sum(dim=1)  # (b,c')
            iscore = iscore.squeeze(-1)  # (b,d)
        else:
            xg = x.mean(dim=1)
            iscore = None

        pred = self.predictor(xg)  # (b,C)

        return pred, sscore, iscore

    def step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = self.forward(x)[0]
        loss = self.criterion(pred, y)

        return loss, pred

    def parameters(self):
        if self._bb_freeze:
            params = [self.predictor.parameters()]
            if self.satt is not None:
                params.append(self.satt.parameters())
            if self.iatt is not None:
                params.append(self.iatt.parameters())
            return chain.from_iterable(params)

        return super().parameters()
