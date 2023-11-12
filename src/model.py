from typing import List, Literal, Optional, Tuple
from itertools import chain

import numpy as np
import timm
import torch
import torch.nn as nn
from monai.networks.layers import Act


def get_resnet_bone(
    name: str = "resnet50",
    pretrained: bool = True,
    in_dims: int = 1,
) -> nn.Module:
    model = timm.create_model(
        name,
        pretrained=pretrained,
        in_chans=in_dims,
        features_only=True,
        out_indices=[4] if name in ["resnet18"] else [3],
    )
    out_dims = 512 if name == "resnet18" else 1024
    return model, out_dims


class SpatialAttention(nn.Module):
    def __init__(
        self,
        in_channel: int,
        hiddens: List[int],
        activations: List[str],
        bn: bool = False,
        dp: Optional[float] = None,
    ) -> None:
        super().__init__()
        layers = []
        for i, o, ai in zip([in_channel] + hiddens[:-1], hiddens, activations):
            layers.append(nn.Conv2d(i, o, 1))
            if bn:
                layers.append(nn.BatchNorm2d(o))
            layers.append(Act[ai]())
            if dp is not None:
                layers.append(nn.Dropout2d(dp))
        layers.append(nn.Conv2d(hiddens[-1], 1, 1))
        self._net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sscore = self._net(x)  # (bz,1,x,y)
        sscore = torch.flatten(sscore, 2, -1)  # (bz,1,xy)
        sscore = torch.softmax(sscore, dim=-1)
        sscore = sscore.reshape(x.size(0), 1, *x.shape[-2:])  # (bz,1,x,y)
        return sscore


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


class MLP(nn.Module):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        hiddens: List[int],
        act: str,
        bn: bool,
        dp: Optional[float] = None,
    ) -> None:
        super().__init__()
        layers = []
        if len(hiddens) > 0:
            for inpc, outc in zip([in_dims] + hiddens[:-1], hiddens):
                layers.append(nn.Linear(inpc, outc))
                if bn:
                    layers.append(nn.BatchNorm1d(outc))
                layers.append(Act[act]())
                if dp is not None:
                    layers.append(nn.Dropout(dp))
            layers.append(nn.Linear(hiddens[-1], out_dims))
        else:
            layers.append(nn.Linear(in_dims, out_dims))
        self._net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


class FocalLoss(nn.Module):
    def __init__(
        self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        if reduction not in ["mean", "none", "sum"]:
            raise NotImplementedError(
                "Reduction {} not implemented.".format(reduction)
            )
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = np.finfo(float).eps
        x = torch.softmax(x, dim=1)
        p_t = x[:, 1] * (target == 1) + x[:, 0] * (target == 0)
        alpha = self.alpha * (target == 1) + (1 - self.alpha) * (target == 0)
        # p_t = torch.where(target == 1, x, 1-x)
        fl = -1 * (1 - p_t) ** self.gamma * torch.log(p_t + eps)
        fl *= alpha
        return self._reduce(fl)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return x.mean()
        elif self.reduction == "sum":
            return x.sum()
        else:
            return


class CNN2dATT(nn.Module):
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
            # self.backbone, nhidden, _ = get_conv_bone(bb_args.cnn_kwargs,
            #                                           n_spatials=2)

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
        self._nb = x.size(0)
        x = x.permute(0, 4, 1, 2, 3)  # (b,z,c,x,y)
        x = x.reshape(-1, *x.shape[-3:])  # (bz,c,x,y)
        x = self.backbone(x)[0]

        if self.satt is not None:
            sscore = self.satt(x)  # (bz,1,x,y)
            x = (x * sscore).sum(dim=(2, 3))  # (bz,c)
        else:
            x = x.sum(dim=(2, 3))
            sscore = None

        x = x.reshape(self._nb, -1, x.size(1))  # (b,z,c)

        if self.iatt is not None:
            iscore = self.iatt(x)  # (b,z,1)
            xg = (x * iscore).sum(dim=1)  # (b,c)
        else:
            xg = x.sum(dim=1)
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
