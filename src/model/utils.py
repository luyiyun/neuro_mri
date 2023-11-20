from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from monai.networks.layers import Act


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
