import random
import json
from typing import Dict

import numpy as np
import torch


def read_json(fn: str) -> Dict:
    with open(fn, "r") as f:
        res = json.load(f)
    return res


def save_json(obj: Dict, fn: str) -> None:
    with open(fn, "w") as f:
        json.dump(obj, f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
