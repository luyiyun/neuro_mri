import os
import os.path as osp
import re
import json
import random
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

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


def get_config_from_args(args: Dict, key: str) -> Any:
    value = args.get(key, "")
    if isinstance(value, (list, tuple)):
        value = ",".join(
            "%.3f" % vi if isinstance(vi, float) else str(vi) for vi in value
        )
    return value


def filter_runs(
    root: str, start: Optional[str] = None, end: Optional[str] = None
) -> Sequence[Dict]:
    run_dirs = [
        {"fdir": osp.join(root, di), "dir": di} for di in os.listdir(root)
    ]
    run_dirs = filter(lambda x: osp.isdir(x["fdir"]), run_dirs)
    if start is not None or end is not None:
        # 只选择那些是时间命名的路径
        run_dirs = filter(
            lambda x: re.search(
                r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", x["dir"]
            ),
            run_dirs,
        )
    if start is not None:
        start_time = datetime.fromisoformat(start)
        run_dirs = filter(
            lambda x: datetime.strptime(x["dir"], "%Y-%m-%d_%H-%M-%S")
            >= start_time,
            run_dirs,
        )
    if end is not None:
        end_time = datetime.fromisoformat(end)
        run_dirs = filter(
            lambda x: datetime.strptime(x["dir"], "%Y-%m-%d_%H-%M-%S")
            <= end_time,
            run_dirs,
        )
    run_dirs = sorted(run_dirs, key=lambda x: x["dir"])
    return run_dirs
