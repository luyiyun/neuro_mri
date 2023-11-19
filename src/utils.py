import json
import os
import os.path as osp
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


def read_json(fn: str) -> Dict:
    with open(fn, "r") as f:
        res = json.load(f)
    return res


def save_json(obj: Dict, fn: str) -> None:
    with open(fn, "w") as f:
        json.dump(obj, f)


def set_seed(seed: int, deterministic_algorithm: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic_algorithm:
        torch.use_deterministic_algorithms(True)


def get_config_from_args(args: Dict, key: str) -> Any:
    value = args.get(key, "")
    if isinstance(value, (list, tuple)):
        value = ",".join(
            "%.3f" % vi if isinstance(vi, float) else str(vi) for vi in value
        )
    return value


def parse_config(conf: Optional[str]) -> Union[str, Tuple[str, str]]:
    if conf is None:
        return None
    return tuple(conf.split("=")) if "=" in conf else conf


def filter_runs_by_datetime(
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


def filter_runs_by_configs(run_dirs: List[Dict], configs: List) -> List[Dict]:
    # separate configs into w_value and wo_value
    conf_w_v, conf_wo_v = [], []
    for k in configs:
        if isinstance(k, tuple):
            conf_w_v.append(k)
        elif isinstance(k, str):
            conf_wo_v.append(k)
        else:
            raise ValueError

    # select the suitable runs
    new_run_dirs = []
    for runi in run_dirs:
        fdir = runi["fdir"]
        if "fold0" not in os.listdir(fdir):
            config_fn = osp.join(fdir, "args.json")
        else:
            # 默认如果是交叉验证一定存在fold0
            config_fn = osp.join(fdir, "fold0", "args.json")
        configs = read_json(config_fn)

        flag_reject = False
        for k, v in conf_w_v:
            arg_in_f = get_config_from_args(configs, k)
            if str(arg_in_f) != v:
                flag_reject = True
                break
            else:
                runi[k] = arg_in_f
        if flag_reject:
            continue

        for k in conf_wo_v:
            runi[k] = get_config_from_args(configs, k)
        new_run_dirs.append(runi)

    return new_run_dirs
