import argparse
import logging
import math
import os
import os.path as osp
import re
from datetime import datetime
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as ssp
import torch
from monai.transforms import Resize
from tqdm import tqdm

# from src.dataset import dfs2loaders
# from src.model import CNN2dATT
# from src.train import pred_model
# from src.utils import read_json


def main():
    logging.basicConfig(level=logging.INFO)

    # 0. argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/mnt/data1/tiantan/results/", type=str
    )
    parser.add_argument("--run_dir", default=None, type=str)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    # 1.
    if args.run_dir is None:
        runs = os.listdir(args.root)
        runs_df = [
            {
                "date": datetime.strptime(runi, "%Y-%m-%d_%H-%M-%S"),
                "path": runi,
            }
            for runi in runs
            if re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", runi)
        ]
        runs_df = pd.DataFrame.from_records(runs_df).sort_values(
            "date", ascending=False
        )
        last_run = runs_df["path"].iloc[0]
        logging.info("select the results of last run: %s" % last_run)
        run_dir = osp.join(args.root, last_run)
    else:
        run_dir = osp.join(args.root, args.run_dir)


if __name__ == "__main__":
    main()
