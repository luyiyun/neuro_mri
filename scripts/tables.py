import os.path as osp
import re
import sys
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.append("/".join(osp.abspath(__file__).split("/")[:-2]))
from src.utils import read_json


def extract_MS_case_metas(ffns: Sequence) -> pd.DataFrame:
    # just for MS cases
    pmetas = []
    for ffn in ffns:
        fn = osp.basename(ffn)[:-4]
        match = re.search(r"^(TT\d{4})_([A-Z]*?)-(\d{8})_.*?$", fn)
        if match:
            pid, name, date = match.group(1, 2, 3)
            pmetas.append({"ffn": ffn, "pid": pid, "name": name, "date": date})
    pmetas = pd.DataFrame.from_records(pmetas)
    return pmetas


def main():
    root = "/mnt/data1/tiantan/results"
    df = pd.read_csv("/mnt/data1/tiantan/fn_rm_skull_mv.csv", index_col=0)
    df = df.loc[df.v1_filter2, :]

    # 1. value_counts for labels
    print(df.label.value_counts())

    # 2. slices counting
    df["num_slices"] = (
        df["shape"].str.split("x", expand=True).iloc[:, -1].map(int)
    )
    print(df.groupby("label")["num_slices"].sum())

    # 3. test scores of the comparison methods
    run_dirs = {
        "DualAttCNN (Focal Loss, Proposed)": "2023-11-19_21-00-03",
        "DualAttCNN (Cross Entropy)": "2023-11-19_18-06-19",
        "3D-CNN (Focal Loss)": "2023-11-20_15-36-37",
        "3D-CNN (Cross Entropy)": "2023-11-21_01-14-01",
        "2D-CNN (Focal Loss)": "2023-11-19_16-30-42",
        "2D-CNN (Cross Entropy)": "2023-11-21_01-41-11",
        "SIFT & HOG + SVM": "2023-11-20_23-22-28",
        "SIFT & HOG + RF": "2023-11-21_00-26-24",
    }
    test_scores = []
    for k, run_dir in run_dirs.items():
        for i in range(5):
            fn = osp.join(root, run_dir, "fold%d" % i, "test_scores.json")
            test_scores_i = read_json(fn)
            test_scores_i["fold"] = i
            test_scores_i["method"] = k
            test_scores.append(test_scores_i)
    test_scores = pd.DataFrame.from_records(test_scores)
    metric_mapping = {
        "bacc": "Balanced Accuracy",
        "acc": "Accuracy",
        "auc": "Area Under ROC Curver",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
    }
    test_scores.rename(columns=metric_mapping, inplace=True)
    test_scores_summary = test_scores.groupby("method")[
        list(metric_mapping.values())
    ].agg(lambda x: "%.4f±%.4f" % (np.mean(x), np.std(x)))
    test_scores_summary.to_csv(
        osp.join(root, "table_comparison.csv"), encoding="cp936"  # for windows
    )
    print(test_scores_summary)


if __name__ == "__main__":
    main()
