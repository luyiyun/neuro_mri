import argparse
import logging
import os
import os.path as osp
import re
import sys
from typing import Dict, List

import pandas as pd

sys.path.append("/".join(osp.abspath(__file__).split("/")[:-2]))
from src.utils import (filter_runs_by_configs, filter_runs_by_datetime,
                       parse_config, read_json)


def read_test_score(fn: str) -> Dict:
    test_scores_i = read_json(fn)
    return test_scores_i


def load_test_scores_by_run_dirs(run_dirs: List[Dict]) -> pd.DataFrame:
    all_test_scores = []
    for runi in run_dirs:
        # get test score filenames
        fdir = runi["fdir"]
        if "fold0" not in os.listdir(fdir):
            test_scores_fns = [(None, osp.join(fdir, "test_scores.json"))]
        else:
            test_scores_fns = []
            for subdir in os.listdir(fdir):
                match_res = re.search(r"fold(\d+?)", subdir)
                if match_res:
                    test_scores_fns.append(
                        (
                            int(match_res.group(1)),
                            osp.join(fdir, subdir, "test_scores.json"),
                        )
                    )

        # load the test scores and bind the corresponding configs
        for foldi, test_score_fn in test_scores_fns:
            test_scores_i = read_json(test_score_fn)
            test_scores_i["fold"] = foldi
            for k, v in runi.items():
                if k == "fdir":
                    continue
                elif k == "dir":
                    test_scores_i["run"] = v
                else:
                    test_scores_i[k] = v

            all_test_scores.append(test_scores_i)

    return pd.DataFrame.from_records(all_test_scores)


def main():
    logging.basicConfig(level=logging.INFO)

    # 0. argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/mnt/data1/tiantan/results/", type=str
    )
    parser.add_argument("--start", default="2023-11-14", type=str)
    parser.add_argument("--end", default=None, type=str)
    parser.add_argument("--run_dir", default=None, type=str)
    parser.add_argument("--config", default=None, nargs="+", type=parse_config)
    parser.add_argument("--just_summary", action="store_true")
    args = parser.parse_args()

    # 1. Filter runs by time point
    if args.run_dir is not None:
        run_dirs = [
            {"fdir": osp.join(args.root, args.run_dir), "dir": args.run_dir}
        ]
    else:
        run_dirs = filter_runs_by_datetime(args.root, args.start, args.end)

    # 2. Filter runs by config
    if args.config is not None:
        run_dirs = filter_runs_by_configs(run_dirs, args.config)

    # 3. Read test_scores
    all_test_scores = load_test_scores_by_run_dirs(run_dirs)

    # 4. summarize the test_scores as a table to show in cmdline
    if args.just_summary:
        index_names = ["run"]
        if args.config is not None:
            for confi in args.config:
                index_names.append(
                    confi[0] if isinstance(confi, tuple) else confi
                )
        all_test_scores = (
            all_test_scores.drop(columns=["fold"])
            .groupby(index_names, dropna=False)
            .agg(lambda x: "%.4f±%.4f" % (x.mean(), x.std()))
        )
        print(all_test_scores)
        return

    def _sort_summary(df):
        if "fold" in df.columns:
            df = df.sort_values("fold")
            metric_cols = df.columns.isin(
                [
                    "main",
                    "kl",
                    "bacc",
                    "acc",
                    "auc",
                    "sensitivity",
                    "specificity",
                ]
            )
            df_metric = df.loc[:, metric_cols]
            df_summ = pd.DataFrame(
                {"mean": df_metric.mean(axis=0), "std": df_metric.std(axis=0)}
            ).T.reset_index(names="fold")
            df = pd.concat([df, df_summ], axis=0)
            df = df.ffill().reset_index(drop=True)
        return df

    all_test_scores = all_test_scores.groupby("run").apply(_sort_summary)
    index_names = ["run"]
    if args.config is not None:
        for confi in args.config:
            index_names.append(confi[0] if isinstance(confi, tuple) else confi)
    if "fold" in all_test_scores.columns:
        index_names.append("fold")
    all_test_scores = all_test_scores.set_index(index_names)

    # with pd.option_context(
    #     "display.max_rows", None, "display.max_columns", None
    # ):  # more options can be specified also
    #     print(all_test_scores)
    print(all_test_scores)


if __name__ == "__main__":
    main()
