import argparse
import logging
import os
import os.path as osp
import re
from typing import Dict

import pandas as pd

from src.utils import read_json, get_config_from_args, filter_runs


def read_test_score(fn: str) -> Dict:
    test_scores_i = read_json(fn)
    return test_scores_i


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
    parser.add_argument("--config", default=None, nargs="+", type=str)
    parser.add_argument("--just_summary", action="store_true")
    args = parser.parse_args()

    if args.run_dir is not None:
        run_dirs = [
            {"fdir": osp.join(args.root, args.run_dir), "dir": args.run_dir}
        ]
    else:
        run_dirs = filter_runs(args.root, args.start, args.end)

    all_test_scores = []
    for runi in run_dirs:
        fdir = runi["fdir"]
        if "fold0" not in os.listdir(fdir):
            test_scores_i = read_json(osp.join(fdir, "test_scores.json"))
            test_scores_i["run"] = runi["dir"]
            if args.config is not None:
                configs = read_json(osp.join(fdir, "args.json"))
                for k in args.config:
                    test_scores_i[k] = get_config_from_args(configs, k)
            all_test_scores.append(test_scores_i)
            continue

        for subdir in os.listdir(fdir):
            match_res = re.search(r"fold(\d+?)", subdir)
            if match_res:
                test_scores_i = read_json(
                    osp.join(fdir, subdir, "test_scores.json")
                )
                test_scores_i["run"] = runi["dir"]
                test_scores_i["fold"] = int(match_res.group(1))
                if args.config is not None:
                    configs = read_json(osp.join(fdir, subdir, "args.json"))
                    for k in args.config:
                        test_scores_i[k] = get_config_from_args(configs, k)
                all_test_scores.append(test_scores_i)

    all_test_scores = pd.DataFrame.from_records(all_test_scores)

    if args.just_summary:
        index_names = ["run"]
        if args.config is not None:
            index_names += args.config
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
            run_id = df.columns.get_loc("run")
            df_metric = df.iloc[:, :run_id]
            df_summ = pd.DataFrame(
                {"mean": df_metric.mean(axis=0), "std": df_metric.std(axis=0)}
            ).T.reset_index(names="fold")
            df = pd.concat([df, df_summ], axis=0)
            df = df.ffill().reset_index(drop=True)
        return df

    all_test_scores = all_test_scores.groupby("run").apply(_sort_summary)
    index_names = ["run"]
    if args.config is not None:
        index_names += args.config
    if "fold" in all_test_scores.columns:
        index_names.append("fold")
    all_test_scores = all_test_scores.set_index(index_names)
    print(all_test_scores)


if __name__ == "__main__":
    main()
