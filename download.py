"""
Download data and store a subset of it used for further analysis.
"""
import pandas as pd
from datasets import load_dataset
import os
import argparse


def main(args):
    dataset = load_dataset("Fraser/news-category-dataset", split="train")
    dataset = dataset.shuffle(seed=args.seed)[: args.n]
    df = pd.DataFrame(dataset)
    df = df.drop(["category_num", "category"], axis=1)
    os.makedirs('data', exist_ok=True)
    df.to_csv("data/news.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    main(args)
