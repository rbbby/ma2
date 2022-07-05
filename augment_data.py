"""
Impute missing short descriptions by promtping GPT2 with the headlines
"""
import pandas as pd
from transformers import pipeline, set_seed
from tqdm import tqdm
import argparse


def main(args):
    set_seed(args.seed)
    generator = pipeline("text-generation", model="gpt2")
    df = pd.read_csv("data/news.csv")
    df_missing = df[df["short_description"].isna()]

    for i, row in tqdm(df_missing.iterrows(), total=len(df_missing)):
        headline = row["headline"]
        # Arguments set to stochastically generate short texts sticking to the topic
        gen = generator(headline, max_length=50, do_sample=True, top_k=50)
        # Remove headline from generated text
        short_description = gen[0]["generated_text"][len(headline) :].lstrip()
        df.loc[i, "short_description"] = short_description
    df.to_csv("data/news_imputed.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    main(args)
