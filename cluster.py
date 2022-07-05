"""
Main modeling script that:
- Generates embeddings using SBERT
- Reduces embedding dimensionality using UMAP'
- Performs KMeans clustering on the low dimensional embedding representations
"""
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import umap
from sentence_transformers import SentenceTransformer
import argparse


def main(args):
    df = pd.read_csv("data/news_imputed.csv")

    # Join headline and short description prior to embedding
    df["text"] = df[["headline", "short_description"]].apply(" ".join, 1)

    # Use SBERT to produce embeddings
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embeddings = model.encode(df["text"], show_progress_bar=True)

    # Reduce dimensionality of embeddings
    u = umap.UMAP(
        random_state=args.seed,
        min_dist=0.0,
        n_neighbors=25,
        n_components=20,
        metric="cosine",
    ).fit_transform(embeddings)

    # Cluster low dimensional features
    preds = KMeans(n_clusters=40, random_state=args.seed).fit_predict(u)

    # Optional 3d plot of clusters in embedding space
    if args.plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=preds)
        plt.show()

    # Store results
    df["prediction"] = preds
    df.to_csv("data/news_result.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    main(args)
