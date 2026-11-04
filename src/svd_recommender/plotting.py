"""Evaluation plots for the SVD recommender."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Render to files; no display needed.
import matplotlib.pyplot as plt  # noqa: E402

from svd_recommender.recommender import SVDRecommender  # noqa: E402


def plot_singular_values(model: SVDRecommender, path: Path) -> None:
    """Save a plot of the singular values in descending order."""
    values = model.singular_values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(values) + 1), values, marker="o", color="tab:blue")
    ax.set_title("Singular value decay")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rmse_vs_rank(model: SVDRecommender, max_k: int, path: Path) -> None:
    """Save a plot of training RMSE for every rank ``k`` from 1 to ``max_k``."""
    ranks = range(1, max_k + 1)
    errors = [model.training_rmse(k) for k in ranks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ranks, errors, marker="s", color="tab:red")
    ax.set_title("Reconstruction error (RMSE) vs number of latent features (k)")
    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("RMSE")
    ax.set_xticks(list(ranks))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
