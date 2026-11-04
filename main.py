"""Run the full pipeline: load data, validate SVD, recommend movies, save plots."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402

from svd_recommender import SVDRecommender, load_movielens, vector_norm  # noqa: E402
from svd_recommender.plotting import plot_rmse_vs_rank, plot_singular_values  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", type=int, default=2, help="MovieLens user id (default: 2)")
    parser.add_argument("--k", type=int, default=3, help="latent features used for prediction")
    parser.add_argument("--top-n", type=int, default=3, help="number of recommendations")
    parser.add_argument("--max-users", type=int, default=100)
    parser.add_argument("--max-movies", type=int, default=130)
    parser.add_argument("--plot-max-k", type=int, default=15, help="largest k in the RMSE plot")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/images"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading MovieLens 100k...")
    ratings, user_ids, titles = load_movielens(args.max_users, args.max_movies)
    print(f"Rating matrix: {ratings.shape[0]} users x {ratings.shape[1]} movies")

    if args.user not in user_ids:
        sys.exit(f"User {args.user} is not in the loaded subset (ids 1..{max(user_ids)}).")
    user_index = user_ids.index(args.user)

    # Full decomposition so the singular values can be compared with NumPy.
    model = SVDRecommender(ratings, titles, max_rank=min(ratings.shape))

    reference = np.linalg.svd(model.centered, compute_uv=False)
    error = vector_norm(model.singular_values - reference)
    print(f"L2 error of singular values vs numpy.linalg.svd: {error:.4e}\n")

    print(f"Top {args.top_n} recommendations for user {args.user} (k={args.k}):")
    for title, score in model.recommend(user_index, args.k, args.top_n):
        print(f"  -> {title}  (predicted rating: {score:.2f} / 5)")

    print(f"\nMovies already rated by user {args.user}:")
    for title, rating in model.history(user_index):
        print(f"  -> {title}  (rating: {rating:.0f})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_singular_values(model, args.output_dir / "singular_values.png")
    plot_rmse_vs_rank(model, args.plot_max_k, args.output_dir / "rmse_vs_k.png")
    print(f"\nPlots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
