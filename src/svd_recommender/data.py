"""Downloading and loading the MovieLens 100k dataset."""

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DEFAULT_DATA_DIR = Path("data")


def download_movielens(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download and extract MovieLens 100k into ``data_dir`` if missing.

    Returns:
        Path to the extracted ``ml-100k`` folder.
    """
    dataset_dir = data_dir / "ml-100k"
    if dataset_dir.exists():
        return dataset_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-100k.zip"
    print(f"Downloading MovieLens 100k from {MOVIELENS_URL} ...")
    urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)
    zip_path.unlink()
    return dataset_dir


def load_movielens(
    max_users: int = 100,
    max_movies: int = 130,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Load a subset of MovieLens 100k as a dense user x movie rating matrix.

    Only users with ``user_id <= max_users`` and movies with
    ``movie_id <= max_movies`` are kept, which keeps the from-scratch SVD fast.

    Returns:
        ``(ratings, user_ids, titles)`` where ``ratings`` has shape
        ``(n_users, n_movies)`` with ``NaN`` for unrated movies, ``user_ids[i]``
        is the MovieLens id of row ``i`` and ``titles[j]`` is the title of
        column ``j``.
    """
    dataset_dir = download_movielens(data_dir)

    ratings = pd.read_csv(
        dataset_dir / "u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    items = pd.read_csv(
        dataset_dir / "u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["movie_id", "title"],
    )

    ratings = ratings[(ratings["user_id"] <= max_users) & (ratings["movie_id"] <= max_movies)]
    matrix = ratings.pivot_table(index="user_id", columns="movie_id", values="rating")

    title_by_id = dict(zip(items["movie_id"], items["title"]))
    titles = [title_by_id.get(movie_id, f"Movie {movie_id}") for movie_id in matrix.columns]

    return matrix.to_numpy(dtype=float), list(matrix.index), titles
