"""Collaborative-filtering recommender based on a low-rank SVD approximation."""

import numpy as np

from svd_recommender.linalg import svd

MIN_RATING = 1.0
MAX_RATING = 5.0


def center_ratings(ratings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill missing ratings with each user's mean and subtract that mean.

    Returns:
        ``(centered, user_means)``; ``centered`` has no ``NaN`` values.
    """
    user_means = np.nanmean(ratings, axis=1)
    filled = np.where(np.isnan(ratings), user_means[:, None], ratings)
    return filled - user_means[:, None], user_means


def rmse(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> float:
    """Root mean squared error computed only over entries where ``mask`` is true."""
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))


class SVDRecommender:
    """Predicts ratings with a rank-``k`` SVD of the mean-centered rating matrix."""

    def __init__(self, ratings: np.ndarray, titles: list[str], max_rank: int):
        self.ratings = np.asarray(ratings, dtype=float)
        self.titles = titles
        self.observed = ~np.isnan(self.ratings)
        self.centered, self.user_means = center_ratings(self.ratings)
        # Factorize once; any rank k <= max_rank is a cheap slice of these factors.
        self.u, self.sigma, self.vt = svd(self.centered, k=max_rank)

    @property
    def singular_values(self) -> np.ndarray:
        return np.diag(self.sigma)

    def predict(self, k: int) -> np.ndarray:
        """Return the full predicted rating matrix using the top ``k`` components."""
        approx = self.u[:, :k] @ self.sigma[:k, :k] @ self.vt[:k, :]
        return np.clip(approx + self.user_means[:, None], MIN_RATING, MAX_RATING)

    def training_rmse(self, k: int) -> float:
        """RMSE of the rank-``k`` reconstruction on the observed ratings."""
        return rmse(self.ratings, self.predict(k), self.observed)

    def recommend(self, user_index: int, k: int, top_n: int = 3) -> list[tuple[str, float]]:
        """Return the ``top_n`` unseen movies with the highest predicted rating."""
        scores = self.predict(k)[user_index].copy()
        scores[self.observed[user_index]] = -np.inf
        best = np.argsort(scores)[::-1][:top_n]
        return [(self.titles[j], float(scores[j])) for j in best if np.isfinite(scores[j])]

    def history(self, user_index: int) -> list[tuple[str, float]]:
        """Return the movies a user has rated, sorted from highest to lowest rating."""
        row = self.ratings[user_index]
        rated = np.flatnonzero(~np.isnan(row))
        return sorted(((self.titles[j], float(row[j])) for j in rated), key=lambda x: -x[1])
