"""Movie recommender built on a from-scratch SVD (Householder QR + QR iteration)."""

from svd_recommender.data import load_movielens
from svd_recommender.linalg import frobenius_norm, qr_householder, svd, vector_norm
from svd_recommender.recommender import SVDRecommender

__all__ = [
    "SVDRecommender",
    "frobenius_norm",
    "load_movielens",
    "qr_householder",
    "svd",
    "vector_norm",
]
