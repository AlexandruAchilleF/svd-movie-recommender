import numpy as np
import pytest

from svd_recommender.linalg import frobenius_norm, qr_householder, svd, vector_norm
from svd_recommender.recommender import center_ratings

RNG = np.random.default_rng(0)


def test_norms():
    assert vector_norm(np.array([3.0, 4.0])) == pytest.approx(5.0)
    assert frobenius_norm(np.ones((2, 2))) == pytest.approx(2.0)


@pytest.mark.parametrize("shape", [(5, 5), (8, 4), (4, 8)])
def test_qr_householder(shape):
    a = RNG.standard_normal(shape)
    q, r = qr_householder(a)
    np.testing.assert_allclose(q @ r, a, atol=1e-10)
    np.testing.assert_allclose(q.T @ q, np.eye(shape[0]), atol=1e-10)
    np.testing.assert_allclose(np.tril(r, -1), 0, atol=1e-10)


def test_svd_matches_numpy():
    a = RNG.standard_normal((12, 6))
    u, sigma, vt = svd(a)
    np.testing.assert_allclose(np.diag(sigma), np.linalg.svd(a, compute_uv=False), atol=1e-8)
    np.testing.assert_allclose(u @ sigma @ vt, a, atol=1e-8)


def test_center_ratings_fills_nan():
    ratings = np.array([[5.0, np.nan, 3.0], [np.nan, 2.0, 4.0]])
    centered, means = center_ratings(ratings)
    np.testing.assert_allclose(means, [4.0, 3.0])
    assert not np.isnan(centered).any()
    np.testing.assert_allclose(centered.mean(axis=1), 0, atol=1e-12)
