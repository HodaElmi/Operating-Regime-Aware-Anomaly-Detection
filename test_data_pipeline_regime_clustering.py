import unittest

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from data_pipeline import _fit_regime_model, _fit_regime_model_healthy_only, _predict_regimes


def _three_blobs(seed=0):
    rng = np.random.RandomState(seed)
    blob0 = rng.normal(loc=[0, 0], scale=0.5, size=(15, 2))
    blob1 = rng.normal(loc=[20, 20], scale=0.5, size=(15, 2))
    blob2 = rng.normal(loc=[40, 40], scale=0.5, size=(15, 2))
    return np.vstack([blob0, blob1, blob2])


class TestFitRegimeModel(unittest.TestCase):
    def test_kmeans_default_matches_explicit_algorithm_kmeans(self):
        X = _three_blobs()
        scaler_a, model_a, feat_idx_a, knn_a = _fit_regime_model(X, n_regimes=3, regime_feature_idx=[0, 1])
        scaler_b, model_b, feat_idx_b, knn_b = _fit_regime_model(
            X, n_regimes=3, regime_feature_idx=[0, 1], algorithm = "kmeans"
        )
        np.testing.assert_array_equal(model_a.labels_, model_b.labels_)
        np.testing.assert_array_almost_equal(model_a.cluster_centers_, model_b.cluster_centers_)
        self.assertIsNone(knn_a)
        self.assertIsNone(knn_b)

    def test_hdbscan_returns_fitted_knn_companion(self):
        X = _three_blobs()
        scaler, model, feat_idx, knn = _fit_regime_model(
            X, regime_feature_idx=[0, 1], algorithm="hdbscan", hdbscan_min_cluster_size=5
        )
        self.assertIsNotNone(knn)
        Z = scaler.transform(X[:, feat_idx])
        non_noise = model.labels_ != -1
        self.assertGreater(non_noise.sum(), 0)
        predicted = knn.predict(Z[non_noise])
        np.testing.assert_array_equal(predicted, model.labels_[non_noise])

    def test_hdbscan_requires_min_cluster_size(self):
        X = _three_blobs()
        with self.assertRaises(ValueError):
            _fit_regime_model(X, regime_feature_idx=[0, 1], algorithm="hdbscan")

    def test_unknown_algorithm_raises(self):
        X = _three_blobs()
        with self.assertRaises(ValueError):
            _fit_regime_model(X, regime_feature_idx=[0, 1], algorithm="not_a_real_algorithm")


class TestFitRegimeModelHealthyOnly(unittest.TestCase):
    def test_threads_algorithm_through(self):
        X = _three_blobs()
        y = np.zeros(len(X), dtype=int)  # all "healthy"
        scaler, model, feat_idx, knn = _fit_regime_model_healthy_only(
            X, y, n_regimes=3, regime_feature_idx=[0, 1],
            algorithm="hdbscan", hdbscan_min_cluster_size=5,
        )
        self.assertIsNotNone(knn)


class TestPredictRegimes(unittest.TestCase):
    def test_defaults_to_model_predict_when_no_knn_given(self):
        X = _three_blobs()
        scaler, model, feat_idx, _ = _fit_regime_model(X, n_regimes=3, regime_feature_idx=[0, 1])

        old_style = _predict_regimes(X, scaler, model, feat_idx)
        new_style = _predict_regimes(X, scaler, model, feat_idx, regime_knn=None)
        np.testing.assert_array_equal(old_style, new_style)
        np.testing.assert_array_equal(old_style, model.predict(scaler.transform(X[:, feat_idx])))

    def test_dispatches_to_knn_when_provided(self):
        X = _three_blobs()
        scaler, model, feat_idx, _ = _fit_regime_model(X, n_regimes=3, regime_feature_idx=[0, 1])

        class _AlwaysZero:
            def predict(self, Z):
                return np.zeros(len(Z), dtype=int)

        Z = scaler.transform(X[:, feat_idx])
        fake_knn = KNeighborsClassifier(n_neighbors=1)
        fake_knn.fit(Z, model.labels_)  # real, non-trivial labels

        result = _predict_regimes(X, scaler, _AlwaysZero(), feat_idx, regime_knn=fake_knn)
        np.testing.assert_array_equal(result, fake_knn.predict(Z))
        self.assertFalse(np.all(result == 0))


if __name__ == "__main__":
    unittest.main()
