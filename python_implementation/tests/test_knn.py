from neighbors import KNeighborsClassifier
import numpy as np
import pytest

class TestKNN:

    def test_fit_classes_sorted(self, train_data):
        X, y = train_data
        knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        assert knn.classes_ == sorted(set(y))

    def test_predict_len(self, train_data, test_data):
        X, y = train_data
        knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        assert len(knn.predict(test_data)) == len(test_data)

    def test_predict_proba_shape(self, train_data, test_data):
        X, y = train_data
        knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        proba = knn.predict_proba(test_data)
        assert proba.shape == (len(test_data), len(knn.classes_))

    def test_predict_proba_eq1(self, train_data, test_data):
        X, y = train_data
        knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        proba = knn.predict_proba(test_data)
        for i in range(len(test_data)):
            assert abs(sum(proba[i]) - 1.0) < 1e-9

    def test_predict_proba_0_neg(self, train_data, test_data):
        X, y = train_data
        knn = KNeighborsClassifier(n_neighbors=5).fit(X, y)
        proba = knn.predict_proba(test_data)
        assert (proba >= 0).all()

    def test_k1_predict_eq_nearest(self, train_data):
        X, y = train_data
        knn = KNeighborsClassifier(n_neighbors=1).fit(X, y)
        preds = knn.predict(X)
        assert preds == list(y)

    def test_euclidean_distance(self):
        knn = KNeighborsClassifier()
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert abs(knn._euclidean_distance(a, b) - 5.0) < 1e-10

    def test_euclidean_distance_0(self):
        knn = KNeighborsClassifier()
        a = np.array([1.0, 2.0, 3.0])
        assert knn._euclidean_distance(a, a) == 0.0

    def test_majority_vote_win(self):
        knn = KNeighborsClassifier()
        assert knn._majority_vote(["a", "a", "b"]) == "a"

    def test_not_fitted_error(self, test_data):
        knn = KNeighborsClassifier(n_neighbors=3)
        with pytest.raises(RuntimeError):
            knn.predict(test_data)
