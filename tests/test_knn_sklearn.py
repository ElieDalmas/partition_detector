from neighbors import KNeighborsClassifier
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN


class TestKNNvsSklearn:

    def test_predict_match_k1(self, train_data, test_data):
        X, y = train_data
        sk = SklearnKNN(n_neighbors=1).fit(X, y)
        my = KNeighborsClassifier(n_neighbors=1).fit(X, y)
        assert my.predict(test_data) == list(sk.predict(test_data))

    @pytest.mark.parametrize("k", [3, 5, 8])
    def test_predict_match_no_tie_points(self, train_data, test_data, k):
        def _has_tie(proba_row) -> bool:
            """True if multiple winner (diff with sklearn)"""
            max_val = max(proba_row)
            return list(proba_row).count(max_val) > 1

        X, y = train_data
        sk = SklearnKNN(n_neighbors=k).fit(X, y)
        my = KNeighborsClassifier(n_neighbors=k).fit(X, y)

        proba = my.predict_proba(test_data)
        pred_my = my.predict(test_data)
        pred_sk = list(sk.predict(test_data))

        for i in range(len(test_data)):
            if not _has_tie(proba[i]):
                assert pred_my[i] == pred_sk[i], (
                    f"Point {i} no tie : my={pred_my[i]} sk={pred_sk[i]}"
                )

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_predict_proba_match(self, train_data, test_data, k):
        X, y = train_data
        sk = SklearnKNN(n_neighbors=k).fit(X, y)
        my = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        assert sk.classes_.tolist() == my.classes_
        assert np.allclose(sk.predict_proba(test_data), my.predict_proba(test_data))

    def test_classes_order_matches(self, train_data):
        X, y = train_data
        sk = SklearnKNN(n_neighbors=3).fit(X, y)
        my = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        assert sk.classes_.tolist() == my.classes_