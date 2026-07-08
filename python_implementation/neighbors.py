import math
import numpy as np


class KNeighborsClassifier:
    """
    Classifier implementing the k-nearest neighbors vote.
    Very closed to sklearn.neighbors.KNeighborsClassifier

    default:
    - weights: uniform (majority vote)
    - metric: Euclidean

    Usage :
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        preds  = knn.predict(X_test)
        probas = knn.predict_proba(X_test)
    """

    def __init__(self, n_neighbors: int = 5):
        """
        :param n_neighbors: Number of neighbors
        """
        if n_neighbors < 1:
            raise ValueError("n_neighbors doit être >= 1")
        self.n_neighbors = n_neighbors
        self._X_train: np.ndarray | None = None
        self._y_train: list | None = None
        self.classes_: list | None = None

    def fit(self, X, y) -> "KNeighborsClassifier":
        """
        Fit the k-nearest neighbors classifier from the training dataset.
        Supervised => 0 centroids, only comparing with training dataset.

        :param X: Training data.
        :param y: Target values.
        :return: The fitted k-nearest neighbors classifier.
        """
        self._X_train = np.array(X, dtype=float)
        self._y_train = list(y)

        unique_labels = []
        for label in self._y_train:
            if label not in unique_labels:
                unique_labels.append(label)

        # sklearn sort classes, so be it
        self.classes_ = sorted(unique_labels)
        return self

    # region Predictions

    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance between two vectors."""
        n_features = len(a)
        s = 0.0
        for f in range(n_features):
            diff = a[f] - b[f]
            s += diff * diff
        return math.sqrt(s)

    def _k_nearest_labels(self, query: np.ndarray) -> list:
        """
         :return: k nearest neighbours labels of query.
        """
        k = self.n_neighbors
        n_train = len(self._X_train)

        # List[(distance, label)], max dist: +inf
        best: list[tuple[float, object]] = [(math.inf, None)] * k

        for i in range(n_train):
            d = self._euclidean_distance(query, self._X_train[i])

            # sort d if lower than worst
            if d < best[-1][0]:
                best[-1] = (d, self._y_train[i])

                # partial insertion sort
                j = k - 1
                while j > 0 and best[j][0] < best[j - 1][0]:
                    best[j], best[j - 1] = best[j - 1], best[j]
                    j -= 1

        return [label for _, label in best]

    def _majority_vote(self, labels: list) -> object:
        """Return the most frequent label."""
        counts: dict = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

        best_label = None
        best_count = -1
        for label, count in counts.items():
            if count > best_count:
                best_count = count
                best_label = label
        return best_label

    def predict(self, X) -> list:
        """
        Predict the class labels for the provided data.

        :param X: Test samples
        :return: Class labels for each data sample.
        """
        self._check_is_fitted()
        X = np.array(X, dtype=float)

        predictions = []
        for i in range(len(X)):
            neighbor_labels = self._k_nearest_labels(X[i])
            predictions.append(self._majority_vote(neighbor_labels))
        return predictions

    def predict_proba(self, X) -> np.ndarray:
        """
        Return probability estimates for the test data X.

        :return: array[n_samples, n_classes],
        """
        self._check_is_fitted()
        X = np.array(X, dtype=float)
        n_classes = len(self.classes_)
        class_idx = {c: i for i, c in enumerate(self.classes_)}

        proba = np.zeros((len(X), n_classes), dtype=float)

        for i in range(len(X)):
            neighbor_labels = self._k_nearest_labels(X[i])
            # histogram
            for label in neighbor_labels:
                proba[i][class_idx[label]] += 1.0
            # ratio
            for c in range(n_classes):
                proba[i][c] /= self.n_neighbors

        return proba

    # endregion Predictions

    # region Helpers

    def _check_is_fitted(self):
        if self._X_train is None:
            raise RuntimeError(
                "Classifier not fitted. Call fit() first."
            )

    def __repr__(self):
        return f"KNeighborsClassifier(n_neighbors={self.n_neighbors})"

    # endregion Helpers