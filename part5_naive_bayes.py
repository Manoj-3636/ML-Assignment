import numpy as np


class GaussianNB:
    """Flat Gaussian Naive Bayes classifier implemented from scratch using NumPy."""

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        """
        Initialize the Gaussian Naive Bayes classifier
        :param var_smoothing: portion of the largest variance of all features added for stability
        """
        self.var_smoothing: float = var_smoothing
        self.classes_: np.ndarray = np.array([])
        self.priors_: np.ndarray = np.array([])
        self.means_: np.ndarray = np.array([])
        self.vars_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        """
        Fit Gaussian Naive Bayes by estimating class priors, means, and smoothed variances
        :param X: training feature matrix of shape (N, d)
        :param y: training label vector of shape (N,)
        :return: fitted GaussianNB instance
        """

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.priors_ = np.zeros(n_classes, dtype=np.float64)
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.vars_ = np.zeros((n_classes, n_features), dtype=np.float64)

        epsilon: float = float(self.var_smoothing * np.max(np.var(X, axis=0)))
        for idx, c in enumerate(self.classes_):
            x_c = X[y == c]
            self.priors_[idx] = x_c.shape[0] / n_samples
            self.means_[idx, :] = np.mean(x_c, axis=0)
            self.vars_[idx, :] = np.var(x_c, axis=0) + epsilon

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X using 3D vectorized log-likelihoods
        :param X: input feature matrix of shape (n, d)
        :return: predicted class labels of shape (n,)
        """

        diff_sq = (X[:, np.newaxis, :] - self.means_[np.newaxis, :, :]) ** 2
        log_var = np.log(2.0 * np.pi * self.vars_[np.newaxis, :, :])
        log_likelihood = -0.5 * np.sum(
            log_var + (diff_sq / self.vars_[np.newaxis, :, :]), axis=2
        )
        log_posterior = np.log(self.priors_)[np.newaxis, :] + log_likelihood
        best_class_idx = np.argmax(log_posterior, axis=1)
        return self.classes_[best_class_idx]


def run_sanity_check() -> None:
    """
    Run the mandatory sanity check specified in Part 5 of the assignment
    :return: None
    """
    x_train_sanity = np.array([[0.0], [2.0], [10.0], [12.0]], dtype=np.float64)
    y_train_sanity = np.array([0, 0, 1, 1])

    model = GaussianNB(var_smoothing=1e-9)
    model.fit(x_train_sanity, y_train_sanity)

    x_test_sanity = np.array([[3.0], [9.0]], dtype=np.float64)
    preds = model.predict(x_test_sanity)

    print("=== Part 5 Sanity Check ===")
    print(f"Estimated Means (Expected [1.0, 11.0]): {model.means_.ravel()}")
    print(f"Estimated Variances (Expected ~[1.0, 1.0]): {model.vars_.ravel()}")
    print(f"Prediction for x=3 (Expected 0): {preds[0]}")
    print(f"Prediction for x=9 (Expected 1): {preds[1]}")
    assert preds[0] == 0 and preds[1] == 1, "Sanity check failed!"
    print("Sanity check PASSED!\n")


if __name__ == "__main__":
    run_sanity_check()
