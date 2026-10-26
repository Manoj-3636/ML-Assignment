from typing import Dict

import numpy as np
import pandas as pd
from common import DATASET_PATH, _stratified_split, compute_accuracy, get_df_split


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




# TODO move sanity checks to test folder (is this ok for submission tho?)

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


def run_part5() -> None:
    """
    Train and evaluate Flat Gaussian Naive Bayes on the Anuran Calls dataset
    :return: None
    """
    train_df,test_df = get_df_split()
    x_train = train_df.iloc[:, 0:22].to_numpy()
    x_test = test_df.iloc[:, 0:22].to_numpy()
    y_species_train = train_df["Species"].to_numpy()
    y_species_test = test_df["Species"].to_numpy()
    y_family_train = train_df["Family"].to_numpy()
    y_family_test = test_df["Family"].to_numpy()
    
    model = GaussianNB(var_smoothing=1e-9)
    model.fit(x_train, y_species_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    train_acc = compute_accuracy(y_species_train, y_train_pred)
    test_acc = compute_accuracy(y_species_test, y_test_pred)
    
    species_to_family: Dict[str, str] = dict(zip(y_species_train, y_family_train))
    y_family_pred_test = np.array([species_to_family[s] for s in y_test_pred])
    derived_family_acc = compute_accuracy(y_family_test, y_family_pred_test)

    print("\n=== Part 5: Flat Gaussian Naive Bayes Results ===")
    print(f"Train Accuracy (Species):         {train_acc:.2f}%")
    print(f"Test Accuracy (Species):          {test_acc:.2f}%")
    print(f"Derived Family Accuracy (Test):   {derived_family_acc:.2f}%\n")
    
    corr_matrix = np.corrcoef(x_train, rowvar=False)
    off_diag_mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    abs_off_diag = np.abs(corr_matrix[off_diag_mask])

    max_corr = float(np.max(abs_off_diag))
    mean_corr = float(np.mean(abs_off_diag))
    high_corr_pairs = int(np.sum(abs_off_diag > 0.5) // 2)

    print("=== Feature Correlation Analysis (22 MFCCs on Training Set) ===")
    print(f"Correlation Matrix Shape:         {corr_matrix.shape}")
    print(f"Mean |Correlation| (off-diag):    {mean_corr:.4f}")
    print(f"Max  |Correlation| (off-diag):    {max_corr:.4f}")
    print(f"Feature pairs with |r| > 0.5:     {high_corr_pairs} pairs")
    
    
    
    

if __name__ == "__main__":
    run_sanity_check()
    run_part5()
