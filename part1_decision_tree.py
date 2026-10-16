import numpy as np
import pandas as pd
import nptyping as npt

from common import DATASET_PATH, _stratified_split, get_species_split

def gini(prob:npt.NDArray):
    return 1 - np.sum(prob ** 2)



class DecisionTree:
    """Decision tree classifier supporting Gini and entropy criteria."""

    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        max_features=None,
        rng=None,
    ):
        """Initialize the decision tree classifier."""
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")

        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None.")

        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")

        if max_features is not None and max_features < 1:
            raise ValueError("max_features must be positive or None.")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng

    def _best_split(self, X, y, feat, sample_weight=None):
        """Find the best split for a single feature.

        Args:
            X: Feature matrix for the current node.
            y: Class labels for the current node.
            feat: Index of the feature to evaluate.
            sample_weight: Optional weight for each sample.

        Returns:
            A tuple containing the best threshold and its information gain.
            Returns (None, 0.0) if no valid split has positive gain.
        """
        x = X[:, feat]
        n_samples = len(y)

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)

        classes, y_encoded = np.unique(y, return_inverse=True) #y_encoded has y but the elements are the indexes of classes
                                                               # so for each y if we go to the index in classes we reconstruct y
        n_classes = len(classes)

        order = np.argsort(x, kind="stable") # gives the indices that sort the array x
        x_sorted = x[order]
        y_sorted = y_encoded[order]
        weights_sorted = sample_weight[order]

        class_weights = np.bincount(
            y_encoded, weights=sample_weight, minlength=n_classes
        ) # counts occurrences of x and puts in class_weights[x] if sample weight is provided then that is added instead of 1
        total_weight = class_weights.sum()

        if self.criterion == "gini":
            parent_impurity = 1.0 - np.sum(
                (class_weights / total_weight) ** 2
            )
        else:
            probabilities = class_weights / total_weight
            nonzero = probabilities > 0
            parent_impurity = -np.sum(
                probabilities[nonzero] * np.log2(probabilities[nonzero])
            )

        cumulative_weights = np.zeros(
            (n_samples, n_classes), dtype=float
        )
        # this creates a one hot encoding for each sample rowwise that is if a sample has the 2nd feature [0 1 0 0]
        # same as putting a 1 at the encoded feature index,
        # but we may need to put weights for each sample instead of 1 so we do this below


        cumulative_weights[np.arange(n_samples), y_sorted] = weights_sorted
        #row is just every sample and the column index is the encoded feature

        cumulative_weights = np.cumsum(cumulative_weights, axis=0)
        #sum up the weights cumulatively to track occurrences to the left

        left_weights = cumulative_weights[:-1] # we will split between x[i] and x[i+1] so the last index is useless
        right_weights = class_weights - left_weights

        left_total = left_weights.sum(axis=1) # weights of all samples so one sum per column equals the total no of examples if weights = 1
        right_total = right_weights.sum(axis=1)

        valid = x_sorted[:-1] < x_sorted[1:] # check if x[i} is greater than x[i+1]

        if not np.any(valid):
            return None, 0.0

        left_probabilities = left_weights / left_total[:, None]
        right_probabilities = right_weights / right_total[:, None]
        # this is done for broadcasting so that so every column is divided by its row total weight

        if self.criterion == "gini":
            left_impurity = 1.0 - np.sum(
                left_probabilities**2, axis=1
            )
            right_impurity = 1.0 - np.sum(
                right_probabilities**2, axis=1
            )
        else:
            left_terms = np.zeros_like(left_probabilities)
            right_terms = np.zeros_like(right_probabilities)

            left_mask = left_probabilities > 0
            right_mask = right_probabilities > 0

            left_terms[left_mask] = (
                left_probabilities[left_mask]
                * np.log2(left_probabilities[left_mask])
            )
            right_terms[right_mask] = (
                right_probabilities[right_mask]
                * np.log2(right_probabilities[right_mask])
            )

            left_impurity = -np.sum(left_terms, axis=1)
            right_impurity = -np.sum(right_terms, axis=1)

        # we get the left impurity and right impurity for each row now we have everything we can just get the total gain
        # and do argmax

        child_impurity = (
            left_total * left_impurity
            + right_total * right_impurity
        ) / total_weight

        gains = parent_impurity - child_impurity
        gains[~valid] = -np.inf
        # so that only valid points win

        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]

        if best_gain <= 0:
            return None, 0.0

        threshold = x_sorted[best_idx] / 2 + x_sorted[best_idx + 1] / 2
        # the final split point

        return threshold, best_gain
