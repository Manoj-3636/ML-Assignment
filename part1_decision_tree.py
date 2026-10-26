from typing import Dict

import numpy as np
import pandas
import pandas as pd
import numpy.typing as npt
from common import TreeNode,get_df_split,compute_accuracy

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
        self.classes_ = None
        self.root = None
        self.n_features_in_ = None
        self.n_classes_ = None
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

    def _best_feature_split(self, X,y_encoded, feat,n_classes ,sample_weight=None):
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
        n_samples = len(y_encoded)

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)

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

    def _best_split(self, X, y_encoded, n_classes, sample_weight=None):
        """
        Find the best split for the current dataset.

        Args:
            X: Feature matrix for the current node.
            y_encoded: Encoded class labels for the current node.
            n_classes: Total number of classes in the training data.
            sample_weight: Optional weight for each sample.

        Returns:
            A tuple containing the best feature index, threshold, and gain.
            Returns (None, None, 0.0) if no valid split has positive gain.
        """
        n_samples, n_features = X.shape
        # Note that if max_features is left none then RNG can also be left none
        # TODO better way to support this
        if self.max_features is None:
            features = np.arange(n_features)
        else:
            n_features_to_select = min(self.max_features, n_features)
            features = self.rng.choice(
                n_features,
                size=n_features_to_select,
                replace=False,
            )
            features = np.sort(features)

        best_feature = None
        best_threshold = None
        best_gain = 0.0

        for feat in features:
            threshold, gain = self._best_feature_split(
                X,
                y_encoded,
                feat,
                n_classes,
                sample_weight,
            )

            if gain > best_gain:
                best_feature = feat
                best_threshold = threshold
                best_gain = gain

        # this automatically makes sure that when two features are conflicting the one with lower feature id wins
        return best_feature, best_threshold, best_gain

    def _build_tree(
            self,
            X,
            y_encoded,
            n_classes,
            sample_weight,
            depth=0,
    ):
        """
        Recursively build the decision tree for the current dataset.

        Args:
            X: Feature matrix for the current node.
            y_encoded: Encoded class labels for the current node.
            n_classes: Total number of classes in the training data.
            sample_weight: Sample weights for the current node.
            depth: Depth of the current node, with the root at depth 0.

        Returns:
            The root node of the constructed subtree.
        """
        n_samples = len(y_encoded)

        class_weights = np.bincount(
            y_encoded,
            weights=sample_weight,
            minlength=n_classes,
        )

        majority_class = np.argmax(class_weights)

        # Stop if the node is pure.
        if np.count_nonzero(class_weights) == 1:
            return TreeNode(label=majority_class, leaf=True,threshold=-3.14) # specific number to detect bugs down the line

        # Stop if the maximum depth has been reached.
        if self.max_depth is not None and depth >= self.max_depth:
            return TreeNode(label=majority_class, leaf=True,threshold=-3.14)

        # Stop if there are too few samples to split.
        if n_samples < self.min_samples_split:
            return TreeNode(label=majority_class, leaf=True,threshold=-3.14)

        feature_idx, threshold, gain = self._best_split(
            X,
            y_encoded,
            n_classes,
            sample_weight,
        )

        # Stop if no useful split exists.
        if feature_idx is None or gain <= 0:
            return TreeNode(label=majority_class, leaf=True,threshold=-3.14)

        left_mask = X[:, feature_idx] <= threshold # get all satisfying indices
        right_mask = ~left_mask

        # Construct the internal node.
        node = TreeNode(
            label=feature_idx,
            threshold=threshold,
        )

        # Recursively construct the left subtree.
        node.left = self._build_tree(
            X[left_mask],
            y_encoded[left_mask],
            n_classes,
            sample_weight[left_mask],
            depth + 1,
        )

        # Recursively construct the right subtree.
        node.right = self._build_tree(
            X[right_mask],
            y_encoded[right_mask],
            n_classes,
            sample_weight[right_mask],
            depth + 1,
        )

        return node

    def fit(self, X, y, sample_weight=None):
        """
        Fit the decision tree classifier to the training data.

        Args:
            X: Feature matrix for the training samples.
            y: Target class labels.
            sample_weight: Optional weight for each training sample.

        Returns:
            The fitted DecisionTree instance.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        n_samples = len(y)

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)

        if len(X) != n_samples or len(sample_weight) != n_samples:
            raise ValueError("X, y, and sample_weight must have the same number of samples.")

        if np.any(sample_weight < 0) or sample_weight.sum() <= 0:
            raise ValueError("Sample weights must be non-negative with a positive sum.")

        self.n_classes_ = n_classes
        self.n_features_in_ = X.shape[1]

        self.root = self._build_tree(
            X,
            y_encoded,
            n_classes,
            sample_weight,
        )

    def predict(self, X):
        """
        Predict class labels for the input samples.

        Args:
            X: Feature matrix for the samples to predict.

        Returns:
            Predicted class labels (vector).
        """
        X = np.asarray(X)

        def _predict_one(x):
            node = self.root

            while not node.leaf:
                if x[node.label] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            return node.label

        y_encoded = np.array([_predict_one(x) for x in X])

        return self.classes_[y_encoded]

if __name__ == "__main__":
    df_train,df_test = get_df_split()
    x_train = df_train.iloc[:, 0:22].to_numpy()
    x_test = df_test.iloc[:,0:22].to_numpy()

    y_train_species = df_train["Species"]
    y_train_family = df_train["Family"]

    y_test_species = df_test["Species"]
    y_test_family = df_test["Family"]

    dt_gini = DecisionTree("gini")
    dt_gini.fit(x_train,y_train_species)

    train_pred_gini = dt_gini.predict(x_train)
    test_pred_gini = dt_gini.predict(x_test)

    print(f"Gini Training Accuracy: {compute_accuracy(y_train_species,train_pred_gini)}")
    print(f"Gini Testing Accuracy: {compute_accuracy(y_test_species,test_pred_gini)}")

    dt_entropy = DecisionTree("entropy")
    dt_entropy.fit(x_train, y_train_species)

    train_pred_entropy = dt_entropy.predict(x_train)
    test_pred_entropy = dt_entropy.predict(x_test)

    print(f"Entropy Training Accuracy: {compute_accuracy(y_train_species, train_pred_entropy)}")
    print(f"Entropy Testing Accuracy: {compute_accuracy(y_test_species, test_pred_entropy)}")

    species_to_family: Dict[str, str] = dict(zip(y_train_species, y_train_family))

    train_pred_family_gini = np.array([species_to_family[species] for species in train_pred_gini])
    test_pred_family_gini = np.array([species_to_family[species] for species in test_pred_gini])

    train_pred_family_entropy = np.array([species_to_family[species] for species in train_pred_entropy])
    test_pred_family_entropy = np.array([species_to_family[species] for species in test_pred_entropy])

    print(f"Gini Training Derived Family Accuracy: {compute_accuracy(y_train_family, train_pred_family_gini)}")
    print(f"Gini Testing Derived Family Accuracy: {compute_accuracy(y_test_family, test_pred_family_gini)}")

    print(f"Entropy Training Derived Family Accuracy: {compute_accuracy(y_train_family, train_pred_family_entropy)}")
    print(f"Entropy Testing Derived Family Accuracy: {compute_accuracy(y_test_family, test_pred_family_entropy)}")
    print("=" * 40)
    print("Accuracies for various max depths")
    max_depths = [1, 2, 3, 5, 8, 12, None]

    for criterion in ["gini", "entropy"]:
        print(f"\n{criterion.capitalize()} Decision Tree")

        for max_depth in max_depths:
            tree = DecisionTree(criterion, max_depth=max_depth)
            tree.fit(x_train, y_train_species)

            train_pred = tree.predict(x_train)
            test_pred = tree.predict(x_test)

            print(
                f"max_depth={max_depth} | "
                f"Training Accuracy: {compute_accuracy(y_train_species, train_pred)} | "
                f"Testing Accuracy: {compute_accuracy(y_test_species, test_pred)}"
            )