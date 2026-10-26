from part1_decision_tree import DecisionTree
import numpy as np


def majority_vote(predictions, classes):
    """
    To combine predictions of all the Tree and picks the majority one.

    predictions: shape (n_trees, n_samples)
    classes: sorted unique class labels
    """
    encoded = np.searchsorted(classes, predictions)

    n_samples = predictions.shape[1] # no of samples
    votes = np.zeros((n_samples, len(classes)), dtype=int)
    rows = np.arange(n_samples)

    for tree_predictions in encoded:
        votes[rows, tree_predictions] += 1

    # np.argmax chooses the first class in a tie.
    return classes[np.argmax(votes, axis=1)]

class BaggingClassifier:
    def __init__(self, n_estimators=25, criterion="gini"):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.rng = np.random.RandomState(42)
        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = len(y)
        self.classes_ = np.unique(y)
        self.trees = []

        for _ in range(self.n_estimators):
            # Draw bootstrap sample of size N, with replacement
            indices = self.rng.randint(0, n_samples, size=n_samples)

            tree = DecisionTree(
                criterion=self.criterion,
                max_depth=None,
                min_samples_split=2,
                max_features=None,
                rng=self.rng
            )

            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

        return self

    def predict(self, X):
        # Every tree predicts the same input samples
        predictions = np.array([
            tree.predict(X) for tree in self.trees
        ])

        return majority_vote(predictions, self.classes_)

class AdaBoostSAMME:
    def __init__(self, n_estimators=50, criterion="gini", max_depth=3):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth

        self.rng = np.random.RandomState(42)
        self.estimators_ = []
        self.alphas_ = []
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = len(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Initially, every sample has equal weight
        weights = np.full(n_samples, 1.0 / n_samples)

        self.estimators_ = []
        self.alphas_ = []

        for _ in range(self.n_estimators):
            tree = DecisionTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=2,
                max_features=None,
                rng=self.rng
            )

            # Train using sample weights
            tree.fit(X, y, sample_weight=weights)

            predictions = tree.predict(X)
            incorrect = predictions != y

            # Weighted classification error
            error = np.sum(weights[incorrect])
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # Stop if the learner is no better than random guessing
            if error >= 1 - 1 / n_classes:
                break

            # Importance of this learner
            alpha = (
                np.log((1 - error) / error)
                + np.log(n_classes - 1)
            )

            self.estimators_.append(tree)
            self.alphas_.append(alpha)

            # Increase weights of misclassified samples
            weights *= np.exp(alpha * incorrect)
            weights /= weights.sum()

        if not self.estimators_:
            raise ValueError("AdaBoost did not find a useful learner.")

        return self

    def predict(self, X):
        X = np.asarray(X)

        scores = np.zeros((len(X), len(self.classes_)))
        rows = np.arange(len(X))

        for tree, alpha in zip(self.estimators_, self.alphas_):
            predictions = tree.predict(X)
            encoded = np.searchsorted(self.classes_, predictions)

            # Weighted voting: each tree contributes alpha
            scores[rows, encoded] += alpha

        return self.classes_[np.argmax(scores, axis=1)]