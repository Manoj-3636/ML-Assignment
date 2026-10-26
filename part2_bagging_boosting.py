from common import TreeNode, get_species_split, _stratified_split, DATASET_PATH
from part1_decision_tree import DecisionTree
import numpy as np
import pandas as pd


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

# Species to Family mapping
def get_species_family_mapping():
    df = pd.read_csv(DATASET_PATH)

    # Load the species and family labels.
    species = df["Species"].to_numpy()
    family = df["Family"].to_numpy()

    # Use training rows only
    train_idx, _ = _stratified_split(species)

    mapping = {}

    # For each unique species, find its corresponding family and store it in a dictionary.
    for s in np.unique(species[train_idx]):
        matching_families = family[train_idx][species[train_idx] == s]
        mapping[s] = matching_families[0]

    return mapping

# Evaluation
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def evaluate_model(
    name, model, X_train, y_train,
    X_test, y_test, species_to_family
):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"\n{name}")
    print(f"Training species accuracy: {accuracy(y_train, train_pred):.4f}")
    print(f"Testing species accuracy:  {accuracy(y_test, test_pred):.4f}")

    # Convert species predictions into family predictions
    true_family = np.array([
        species_to_family[s] for s in y_test
    ])

    predicted_family = np.array([
        species_to_family[s] for s in test_pred
    ])

    print(
        f"Testing family accuracy:   "
        f"{accuracy(true_family, predicted_family):.4f}"
    )

# Main

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_species_split()
    species_to_family = get_species_family_mapping()

    for criterion in ["gini", "entropy"]: # for each criterion

        # Bagging: 25 unrestricted trees
        bagging = BaggingClassifier(
            n_estimators=25,
            criterion=criterion
        )
        bagging.fit(X_train, y_train)

        evaluate_model(
            f"Bagging ({criterion})",
            bagging,
            X_train, y_train,
            X_test, y_test,
            species_to_family
        )

        print(f"Bagging trees trained: {len(bagging.trees)}")

        # AdaBoost: depth-3 trees, up to 50 rounds
        boosting = AdaBoostSAMME(
            n_estimators=50,
            criterion=criterion,
            max_depth=3
        )
        boosting.fit(X_train, y_train)

        evaluate_model(
            f"AdaBoost ({criterion})",
            boosting,
            X_train, y_train,
            X_test, y_test,
            species_to_family
        )

        print(f"AdaBoost rounds completed: {len(boosting.estimators_)}")

        # Extra experiment: decision stumps (max_depth=1)
        stumps = AdaBoostSAMME(
            n_estimators=50,
            criterion=criterion,
            max_depth=1
        )
        stumps.fit(X_train, y_train)

        print(
            f"AdaBoost stumps ({criterion}) test accuracy: "
            f"{accuracy(y_test, stumps.predict(X_test)):.4f}"
        )

        """
        A decision stump has max_depth = 1, meaning it makes only one split
        and can create at most two leaf nodes. Therefore, it has limited
        ability to distinguish between 10 different species with
        complex decision boundaries.
        """