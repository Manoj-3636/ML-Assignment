from part1_decision_tree import DecisionTree
from common import get_df_split, compute_accuracy
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

def evaluate_model(
    name, model, X_train, y_train,
    X_test, y_test, species_to_family
):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"\n{name}")
    print(f"Training species accuracy: {compute_accuracy(y_train, train_pred):.2f}%")
    print(f"Testing species accuracy:  {compute_accuracy(y_test, test_pred):.2f}%")

    # Convert species predictions into family predictions
    true_family = np.array([
        species_to_family[s] for s in y_test
    ])

    predicted_family = np.array([
        species_to_family[s] for s in test_pred
    ])

    print(
        f"Testing family accuracy:   "
        f"{compute_accuracy(true_family, predicted_family):.2f}%"
    )

# Main

if __name__ == "__main__":
    # Use the shared data interface, just like Part 1.
    df_train, df_test = get_df_split()

    X_train = df_train.iloc[:, 0:22].to_numpy()
    X_test = df_test.iloc[:, 0:22].to_numpy()

    y_train = df_train["Species"].to_numpy()
    y_test = df_test["Species"].to_numpy()

    # Build the species-to-family mapping from training data only.
    species_to_family = dict(
        zip(df_train["Species"], df_train["Family"])
    )

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
            f"{compute_accuracy(y_test, stumps.predict(X_test)):.2f}%"
        )

        """
        A decision stump has max_depth = 1, meaning it makes only one split
        and can create at most two leaf nodes. Therefore, it has limited
        ability to distinguish between 10 different species with
        complex decision boundaries.
        """