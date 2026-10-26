import numpy as np

from common import get_df_split, compute_accuracy
from part1_decision_tree import DecisionTree
from part2_bagging_boosting import majority_vote


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators=50,
        criterion="gini",
        max_features=4,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rng = np.random.RandomState(random_state)
        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = len(y)
        self.classes_ = np.unique(y)
        self.trees = []

        for _ in range(self.n_estimators):
            # Bootstrap sample: sample N rows with replacement.
            indices = self.rng.randint(0, n_samples, size=n_samples)

            tree = DecisionTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                rng=self.rng,
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X)
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return majority_vote(predictions, self.classes_)


def evaluate_model(name, model, X_train, y_train, X_test, y_test, species_to_family):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"\n{name}")
    print(f"Training species accuracy: {compute_accuracy(y_train, train_pred):.4f}%")
    print(f"Testing species accuracy:  {compute_accuracy(y_test, test_pred):.4f}%")

    true_family = np.array([species_to_family[s] for s in y_test])
    predicted_family = np.array([species_to_family[s] for s in test_pred])
    print(f"Testing family accuracy:   {compute_accuracy(true_family, predicted_family):.4f}%")


if __name__ == "__main__":
    # Use the current common.py interface, as in Part 1.
    df_train, df_test = get_df_split()

    X_train = df_train.iloc[:, :22].to_numpy()
    X_test = df_test.iloc[:, :22].to_numpy()
    y_train = df_train["Species"].to_numpy()
    y_test = df_test["Species"].to_numpy()

    # Derive the mapping from training data only.
    species_to_family = dict(zip(y_train, df_train["Family"].to_numpy()))

    # For 22 features, floor(sqrt(22)) = 4 features per split.
    max_features = int(np.sqrt(X_train.shape[1]))

    for crit in ["gini","entropy"]:

        print("=" * 30 + crit + "="* 30)
        forest = RandomForestClassifier(
            n_estimators=50,
            criterion=crit,
            max_features=max_features,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
        )

        print("Random Forest")
        print(f"Number of trees: {forest.n_estimators}")
        print(f"Features considered per split: {max_features}")
        print(f"Criterion: {forest.criterion}")

        forest.fit(X_train, y_train)
        evaluate_model(
            "Random Forest", forest,
            X_train, y_train, X_test, y_test,
            species_to_family,
        )
        print(f"Trees trained: {len(forest.trees)}")
