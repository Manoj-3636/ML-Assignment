import numpy as np
import pandas as pd

from common import DATASET_PATH, get_species_split, _stratified_split
from part1_decision_tree import DecisionTree

# these functions can be reused from part2
from part2_bagging_boosting import majority_vote, get_species_family_mapping, accuracy, evaluate_model

class RandomForestClassifier:
    def __init__(self,
        n_estimators=50,
        criterion="gini",
        max_features=4,
        max_depth=None,
        min_samples_split=2,
        random_state=42):
        
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

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

            # Draw bootstrap sample with replacement
            indices = self.rng.randint(
                0, n_samples, size=n_samples
            )

            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Train a tree using random features at each split
            tree = DecisionTree(
                criterion=self.criterion,
                max_depth=None,
                min_samples_split=2,
                max_features=self.max_features,
                rng=self.rng
            )

            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X)

        # Every tree predicts the same input samples
        predictions = np.array([
            tree.predict(X) for tree in self.trees
        ])

        # Combine predictions using majority voting
        return majority_vote(predictions, self.classes_)

if __name__ == "__main__":

    # Load the exact assignment split.
    X_train, y_train, X_test, y_test = get_species_split()

    species_to_family = get_species_family_mapping()

    # 22 features -> floor(sqrt(22)) = 4 features per split.
    max_features = int(np.sqrt(X_train.shape[1]))

    print("Random Forest :")
    print(f"Number of trees: {50}")
    print(f"Features considered per split: {max_features}")
    print(f"Criterion: gini")
    print()

    # put in all the input parameters
    forest = RandomForestClassifier(
        n_estimators=50,
        criterion="gini",
        max_features=max_features,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )

    forest.fit(X_train, y_train)

    evaluate_model(
        "Random Forest",
        forest,
        X_train,
        y_train,
        X_test,
        y_test,
        species_to_family
    )