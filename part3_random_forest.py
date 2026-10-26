import numpy as np
import pandas as pd

from common import DATASET_PATH, get_species_split, _stratified_split
from part1_decision_tree import DecisionTree

# these functions can be reused from part2
from part2_bagging_boosting import majority_vote, get_species_family_mapping, accuracy, evaluate_model

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
        forest,
        X_train,
        y_train,
        X_test,
        y_test,
        species_to_family
    )