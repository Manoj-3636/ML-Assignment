from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
from common import compute_accuracy, get_df_split
from part1_decision_tree import DecisionTree


class HierarchicalClassifier:
    """Two-level hierarchical decision tree classifier (Router -> Family Experts).

    The router predicts the coarse Family label (4 classes), and routes each
    sample to the corresponding family-specific expert DecisionTree to predict
    the fine-grained Species label (10 classes total).
    """

    def __init__(
        self,
        router: DecisionTree,
        experts: Dict[str, DecisionTree],
    ) -> None:
        """Initialize the HierarchicalClassifier with a trained router and experts.

        :param router: DecisionTree trained on all training samples to predict Family.
        :param experts: Dictionary mapping each family name to its trained expert DecisionTree.
        """
        self.router: DecisionTree = router
        self.experts: Dict[str, DecisionTree] = experts

    def predict_family(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.object_]:
        """Predict the coarse Family label for each sample using the router tree.

        :param X: Feature matrix of shape (n_samples, n_features).
        :return: 1D array of predicted Family labels of shape (n_samples,).
        """
        X_arr = np.asarray(X, dtype=float)
        return self.router.predict(X_arr)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.object_]:
        """Predict the Species label by routing samples to family-specific experts.

        Routes all samples first using the router, groups sample indices by their
        predicted family, invokes each family expert once on its assigned group,
        and scatters predictions back into the original sample order.

        :param X: Feature matrix of shape (n_samples, n_features).
        :return: 1D array of predicted Species labels of shape (n_samples,).
        """
        X_arr = np.asarray(X, dtype=float)
        y_pred_family = self.predict_family(X_arr)

        y_pred_species: npt.NDArray[np.object_] = np.empty(
            len(X_arr), dtype=object
        )

        # Group by predicted family (at most 4 groups) and call each expert once
        for family_name in np.unique(y_pred_family):
            if family_name not in self.experts:
                raise KeyError(
                    f"Predicted family '{family_name}' has no registered expert."
                )
            group_indices = np.where(y_pred_family == family_name)[0]
            y_pred_species[group_indices] = self.experts[family_name].predict(
                X_arr[group_indices]
            )

        return y_pred_species


def train_hierarchical_pipeline(
    X_train: npt.NDArray[np.float64],
    y_train_family: npt.NDArray[np.object_],
    y_train_species: npt.NDArray[np.object_],
    criterion: str = "gini",
) -> HierarchicalClassifier:
    """Train the router tree and the 4 family-specific expert trees.

    :param X_train: Training feature matrix of shape (n_train, 22).
    :param y_train_family: True training Family labels of shape (n_train,).
    :param y_train_species: True training Species labels of shape (n_train,).
    :param criterion: Splitting criterion ('gini' or 'entropy') used for all trees.
    :return: Trained HierarchicalClassifier instance.
    """
    # 1. Train Router on all training rows with true Family labels
    router = DecisionTree(criterion=criterion, max_depth=None, min_samples_split=2)
    router.fit(X_train, y_train_family)

    # 2. Partition training data by true Family and train one expert per family
    experts: Dict[str, DecisionTree] = {}
    unique_families = np.unique(y_train_family)

    for family_name in unique_families:
        family_mask = y_train_family == family_name
        X_family = X_train[family_mask]
        y_species_family = y_train_species[family_mask]

        expert = DecisionTree(
            criterion=criterion, max_depth=None, min_samples_split=2
        )
        expert.fit(X_family, y_species_family)
        experts[str(family_name)] = expert

    return HierarchicalClassifier(router=router, experts=experts)


def inspect_hierarchy(
    y_train_family: npt.NDArray[np.object_],
    y_train_species: npt.NDArray[np.object_],
) -> None:
    """Print the taxonomy structure (Family -> Species counts) in the training split.

    :param y_train_family: True training Family labels.
    :param y_train_species: True training Species labels.
    """
    print("=" * 72)
    print("TAXONOMY INSPECTION (TRAINING SET: FAMILY -> SPECIES)")
    print("=" * 72)
    for fam in np.unique(y_train_family):
        mask = y_train_family == fam
        species_in_fam, counts = np.unique(
            y_train_species[mask], return_counts=True
        )
        species_str = ", ".join(
            f"{sp} ({cnt})" for sp, cnt in zip(species_in_fam, counts)
        )
        print(
            f"  {str(fam):<18} | Total: {int(np.sum(mask)):>4} | "
            f"Species ({len(species_in_fam)}): {species_str}"
        )
    print("=" * 72)


if __name__ == "__main__":
    df_train, df_test = get_df_split()

    X_train: npt.NDArray[np.float64] = df_train.iloc[:, 0:22].to_numpy(dtype=float)
    X_test: npt.NDArray[np.float64] = df_test.iloc[:, 0:22].to_numpy(dtype=float)

    y_train_family: npt.NDArray[np.object_] = df_train["Family"].to_numpy()
    y_train_species: npt.NDArray[np.object_] = df_train["Species"].to_numpy()

    y_test_family: npt.NDArray[np.object_] = df_test["Family"].to_numpy()
    y_test_species: npt.NDArray[np.object_] = df_test["Species"].to_numpy()

    # Step 1: Inspect the data hierarchy
    inspect_hierarchy(y_train_family, y_train_species)

    # Step 2: Verify Gini & Entropy training and basic predictions
    for crit in ("gini", "entropy"):
        clf = train_hierarchical_pipeline(
            X_train, y_train_family, y_train_species, criterion=crit
        )

        train_fam_pred = clf.predict_family(X_train)
        test_fam_pred = clf.predict_family(X_test)

        train_sp_pred = clf.predict(X_train)
        test_sp_pred = clf.predict(X_test)

        print(f"\n[{crit.upper()}] Checkpoint 1 Verification:")
        print(
            f"  Router Accuracy (Family)  -> "
            f"Train: {compute_accuracy(y_train_family, train_fam_pred):.2f}% | "
            f"Test: {compute_accuracy(y_test_family, test_fam_pred):.2f}%"
        )
        print(
            f"  Overall Accuracy (Species) -> "
            f"Train: {compute_accuracy(y_train_species, train_sp_pred):.2f}% | "
            f"Test: {compute_accuracy(y_test_species, test_sp_pred):.2f}%"
        )
        # Verify single-species experts are leaf nodes at root
        for fam_name, exp_tree in clf.experts.items():
            assert exp_tree.root is not None
            print(
                f"  Expert '{fam_name:<16}': n_classes={exp_tree.n_classes_}, "
                f"root_is_leaf={exp_tree.root.leaf}"
            )