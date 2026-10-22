from typing import Dict

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

    def _route_and_predict(
        self,
        X_arr: npt.NDArray[np.float64],
        family_assignments: npt.NDArray[np.object_],
    ) -> npt.NDArray[np.object_]:
        """Group samples by assigned family and predict Species using each expert once.

        :param X_arr: Feature matrix of shape (n_samples, n_features).
        :param family_assignments: 1D array of family names (predicted or oracle true).
        :return: 1D array of predicted Species labels in original sample order.
        """
        y_pred_species: npt.NDArray[np.object_] = np.empty(
            len(X_arr), dtype=object
        )

        for family_name in np.unique(family_assignments):
            if family_name not in self.experts:
                raise KeyError(
                    f"Assigned family '{family_name}' has no registered expert."
                )
            group_indices = np.where(family_assignments == family_name)[0]
            y_pred_species[group_indices] = self.experts[family_name].predict(
                X_arr[group_indices]
            )

        return y_pred_species

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
        return self._route_and_predict(X_arr, y_pred_family)

    def predict_oracle(
        self,
        X: npt.NDArray[np.float64],
        y_true_family: npt.NDArray[np.object_],
    ) -> npt.NDArray[np.object_]:
        """Predict Species using an ideal Oracle router (routes by true Family).

        :param X: Feature matrix of shape (n_samples, n_features).
        :param y_true_family: True Family labels of shape (n_samples,).
        :return: 1D array of oracle-routed Species predictions of shape (n_samples,).
        """
        X_arr = np.asarray(X, dtype=float)
        y_fam_arr = np.asarray(y_true_family, dtype=object)
        return self._route_and_predict(X_arr, y_fam_arr)


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
    router = DecisionTree(criterion=criterion, max_depth=None, min_samples_split=2)
    router.fit(X_train, y_train_family)

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


def evaluate_hierarchy_metrics(
    clf: HierarchicalClassifier,
    X_train: npt.NDArray[np.float64],
    y_train_family: npt.NDArray[np.object_],
    y_train_species: npt.NDArray[np.object_],
    X_test: npt.NDArray[np.float64],
    y_test_family: npt.NDArray[np.object_],
    y_test_species: npt.NDArray[np.object_],
) -> Dict[str, float]:
    """Compute all router, overall, oracle, and failure mode metrics (in %).

    :param clf: Trained HierarchicalClassifier instance.
    :param X_train: Training feature matrix.
    :param y_train_family: True training Family labels.
    :param y_train_species: True training Species labels.
    :param X_test: Test feature matrix.
    :param y_test_family: True test Family labels.
    :param y_test_species: True test Species labels.
    :return: Dictionary of computed percentages and numerical check difference.
    """
    train_pred_fam = clf.predict_family(X_train)
    train_pred_sp = clf.predict(X_train)

    a_router_train = compute_accuracy(y_train_family, train_pred_fam)
    a_overall_train = compute_accuracy(y_train_species, train_pred_sp)

    test_pred_fam = clf.predict_family(X_test)
    test_pred_sp = clf.predict(X_test)
    test_oracle_sp = clf.predict_oracle(X_test, y_test_family)

    a_router_test = compute_accuracy(y_test_family, test_pred_fam)
    a_overall_test = compute_accuracy(y_test_species, test_pred_sp)
    a_oracle_test = compute_accuracy(y_test_species, test_oracle_sp)

    router_right_mask = test_pred_fam == y_test_family
    species_right_mask = test_pred_sp == y_test_species

    p_a = float(np.mean(router_right_mask & (~species_right_mask)) * 100.0)
    p_b = float(np.mean(~router_right_mask) * 100.0)

    lhs = a_overall_test
    rhs = a_router_test - p_a
    abs_diff = abs(lhs - rhs)

    return {
        "a_router_train": a_router_train,
        "a_overall_train": a_overall_train,
        "a_router_test": a_router_test,
        "a_overall_test": a_overall_test,
        "a_oracle_test": a_oracle_test,
        "p_a": p_a,
        "p_b": p_b,
        "rhs_check": rhs,
        "abs_diff": abs_diff,
    }


if __name__ == "__main__":
    df_train, df_test = get_df_split()

    X_train: npt.NDArray[np.float64] = df_train.iloc[:, 0:22].to_numpy(dtype=float)
    X_test: npt.NDArray[np.float64] = df_test.iloc[:, 0:22].to_numpy(dtype=float)

    y_train_family: npt.NDArray[np.object_] = df_train["Family"].to_numpy()
    y_train_species: npt.NDArray[np.object_] = df_train["Species"].to_numpy()

    y_test_family: npt.NDArray[np.object_] = df_test["Family"].to_numpy()
    y_test_species: npt.NDArray[np.object_] = df_test["Species"].to_numpy()

    print("=" * 76)
    print("CHECKPOINT 2: ORACLE ACCURACY & FAILURE MODE DECOMPOSITION (TABLE 2 & 3)")
    print("=" * 76)

    for crit in ("gini", "entropy"):
        clf = train_hierarchical_pipeline(
            X_train, y_train_family, y_train_species, criterion=crit
        )
        m = evaluate_hierarchy_metrics(
            clf,
            X_train,
            y_train_family,
            y_train_species,
            X_test,
            y_test_family,
            y_test_species,
        )

        print(f"\n--- Criterion: {crit.upper()} ---")
        print(
            f"  Router Accuracy (A_router)   : "
            f"Train = {m['a_router_train']:6.2f}% | Test = {m['a_router_test']:6.2f}%"
        )
        print(
            f"  Overall Accuracy (A_overall) : "
            f"Train = {m['a_overall_train']:6.2f}% | Test = {m['a_overall_test']:6.2f}%"
        )
        print(
            f"  Oracle Accuracy (A_oracle)   : "
            f"Test  = {m['a_oracle_test']:6.2f}% "
            f"(Price of router errors: {m['a_oracle_test'] - m['a_overall_test']:.2f}%)"
        )
        print(
            f"  Failure Mode (A) [p_A]       : {m['p_a']:6.2f}% "
            f"(Router right, expert wrong)"
        )
        print(
            f"  Failure Mode (B) [p_B]       : {m['p_b']:6.2f}% "
            f"(Router wrong = 100% - A_router)"
        )
        print(
            f"  Numerical Identity Check     : "
            f"A_overall ({m['a_overall_test']:.6f}%) == "
            f"A_router - p_A ({m['rhs_check']:.6f}%) | "
            f"|diff| = {m['abs_diff']:.2e}"
        )
        assert np.isclose(m["a_overall_test"], m["rhs_check"], atol=1e-12)
