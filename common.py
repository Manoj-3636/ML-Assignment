
from pathlib import Path
from graphviz import Digraph

import numpy as np
import pandas as pd

DATASET_PATH = Path("Frogs_MFCCs.csv")


def ensure_dataset() -> bool:
    """
    Check if the dataset exists in the current folder
    :return: True if dataset exists, False otherwise
    """
    if not DATASET_PATH.exists():
        return False

    print(f"Dataset available at: {DATASET_PATH.resolve()}")
    return True


def _stratified_split(labels, test_frac=0.2, seed=42):
    """
    Split function as per the assignment guidelines
    :param labels: label to stratify on
    :param test_frac: fraction of samples for test
    :param seed: number for rng
    :return: (train,test)
    """
    rng = np.random.RandomState(seed)
    test = []

    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        test.extend(idx[:int(round(test_frac * len(idx)))])

    test = np.sort(np.array(test))
    train = np.setdiff1d(np.arange(len(labels)), test)

    return train, test


def get_species_split():
    """
    Get the train test splits where X is the 22 features and Y is only the species column
    :return: (x_train,y_train,x_test,y_test)
    """
    if not ensure_dataset():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH.resolve()}. "
            "Please download Frogs_MFCCs.csv first."
        )

    df = pd.read_csv(DATASET_PATH)

    x = df.iloc[:, 0:22].to_numpy()
    y = df["Species"].to_numpy()

    train_idx, test_idx = _stratified_split(y)

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("Training shape:", x_train.shape)
    print("Test shape:", x_test.shape)

    return x_train, y_train, x_test, y_test



class TreeNode:
    def __init__(self, label, threshold=None, leaf=False):
        self.label = label
        self.left = None
        self.right = None
        self.threshold = threshold
        self.leaf = leaf

    def visualize(self, feature_labels: list[str]):
        g = Digraph(format='svg')
        node_idx = 0

        def _constructGraph(root):
            nonlocal node_idx

            curr_label = str(node_idx)
            node_idx += 1

            display_name = (
                str(root.label)
                if root.leaf
                else feature_labels[root.label]
            )

            g.node(curr_label, display_name)

            if root.left is not None:
                g.edge(
                    curr_label,
                    _constructGraph(root.left),
                    label=f"<= {root.threshold:.4f}",
                )

            if root.right is not None:
                g.edge(
                    curr_label,
                    _constructGraph(root.right),
                    label=f"> {root.threshold:.4f}",
                )

            return curr_label

        _constructGraph(self)
        return g


if __name__ == "__main__":
    get_species_split()