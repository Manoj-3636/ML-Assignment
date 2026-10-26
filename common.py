from pathlib import Path
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


def get_df_split():
    """
    Get the train test splits of the whole dataset
    :return: (df_train,df_test)
    """
    if not ensure_dataset():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH.resolve()}. "
            "Please download Frogs_MFCCs.csv first."
        )

    df = pd.read_csv(DATASET_PATH)

    y = df["Species"].to_numpy()

    train_idx, test_idx = _stratified_split(y)


    return df.iloc[train_idx],df.iloc[test_idx]


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy as a percentage (0.0 to 100.0)
    :param y_true: 1D array of ground truth labels
    :param y_pred: 1D array of predicted labels
    :return: accuracy percentage
    """
    return float(np.mean(y_true == y_pred) * 100.0)


class TreeNode:
    def __init__(self, label, threshold=None, leaf=False):
        self.label = label
        self.left = None
        self.right = None
        self.threshold = threshold
        self.leaf = leaf


if __name__ == "__main__":
    train_df,test_df = get_df_split()
    print(train_df.shape)
    print(test_df.shape)