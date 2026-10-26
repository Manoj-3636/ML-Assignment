# BITS F464 — Machine Learning Assignment 1
## Flat vs. Hierarchical Classification From Scratch

This repository contains implementations of five machine learning methods for the BITS F464 Machine Learning course at BITS Pilani, Semester I, Academic Year 2026–27.

The assignment explores flat and hierarchical classification using the **Anuran Calls (MFCCs)** dataset. Decision trees, ensemble methods, and Gaussian Naive Bayes are implemented from scratch using NumPy and pandas, without relying on machine learning libraries.

## Table of Contents

- [Overview](#overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Experiments](#running-the-experiments)
- [Implementation Details](#implementation-details)
- [Evaluation](#evaluation)
- [Notes](#notes)

## Overview

The objective is to compare flat species classification against hierarchical classification, while studying the effects of ensemble learning, feature subsetting, and conditional independence assumptions.

The repository implements:

1. A single decision tree using Gini impurity and information gain.
2. Bagging and multiclass AdaBoost using decision trees.
3. Random Forest with feature subsetting at every split.
4. A hierarchical classifier consisting of a family router and family-specific species experts.
5. Gaussian Naive Bayes for flat species classification.

All models use a common train/test split to make their results comparable.

## Algorithms Implemented

| Part | Algorithm | Key concepts |
|---|---|---|
| 1 | Decision Tree | Gini impurity, entropy, information gain, weighted splitting |
| 2 | Bagging | Bootstrap sampling, majority voting |
| 2 | AdaBoost | Multiclass SAMME, sample-weight updates |
| 3 | Random Forest | Bootstrap sampling, random feature subsets at each node |
| 4 | Hierarchical Decision Tree | Family router, family-specific species experts |
| 5 | Gaussian Naive Bayes | Gaussian likelihoods, class priors, log probabilities |

### 1. Single Decision Tree

The decision tree is implemented from scratch and supports both Gini impurity and entropy-based splitting.

Features include:
- Binary threshold splits.
- Vectorised evaluation of candidate thresholds.
- Optional sample weights.
- Configurable maximum depth and minimum samples required to split.
- Optional feature subsetting.
- Deterministic tie-breaking and majority-class prediction.

The implementation serves as the base estimator for the ensemble methods.

### 2. Bagging and AdaBoost

**Bagging** trains multiple decision trees on bootstrap samples and combines their predictions through majority voting.

**AdaBoost** implements the multiclass SAMME algorithm, using shallow decision trees as weak learners. Sample weights are updated between boosting rounds to give greater importance to misclassified training examples.

Both methods are evaluated with Gini and entropy criteria.

### 3. Random Forest

The Random Forest combines bootstrap sampling with random feature selection at every tree node.

Instead of considering all 22 features at each split, each node considers a random subset of four features. Predictions are combined using majority voting across the forest.

### 4. Hierarchical Decision Tree

The hierarchical classifier separates prediction into two stages:

1. **Router:** predicts the family of the input sample.
2. **Expert:** predicts the species using the expert associated with the predicted family.

Each expert is trained only on samples belonging to its corresponding family in the training set.

The implementation evaluates router accuracy, overall species accuracy, oracle accuracy, failure fractions, confusion matrices, and per-family accuracy.

### 5. Gaussian Naive Bayes

Gaussian Naive Bayes predicts species using class priors and Gaussian feature likelihoods.

The implementation:
- Estimates class-wise means and variances from training data.
- Applies variance smoothing for numerical stability.
- Calculates probabilities in log space to avoid numerical underflow.
- Supports prediction using NumPy operations.

## Dataset

The assignment uses the [Anuran Calls (MFCCs) dataset](https://archive.ics.uci.edu/dataset/406/anuran+calls+mfccs) from the UCI Machine Learning Repository.

The dataset contains 7,195 records with 22 numeric Mel-Frequency Cepstral Coefficient (MFCC) features and the following labels:

| Column | Description | Number of classes |
|---|---|---:|
| `MFCCs_1` – `MFCCs_22` | Input features | — |
| `Family` | Frog family | 4 |
| `Genus` | Frog genus | 8 |
| `Species` | Frog species | 10 |
| `RecordID` | Recording identifier | — |

Only the 22 MFCC columns are used as model inputs. `Family` and `Species` provide the target labels for the respective tasks. `Genus` and `RecordID` are not used as features.

### Dataset Setup

Download `Frogs_MFCCs.csv` from the UCI dataset linked above and place it in the repository root, alongside `common.py` and the five part scripts.

The expected path is:

```text
ML-Assignment/
├── Frogs_MFCCs.csv
├── common.py
├── part1_decision_tree.py
├── part2_bagging_boosting.py
├── part3_random_forest.py
├── part4_hierarchical.py
└── part5_naive_bayes.py
```

The dataset is not generated by the code. It must be available locally before running the experiments.

## Repository Structure

```text
ML-Assignment/
├── common.py
├── part1_decision_tree.py
├── part2_bagging_boosting.py
├── part3_random_forest.py
├── part4_hierarchical.py
├── part5_naive_bayes.py
├── pyproject.toml
├── uv.lock
├── .gitignore
└── Frogs_MFCCs.csv          # Download separately
```

| File | Responsibility |
|---|---|
| `common.py` | Dataset loading, stratified splitting, accuracy calculation, and shared utilities |
| `part1_decision_tree.py` | Decision tree implementation and flat classification experiments |
| `part2_bagging_boosting.py` | Bagging and multiclass AdaBoost |
| `part3_random_forest.py` | Random Forest |
| `part4_hierarchical.py` | Hierarchical router-and-experts classifier |
| `part5_naive_bayes.py` | Gaussian Naive Bayes |
| `pyproject.toml` | Project metadata and dependency declarations |
| `uv.lock` | Locked dependency versions for reproducible environment setup |

## Requirements

- Python 3.12 or later.
- NumPy.
- pandas.
- Graphviz Python package for the tree visualisation utility.
- Matplotlib for optional plots.

The project metadata also includes development tools such as Ruff and mypy.

**No machine learning libraries are used to implement the classifiers.** Libraries such as scikit-learn, PyTorch, TensorFlow, SciPy, XGBoost, and LightGBM are prohibited by the assignment.

## Installation

### Option 1: Using uv (recommended)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available.

Clone the repository and enter its directory:

```bash
git clone https://github.com/Manoj-3636/ML-Assignment.git
cd ML-Assignment
```

Create the project environment and install the dependencies declared in `pyproject.toml` and `uv.lock`:

```bash
uv sync
```

### Option 2: Using pip

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install numpy pandas matplotlib graphviz
```

If tree visualisation is needed, install the Graphviz system application as well as its Python package.

## Running the Experiments

Ensure that `Frogs_MFCCs.csv` is present in the repository root before running any script.

Run each part independently from the repository root.

### Part 1 — Decision Tree

```bash
python part1_decision_tree.py
```

Using uv:

```bash
uv run python part1_decision_tree.py
```

Evaluates Gini and entropy decision trees and prints training and test accuracies. It also evaluates multiple maximum-depth settings to illustrate bias and variance.

### Part 2 — Bagging and AdaBoost

```bash
python part2_bagging_boosting.py
```

Using uv:

```bash
uv run python part2_bagging_boosting.py
```

Evaluates bagging and multiclass AdaBoost with both splitting criteria.

### Part 3 — Random Forest

```bash
python part3_random_forest.py
```

Using uv:

```bash
uv run python part3_random_forest.py
```

Evaluates Random Forest models using Gini impurity and entropy.

### Part 4 — Hierarchical Decision Tree

```bash
python part4_hierarchical.py
```

Using uv:

```bash
uv run python part4_hierarchical.py
```

Evaluates the family router and species experts, including confusion matrices and the hierarchical failure analysis.

### Part 5 — Gaussian Naive Bayes

```bash
python part5_naive_bayes.py
```

Using uv:

```bash
uv run python part5_naive_bayes.py
```

Evaluates flat species classification using Gaussian Naive Bayes.

Each script can be executed independently. Run all five commands to obtain the complete set of experiment outputs.

## Implementation Details

### Common Train/Test Split

The shared utility in `common.py` performs a stratified 80/20 train/test split using species labels and a random seed of 42.

- The training set is used for fitting models.
- The test set is reserved for final evaluation.
- All five parts use the same split.
- The family labels are available for hierarchical classification and derived family-accuracy calculations.

Using a common split allows meaningful comparisons between the different approaches.

### Fixed Hyperparameters

The experiments follow the assignment's prescribed settings:

| Method | Configuration |
|---|---|
| Single decision tree | Unconstrained maximum depth |
| Bagging | 25 trees, bootstrap sampling, all features |
| AdaBoost | 50 rounds, maximum tree depth of 3, multiclass SAMME |
| Random Forest | 50 trees, bootstrap sampling, 4 candidate features per split |
| Hierarchical classifier | Unconstrained router and family experts |
| Gaussian Naive Bayes | Variance smoothing of \(10^{-9}\) times the maximum training-feature variance |

The tree-based methods use both Gini and entropy wherever specified by the assignment.

### Reproducibility

The prescribed random seed is used for the common split and ensemble randomness. Deterministic tie-breaking is used where specified by the assignment.

## Evaluation

The primary metric is classification accuracy.

The experiments report:

- Training species accuracy.
- Test species accuracy.
- Test family accuracy for flat classifiers, obtained by mapping predicted species to their corresponding families.
- Router accuracy for the hierarchical classifier.
- Hierarchical oracle accuracy and failure fractions.
- Router and species confusion matrices.
- Per-family test accuracy.

The results can be used to compare the single tree with bagging, boosting, Random Forest, hierarchical classification, and Gaussian Naive Bayes.

## Notes

- Run scripts from the repository root so that the dataset path resolves correctly.
- Do not use the test set for fitting, hyperparameter tuning, or training-data statistics.
- The assignment notes that calls from the same recording can appear in both training and test sets, potentially making the measured accuracy optimistic for unseen recordings.
- Model performance should be taken from the actual script outputs rather than assumed in advance.

---

**Course:** BITS F464 — Machine Learning  
**Assignment:** Assignment 1 — Flat vs. Hierarchical Classification From Scratch  
**Repository:** [Manoj-3636/ML-Assignment](https://github.com/Manoj-3636/ML-Assignment)
