Installation

Option 1: Using uv (recommended)

Install uv if it is not already available.

Clone the repository and enter its directory:

git clone https://github.com/Manoj-3636/ML-Assignment.git
cd ML-Assignment

Create the project environment and install the dependencies declared in pyproject.toml and uv.lock:

uv sync

Option 2: Using pip

Create and activate a virtual environment:

python -m venv .venv

On Linux or macOS:

source .venv/bin/activate

On Windows PowerShell:

.venv\Scripts\Activate.ps1

Install the required packages:

python -m pip install numpy pandas matplotlib

Running the Experiments

Ensure that Frogs_MFCCs.csv is present in the repository root before running any script.

Run each part independently from the repository root.

Part 1 — Decision Tree

Run using Python:

python part1_decision_tree.py

Using uv:

uv run python part1_decision_tree.py

Evaluates Gini and entropy decision trees and prints training and test accuracies. It also evaluates multiple maximum-depth settings to illustrate bias and variance.

Part 2 — Bagging and AdaBoost

Run using Python:

python part2_bagging_boosting.py

Using uv:

uv run python part2_bagging_boosting.py

Evaluates bagging and multiclass AdaBoost with both splitting criteria.

Part 3 — Random Forest

Run using Python:

python part3_random_forest.py

Using uv:

uv run python part3_random_forest.py

Evaluates Random Forest models using Gini impurity and entropy.

Part 4 — Hierarchical Decision Tree

Run using Python:

python part4_hierarchical.py

Using uv:

uv run python part4_hierarchical.py

Evaluates the family router and species experts, including confusion matrices and the hierarchical failure analysis.

Part 5 — Gaussian Naive Bayes

Run using Python:

python part5_naive_bayes.py

Using uv:

uv run python part5_naive_bayes.py

Evaluates flat species classification using Gaussian Naive Bayes.

Each script can be executed independently. Run all five commands to obtain the complete set of experiment outputs.