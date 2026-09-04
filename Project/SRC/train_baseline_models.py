from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# file paths
X_train_path = Path("Project/Data/Processed/X_train.csv")
X_test_path = Path("Project/Data/Processed/X_test.csv")
y_train_path = Path("Project/Data/Processed/y_train.csv")
y_test_path = Path("Project/Data/Processed/y_test.csv")

output_dir = Path("Project/Outputs")
tables_dir = output_dir / "Tables"
models_dir = output_dir / "Models"

tables_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

# these should match what I used in preprocessing
binary_cols = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "DiffWalk", "Sex"
]

ordinal_cols = [
    "GenHlth", "Age", "Education", "Income"
]

numeric_cols = [
    "BMI", "MentHlth", "PhysHlth"
]


# load the saved train/test splits
X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)

# squeeze just turns the one-column csv into a Series
y_train = pd.read_csv(y_train_path).squeeze("columns")
y_test = pd.read_csv(y_test_path).squeeze("columns")

print("Loaded train/test splits.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print()

# logistic regression usually benefits from scaling
logreg_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols)
    ],
    remainder="passthrough"
)

X_train_logreg = logreg_preprocessor.fit_transform(X_train)
X_test_logreg = logreg_preprocessor.transform(X_test)

print("Logistic regression preprocessing complete.")
print("Scaled X_train shape:", X_train_logreg.shape)
print("Scaled X_test shape:", X_test_logreg.shape)
print()

# tree model can just use the original values
X_train_tree = X_train.copy()
X_test_tree = X_test.copy()

# logistic regression
logreg_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logreg_model.fit(X_train_logreg, y_train)

logreg_pred = logreg_model.predict(X_test_logreg)
logreg_prob = logreg_model.predict_proba(X_test_logreg)[:, 1]

# decision tree
# keeping it a bit constrained so it does not go completely wild
tree_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

tree_model.fit(X_train_tree, y_train)

tree_pred = tree_model.predict(X_test_tree)
tree_prob = tree_model.predict_proba(X_test_tree)[:, 1]


def evaluate_model(model_name, y_true, y_pred, y_prob):
    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob)
    }

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    print(f"----- {model_name} -----")
    for metric, value in results.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)
    print()

    return results, cm, report


# evaluate both baseline models
logreg_results, logreg_cm, logreg_report = evaluate_model(
    "Logistic Regression", y_test, logreg_pred, logreg_prob
)

tree_results, tree_cm, tree_report = evaluate_model(
    "Decision Tree", y_test, tree_pred, tree_prob
)

# save the main results table
results_df = pd.DataFrame([logreg_results, tree_results])
results_df.to_csv(tables_dir / "baseline_results.csv", index=False)

# save confusion matrices too
pd.DataFrame(logreg_cm).to_csv(tables_dir / "logreg_confusion_matrix.csv", index=False)
pd.DataFrame(tree_cm).to_csv(tables_dir / "tree_confusion_matrix.csv", index=False)

# save classification reports as text files
with open(tables_dir / "logreg_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(logreg_report)

with open(tables_dir / "tree_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(tree_report)

# save the trained models and the preprocessor as well
joblib.dump(logreg_model, models_dir / "logreg_model.joblib")
joblib.dump(tree_model, models_dir / "decision_tree_model.joblib")
joblib.dump(logreg_preprocessor, models_dir / "logreg_preprocessor.joblib")

print("Saved outputs to:", tables_dir, "and", models_dir)
print(results_df)