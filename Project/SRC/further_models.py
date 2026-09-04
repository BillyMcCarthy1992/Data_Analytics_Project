from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "Processed"
TABLES_DIR = PROJECT_ROOT / "Outputs" / "Tables"
MODELS_DIR = PROJECT_ROOT / "Outputs" / "Models"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"

BASELINE_RESULTS_PATH = TABLES_DIR / "baseline_results.csv"

# load the saved train/test splits
X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")
y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

print("Loaded train/test splits.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print()

# tree models can just use the saved data as is
X_train_tree = X_train.copy()
X_test_tree = X_test.copy()


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


# random forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_tree, y_train)

rf_pred = rf_model.predict(X_test_tree)
rf_prob = rf_model.predict_proba(X_test_tree)[:, 1]

rf_results, rf_cm, rf_report = evaluate_model(
    "Random Forest", y_test, rf_pred, rf_prob
)



gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train_tree, y_train)

gb_pred = gb_model.predict(X_test_tree)
gb_prob = gb_model.predict_proba(X_test_tree)[:, 1]

gb_results, gb_cm, gb_report = evaluate_model(
    "Gradient Boosting", y_test, gb_pred, gb_prob
)

# save the results
new_results_df = pd.DataFrame([rf_results, gb_results])

if BASELINE_RESULTS_PATH.exists():
    old_results_df = pd.read_csv(BASELINE_RESULTS_PATH)
    all_results_df = pd.concat([old_results_df, new_results_df], ignore_index=True)
else:
    all_results_df = new_results_df.copy()

all_results_df.to_csv(TABLES_DIR / "all_model_results.csv", index=False)

pd.DataFrame(rf_cm).to_csv(TABLES_DIR / "rf_confusion_matrix.csv", index=False)
pd.DataFrame(gb_cm).to_csv(TABLES_DIR / "gb_confusion_matrix.csv", index=False)

with open(TABLES_DIR / "rf_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(rf_report)

with open(TABLES_DIR / "gb_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(gb_report)

# save the trained models too so I can reuse them later
joblib.dump(rf_model, MODELS_DIR / "random_forest_model.joblib")
joblib.dump(gb_model, MODELS_DIR / "gradient_boosting_model.joblib")

print("Saved outputs to:", TABLES_DIR, "and", MODELS_DIR)
print()
print("Full comparison table:")
print(all_results_df)