from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore", category=ConvergenceWarning)

# basic settings
RANDOM_STATE = 42
CV_FOLDS = 3
REFIT_METRIC = "f1"

# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "Processed"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
TABLES_DIR = OUTPUTS_DIR / "Tables"
MODELS_DIR = OUTPUTS_DIR / "Models"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"

# should match the earlier preprocessing file
binary_cols = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "DiffWalk", "Sex"
]

ordinal_cols = ["GenHlth", "Age", "Education", "Income"]
numeric_cols = ["BMI", "MentHlth", "PhysHlth"]

feature_cols = binary_cols + ordinal_cols + numeric_cols

# load the train/test splits
X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")
y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

print("Loaded train/test splits")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print()

missing_train = [col for col in feature_cols if col not in X_train.columns]
missing_test = [col for col in feature_cols if col not in X_test.columns]

if missing_train:
    raise ValueError(f"Missing expected columns in X_train: {missing_train}")

if missing_test:
    raise ValueError(f"Missing expected columns in X_test: {missing_test}")

# just making sure the column order stays consistent
X_train = X_train[feature_cols].copy()
X_test = X_test[feature_cols].copy()

cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def evaluate_test_set(model, X_eval, y_true):
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob)
    }

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    return metrics, cm, report


def run_search(model_name, search, X_fit, y_fit, X_eval, y_eval, fit_params=None):
    print(f"===== Tuning {model_name} =====")

    if fit_params is None:
        fit_params = {}

    search.fit(X_fit, y_fit, **fit_params)

    best_model = search.best_estimator_
    best_index = search.best_index_

    cv_metrics = {
        "CV_Accuracy": search.cv_results_["mean_test_accuracy"][best_index],
        "CV_Precision": search.cv_results_["mean_test_precision"][best_index],
        "CV_Recall": search.cv_results_["mean_test_recall"][best_index],
        "CV_F1": search.cv_results_["mean_test_f1"][best_index],
        "CV_ROC_AUC": search.cv_results_["mean_test_roc_auc"][best_index]
    }

    test_metrics, cm, report = evaluate_test_set(best_model, X_eval, y_eval)

    print("Best params:")
    print(search.best_params_)
    print()

    print("CV metrics:")
    for k, v in cv_metrics.items():
        print(f"{k}: {v:.4f}")
    print()

    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    print()

    print("Confusion Matrix:")
    print(cm)
    print()

    print("Classification Report:")
    print(report)
    print()

    result_row = {
        "Model": model_name,
        **cv_metrics,
        **test_metrics
    }

    return best_model, search.best_params_, result_row, cm, report


def try_load_existing_baselines():
    candidate_paths = [
        TABLES_DIR / "all_model_results.csv",
        TABLES_DIR / "baseline_results.csv",
        OUTPUTS_DIR / "all_model_results.csv",
        OUTPUTS_DIR / "baseline_results.csv"
    ]

    for path in candidate_paths:
        if path.exists():
            return pd.read_csv(path), path

    return None, None


# logistic regression
# only scaling the numeric columns here
logreg_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols)
    ],
    remainder="passthrough"
)

logreg_pipeline = Pipeline([
    ("preprocessor", logreg_preprocessor),
    ("model", LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])

logreg_param_dist = {
    "model__C": np.logspace(-2, 1, 6),
    "model__penalty": ["l1", "l2"]
}

logreg_search = RandomizedSearchCV(
    estimator=logreg_pipeline,
    param_distributions=logreg_param_dist,
    n_iter=6,
    scoring=scoring,
    refit=REFIT_METRIC,
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# decision tree
tree_model = DecisionTreeClassifier(random_state=RANDOM_STATE)

tree_param_dist = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [3, 5, 8, 12, None],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 5, 10]
}

tree_search = RandomizedSearchCV(
    estimator=tree_model,
    param_distributions=tree_param_dist,
    n_iter=8,
    scoring=scoring,
    refit=REFIT_METRIC,
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# random forest takes the longest for me, so keeping this search a bit smaller
# also using one core inside the model since the search itself is already parallel
rf_model = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=1
)

rf_param_dist = {
    "n_estimators": [100, 150, 200],
    "max_depth": [10, 15, None],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt"]
}

rf_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_param_dist,
    n_iter=6,
    scoring=scoring,
    refit=REFIT_METRIC,
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# gradient boosting can get slow too, so keeping this one fairly tight as well
gb_model = GradientBoostingClassifier(random_state=RANDOM_STATE)

gb_param_dist = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3],
    "subsample": [0.8, 1.0],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 5]
}

gb_search = RandomizedSearchCV(
    estimator=gb_model,
    param_distributions=gb_param_dist,
    n_iter=6,
    scoring=scoring,
    refit=REFIT_METRIC,
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# run all the tuning
all_results = []
all_best_params = {}

best_logreg, best_logreg_params, logreg_row, logreg_cm, logreg_report = run_search(
    model_name="Logistic Regression (Tuned, F1)",
    search=logreg_search,
    X_fit=X_train,
    y_fit=y_train,
    X_eval=X_test,
    y_eval=y_test
)
all_results.append(logreg_row)
all_best_params["Logistic Regression (Tuned, F1)"] = best_logreg_params

best_tree, best_tree_params, tree_row, tree_cm, tree_report = run_search(
    model_name="Decision Tree (Tuned, F1)",
    search=tree_search,
    X_fit=X_train,
    y_fit=y_train,
    X_eval=X_test,
    y_eval=y_test
)
all_results.append(tree_row)
all_best_params["Decision Tree (Tuned, F1)"] = best_tree_params

best_rf, best_rf_params, rf_row, rf_cm, rf_report = run_search(
    model_name="Random Forest (Tuned, F1)",
    search=rf_search,
    X_fit=X_train,
    y_fit=y_train,
    X_eval=X_test,
    y_eval=y_test
)
all_results.append(rf_row)
all_best_params["Random Forest (Tuned, F1)"] = best_rf_params

best_gb, best_gb_params, gb_row, gb_cm, gb_report = run_search(
    model_name="Gradient Boosting (Tuned, F1)",
    search=gb_search,
    X_fit=X_train,
    y_fit=y_train,
    X_eval=X_test,
    y_eval=y_test
)
all_results.append(gb_row)
all_best_params["Gradient Boosting (Tuned, F1)"] = best_gb_params

# save the tuned results so I can use them later
tuned_results_df = pd.DataFrame(all_results)
tuned_results_path = TABLES_DIR / "tuned_F1_model_results.csv"
tuned_results_df.to_csv(tuned_results_path, index=False)

save_text(TABLES_DIR / "logreg_tuned_F1_classification_report.txt", logreg_report)
save_text(TABLES_DIR / "tree_tuned_F1_classification_report.txt", tree_report)
save_text(TABLES_DIR / "rf_tuned_F1_classification_report.txt", rf_report)
save_text(TABLES_DIR / "gb_tuned_F1_classification_report.txt", gb_report)

pd.DataFrame(logreg_cm).to_csv(TABLES_DIR / "logreg_tuned_F1_confusion_matrix.csv", index=False)
pd.DataFrame(tree_cm).to_csv(TABLES_DIR / "tree_tuned_F1_confusion_matrix.csv", index=False)
pd.DataFrame(rf_cm).to_csv(TABLES_DIR / "rf_tuned_F1_confusion_matrix.csv", index=False)
pd.DataFrame(gb_cm).to_csv(TABLES_DIR / "gb_tuned_F1_confusion_matrix.csv", index=False)

# save the tuned models too
joblib.dump(best_logreg, MODELS_DIR / "logreg_tuned_F1_model.joblib")
joblib.dump(best_tree, MODELS_DIR / "decision_tree_tuned_F1_model.joblib")
joblib.dump(best_rf, MODELS_DIR / "random_forest_tuned_F1_model.joblib")
joblib.dump(best_gb, MODELS_DIR / "gradient_boosting_tuned_F1_model.joblib")

with open(TABLES_DIR / "tuned_best_params_F1.json", "w", encoding="utf-8") as f:
    json.dump(all_best_params, f, indent=2)

# combine with earlier baseline results if they exist
existing_df, existing_path = try_load_existing_baselines()

if existing_df is not None:
    combined_df = pd.concat([existing_df, tuned_results_df], ignore_index=True)
    combined_df.to_csv(TABLES_DIR / "baseline_and_tuned_F1_results.csv", index=False)

    print(f"Loaded existing results from: {existing_path}")
    print("Saved combined baseline+tuned table to:")
    print(TABLES_DIR / "baseline_and_tuned_F1_results.csv")
    print()

print("Saved tuned results to:")
print(tuned_results_path)
print()

print("Tuned model summary:")
print(tuned_results_df.sort_values(by="F1", ascending=False))