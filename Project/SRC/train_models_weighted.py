from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.utils.class_weight import compute_sample_weight



# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "Processed"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
TABLES_DIR = OUTPUTS_DIR / "Tables"
MODELS_DIR = OUTPUTS_DIR / "Models"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"



# columns
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

feature_cols = binary_cols + ordinal_cols + numeric_cols



# helper
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


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# load data
X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")
y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

missing_train = [col for col in feature_cols if col not in X_train.columns]
missing_test = [col for col in feature_cols if col not in X_test.columns]

if missing_train:
    raise ValueError(f"Missing expected columns in X_train: {missing_train}")

if missing_test:
    raise ValueError(f"Missing expected columns in X_test: {missing_test}")

# keep the order tidy and consistent
X_train = X_train[feature_cols].copy()
X_test = X_test[feature_cols].copy()

print("Loaded train/test splits.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print()

# just using the raw tree data directly
X_train_tree = X_train.copy()
X_test_tree = X_test.copy()

# logistic regression gets scaling on numeric columns only
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



# logistic regression
logreg_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

logreg_model.fit(X_train_logreg, y_train)

logreg_pred = logreg_model.predict(X_test_logreg)
logreg_prob = logreg_model.predict_proba(X_test_logreg)[:, 1]

logreg_results, logreg_cm, logreg_report = evaluate_model(
    "Logistic Regression (Weighted)",
    y_test,
    logreg_pred,
    logreg_prob
)


# decision tree
tree_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    class_weight="balanced"
)

tree_model.fit(X_train_tree, y_train)

tree_pred = tree_model.predict(X_test_tree)
tree_prob = tree_model.predict_proba(X_test_tree)[:, 1]

tree_results, tree_cm, tree_report = evaluate_model(
    "Decision Tree (Weighted)",
    y_test,
    tree_pred,
    tree_prob
)



# random forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf_model.fit(X_train_tree, y_train)

rf_pred = rf_model.predict(X_test_tree)
rf_prob = rf_model.predict_proba(X_test_tree)[:, 1]

rf_results, rf_cm, rf_report = evaluate_model(
    "Random Forest (Weighted)",
    y_test,
    rf_pred,
    rf_prob
)



# gradient boosting
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

gb_model.fit(X_train_tree, y_train, sample_weight=gb_sample_weight)

gb_pred = gb_model.predict(X_test_tree)
gb_prob = gb_model.predict_proba(X_test_tree)[:, 1]

gb_results, gb_cm, gb_report = evaluate_model(
    "Gradient Boosting (Weighted)",
    y_test,
    gb_pred,
    gb_prob
)



# save all results together
results_df = pd.DataFrame([
    logreg_results,
    tree_results,
    rf_results,
    gb_results
])

results_df.to_csv(TABLES_DIR / "all_model_results_weighted.csv", index=False)

# confusion matrices
pd.DataFrame(logreg_cm).to_csv(TABLES_DIR / "logreg_confusion_matrix_weighted.csv", index=False)
pd.DataFrame(tree_cm).to_csv(TABLES_DIR / "tree_confusion_matrix_weighted.csv", index=False)
pd.DataFrame(rf_cm).to_csv(TABLES_DIR / "rf_confusion_matrix_weighted.csv", index=False)
pd.DataFrame(gb_cm).to_csv(TABLES_DIR / "gb_confusion_matrix_weighted.csv", index=False)

# classification reports
save_text(TABLES_DIR / "logreg_classification_report_weighted.txt", logreg_report)
save_text(TABLES_DIR / "tree_classification_report_weighted.txt", tree_report)
save_text(TABLES_DIR / "rf_classification_report_weighted.txt", rf_report)
save_text(TABLES_DIR / "gb_classification_report_weighted.txt", gb_report)

# trained models
joblib.dump(logreg_model, MODELS_DIR / "logreg_model_weighted.joblib")
joblib.dump(tree_model, MODELS_DIR / "decision_tree_model_weighted.joblib")
joblib.dump(rf_model, MODELS_DIR / "random_forest_model_weighted.joblib")
joblib.dump(gb_model, MODELS_DIR / "gradient_boosting_model_weighted.joblib")
joblib.dump(logreg_preprocessor, MODELS_DIR / "logreg_preprocessor_weighted.joblib")

print("Saved outputs to:", TABLES_DIR, "and", MODELS_DIR)
print()
print("Full weighted comparison table:")
print(results_df.sort_values(by="F1", ascending=False))