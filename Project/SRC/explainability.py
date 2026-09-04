from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



# basic setup

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "Processed"
TABLES_DIR = PROJECT_ROOT / "Outputs" / "Tables"
FIGURES_DIR = PROJECT_ROOT / "Outputs" / "Figures"
MODELS_DIR = PROJECT_ROOT / "Outputs" / "Models"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"

# these need to match the names saved in train_models_weighted.py
RF_MODEL_PATH = MODELS_DIR / "random_forest_model_weighted.joblib"
GB_MODEL_PATH = MODELS_DIR / "gradient_boosting_model_weighted.joblib"



# outputs

LOGREG_COEF_CSV = TABLES_DIR / "logreg_interpretability_coefficients.csv"
RF_IMPORTANCE_CSV = TABLES_DIR / "rf_weighted_feature_importance.csv"
GB_IMPORTANCE_CSV = TABLES_DIR / "gb_weighted_feature_importance.csv"

LOGREG_COEF_PNG = FIGURES_DIR / "logreg_interpretability_coefficients_top15.png"
RF_IMPORTANCE_PNG = FIGURES_DIR / "rf_weighted_feature_importance_top15.png"
GB_IMPORTANCE_PNG = FIGURES_DIR / "gb_weighted_feature_importance_top15.png"

LOGREG_INTERP_MODEL_PATH = MODELS_DIR / "logreg_interpretability_model.joblib"



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


# helper for saving bar charts

def save_barh(df, value_col, label_col, title, save_path, top_n=15, use_abs=False):
    plot_df = df.copy()

    if use_abs:
        plot_df = plot_df.sort_values("AbsCoefficient", ascending=False)
    else:
        plot_df = plot_df.sort_values(value_col, ascending=False)

    plot_df = plot_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(plot_df[label_col], plot_df[value_col])
    plt.title(title)
    plt.xlabel(value_col.replace("_", " "))
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# load training data

X_train = pd.read_csv(X_TRAIN_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")

missing_cols = [col for col in feature_cols if col not in X_train.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns in X_train: {missing_cols}")

X_train = X_train[feature_cols].copy()

print("Loaded training data for explainability work.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print()



# train a separate logistic regression for coefficient comparison

# here I scale all features on purpose so the coefficient sizes are easier to compare
logreg_interp_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ))
])

logreg_interp_model.fit(X_train, y_train)

logreg_coef = logreg_interp_model.named_steps["model"].coef_[0]

logreg_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": logreg_coef
})

logreg_df["AbsCoefficient"] = logreg_df["Coefficient"].abs()
logreg_df["Direction"] = np.where(
    logreg_df["Coefficient"] > 0, "Positive",
    np.where(logreg_df["Coefficient"] < 0, "Negative", "Zero")
)
logreg_df["LogRegRank"] = (
    logreg_df["AbsCoefficient"]
    .rank(ascending=False, method="dense")
    .astype(int)
)

logreg_df = logreg_df.sort_values("AbsCoefficient", ascending=False).reset_index(drop=True)
logreg_df.to_csv(LOGREG_COEF_CSV, index=False)

joblib.dump(logreg_interp_model, LOGREG_INTERP_MODEL_PATH)

save_barh(
    logreg_df,
    value_col="Coefficient",
    label_col="Feature",
    title="Logistic Regression Coefficients (Top 15 by Absolute Size)",
    save_path=LOGREG_COEF_PNG,
    top_n=15,
    use_abs=True
)

print("Saved logistic regression coefficient output.")
print(LOGREG_COEF_CSV)
print(LOGREG_COEF_PNG)
print()


# load the weighted random forest and save feature importance

if not RF_MODEL_PATH.exists():
    raise FileNotFoundError(f"Could not find weighted Random Forest model: {RF_MODEL_PATH}")

rf_model = joblib.load(RF_MODEL_PATH)

if len(rf_model.feature_importances_) != len(feature_cols):
    raise ValueError("Random Forest feature count does not match feature_cols.")

rf_df = pd.DataFrame({
    "Feature": feature_cols,
    "RFImportance": rf_model.feature_importances_
})

rf_df["RFRank"] = (
    rf_df["RFImportance"]
    .rank(ascending=False, method="dense")
    .astype(int)
)

rf_df = rf_df.sort_values("RFImportance", ascending=False).reset_index(drop=True)
rf_df.to_csv(RF_IMPORTANCE_CSV, index=False)

save_barh(
    rf_df,
    value_col="RFImportance",
    label_col="Feature",
    title="Random Forest Feature Importance (Top 15)",
    save_path=RF_IMPORTANCE_PNG,
    top_n=15
)

print("Saved Random Forest feature importance output.")
print(RF_IMPORTANCE_CSV)
print(RF_IMPORTANCE_PNG)
print()



# load the weighted gradient boosting model and save feature importance

if not GB_MODEL_PATH.exists():
    raise FileNotFoundError(f"Could not find weighted Gradient Boosting model: {GB_MODEL_PATH}")

gb_model = joblib.load(GB_MODEL_PATH)

if len(gb_model.feature_importances_) != len(feature_cols):
    raise ValueError("Gradient Boosting feature count does not match feature_cols.")

gb_df = pd.DataFrame({
    "Feature": feature_cols,
    "GBImportance": gb_model.feature_importances_
})

gb_df["GBRank"] = (
    gb_df["GBImportance"]
    .rank(ascending=False, method="dense")
    .astype(int)
)

gb_df = gb_df.sort_values("GBImportance", ascending=False).reset_index(drop=True)
gb_df.to_csv(GB_IMPORTANCE_CSV, index=False)

save_barh(
    gb_df,
    value_col="GBImportance",
    label_col="Feature",
    title="Gradient Boosting Feature Importance (Top 15)",
    save_path=GB_IMPORTANCE_PNG,
    top_n=15
)

print("Saved Gradient Boosting feature importance output.")
print(GB_IMPORTANCE_CSV)
print(GB_IMPORTANCE_PNG)
print()

print("Done.")