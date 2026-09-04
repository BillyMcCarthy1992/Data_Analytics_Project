from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

try:
    import shap
except ImportError as e:
    raise ImportError(
        "The shap package is not installed. Install it before running this file."
    ) from e



# basic setup

RANDOM_STATE = 42
SHAP_SAMPLE_SIZE = 2000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "Processed"
TABLES_DIR = PROJECT_ROOT / "Outputs" / "Tables"
FIGURES_DIR = PROJECT_ROOT / "Outputs" / "Figures"
MODELS_DIR = PROJECT_ROOT / "Outputs" / "Models"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"

# these are the weighted models saved in train_models_weighted.py
RF_MODEL_PATH = MODELS_DIR / "random_forest_model_weighted.joblib"
GB_MODEL_PATH = MODELS_DIR / "gradient_boosting_model_weighted.joblib"

RF_SHAP_CSV = TABLES_DIR / "rf_shap_importance.csv"
GB_SHAP_CSV = TABLES_DIR / "gb_shap_importance.csv"

RF_SHAP_BEESWARM = FIGURES_DIR / "rf_shap_beeswarm.png"
GB_SHAP_BEESWARM = FIGURES_DIR / "gb_shap_beeswarm.png"
RF_SHAP_BAR = FIGURES_DIR / "rf_shap_bar.png"
GB_SHAP_BAR = FIGURES_DIR / "gb_shap_bar.png"



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



# helpers

def get_stratified_sample(X, y, sample_size, random_state=42):
    # keep the class balance roughly similar to the full test set
    if sample_size >= len(X):
        return X.copy(), y.copy()

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=sample_size,
        stratify=y,
        random_state=random_state
    )
    return X_sample.copy(), y_sample.copy()


def unwrap_binary_shap_values(shap_values):
    # SHAP output shape can vary a bit depending on version and model type
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            return np.array(shap_values[1])
        return np.array(shap_values[0])

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        if shap_values.shape[2] == 2:
            return shap_values[:, :, 1]
        if shap_values.shape[0] == 2:
            return shap_values[1, :, :]

    return shap_values


def build_shap_importance_df(X_data, shap_array, model_prefix):
    mean_abs = np.abs(shap_array).mean(axis=0)

    rows = []
    for i, feature in enumerate(X_data.columns):
        feature_series = pd.Series(X_data.iloc[:, i])
        shap_series = pd.Series(shap_array[:, i])

        # just using a rough direction summary here
        corr = feature_series.corr(shap_series, method="spearman")

        if pd.isna(corr):
            direction = "Unclear"
        elif corr > 0:
            direction = "Positive"
        elif corr < 0:
            direction = "Negative"
        else:
            direction = "Unclear"

        rows.append({
            "Feature": feature,
            f"{model_prefix}SHAP": mean_abs[i],
            f"{model_prefix}SHAPDirectionCorr": corr,
            f"{model_prefix}SHAPDirection": direction
        })

    out_df = pd.DataFrame(rows)
    out_df[f"{model_prefix}SHAPRank"] = (
        out_df[f"{model_prefix}SHAP"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    out_df = out_df.sort_values(f"{model_prefix}SHAP", ascending=False).reset_index(drop=True)
    return out_df


def save_shap_plots(model_name, shap_array, X_data, beeswarm_path, bar_path):
    # beeswarm plot
    shap.summary_plot(shap_array, X_data, show=False, max_display=15)
    plt.title(f"{model_name} SHAP Beeswarm")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # bar plot
    shap.summary_plot(shap_array, X_data, plot_type="bar", show=False, max_display=15)
    plt.title(f"{model_name} SHAP Importance")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_shap_array(model, X_data):
    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X_data, check_additivity=False)
    shap_array = unwrap_binary_shap_values(shap_raw)

    shap_array = np.array(shap_array)

    if shap_array.ndim != 2:
        raise ValueError(f"Unexpected SHAP array shape: {shap_array.shape}")

    if shap_array.shape[1] != X_data.shape[1]:
        raise ValueError(
            f"SHAP output has {shap_array.shape[1]} features, "
            f"but X_data has {X_data.shape[1]}"
        )

    return shap_array



# load test data

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

missing_cols = [col for col in feature_cols if col not in X_test.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns in X_test: {missing_cols}")

X_test = X_test[feature_cols].copy()

# only using y_test here so the sample keeps the same class balance
X_sample, _ = get_stratified_sample(
    X_test,
    y_test,
    sample_size=SHAP_SAMPLE_SIZE,
    random_state=RANDOM_STATE
)

print("Loaded test data for SHAP work.")
print("Full X_test shape:", X_test.shape)
print("SHAP sample shape:", X_sample.shape)
print()



# random forest SHAP
if not RF_MODEL_PATH.exists():
    raise FileNotFoundError(f"Could not find weighted Random Forest model: {RF_MODEL_PATH}")

rf_model = joblib.load(RF_MODEL_PATH)
rf_shap_array = get_shap_array(rf_model, X_sample)

rf_shap_df = build_shap_importance_df(X_sample, rf_shap_array, model_prefix="RF")
rf_shap_df.to_csv(RF_SHAP_CSV, index=False)

save_shap_plots(
    model_name="Random Forest",
    shap_array=rf_shap_array,
    X_data=X_sample,
    beeswarm_path=RF_SHAP_BEESWARM,
    bar_path=RF_SHAP_BAR
)

print("Saved Random Forest SHAP outputs.")
print(RF_SHAP_CSV)
print(RF_SHAP_BEESWARM)
print(RF_SHAP_BAR)
print()



# gradient boosting SHAP
if not GB_MODEL_PATH.exists():
    raise FileNotFoundError(f"Could not find weighted Gradient Boosting model: {GB_MODEL_PATH}")

gb_model = joblib.load(GB_MODEL_PATH)
gb_shap_array = get_shap_array(gb_model, X_sample)

gb_shap_df = build_shap_importance_df(X_sample, gb_shap_array, model_prefix="GB")
gb_shap_df.to_csv(GB_SHAP_CSV, index=False)

save_shap_plots(
    model_name="Gradient Boosting",
    shap_array=gb_shap_array,
    X_data=X_sample,
    beeswarm_path=GB_SHAP_BEESWARM,
    bar_path=GB_SHAP_BAR
)

print("Saved Gradient Boosting SHAP outputs.")
print(GB_SHAP_CSV)
print(GB_SHAP_BEESWARM)
print(GB_SHAP_BAR)
print()

print("Done.")