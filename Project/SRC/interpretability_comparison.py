from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "Outputs" / "Tables"
FIGURES_DIR = PROJECT_ROOT / "Outputs" / "Figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LOGREG_COEF_CSV = TABLES_DIR / "logreg_interpretability_coefficients.csv"
RF_IMPORTANCE_CSV = TABLES_DIR / "rf_weighted_feature_importance.csv"
GB_IMPORTANCE_CSV = TABLES_DIR / "gb_weighted_feature_importance.csv"
RF_SHAP_CSV = TABLES_DIR / "rf_shap_importance.csv"
GB_SHAP_CSV = TABLES_DIR / "gb_shap_importance.csv"

COMPARISON_CSV = TABLES_DIR / "interpretability_comparison_table.csv"
TOP10_TABLE_CSV = TABLES_DIR / "interpretability_top10_table.csv"
TOP10_OVERLAP_CSV = TABLES_DIR / "interpretability_top10_overlap.csv"
RANK_SIMILARITY_CSV = TABLES_DIR / "interpretability_rank_similarity.csv"
DIRECTIONAL_AGREEMENT_CSV = TABLES_DIR / "interpretability_directional_agreement.csv"
SUMMARY_TEXT = TABLES_DIR / "interpretability_summary.txt"

HEATMAP_PNG = FIGURES_DIR / "interpretability_comparison_heatmap.png"
TOP10_TABLE_PNG = FIGURES_DIR / "interpretability_top10_table.png"
AGREEMENT_PNG = FIGURES_DIR / "interpretability_top_feature_agreement.png"



# helpers
def normalise_by_max(series):
    max_val = series.max()
    if pd.isna(max_val) or max_val == 0:
        return series * 0
    return series / max_val


def save_table_png(df, save_path, title=None):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=13, pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def check_file_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find required input file: {path}")


def check_columns(df, required_cols, df_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {df_name}: {missing}")



# make sure the required files are there
for path in [
    LOGREG_COEF_CSV,
    RF_IMPORTANCE_CSV,
    GB_IMPORTANCE_CSV,
    RF_SHAP_CSV,
    GB_SHAP_CSV
]:
    check_file_exists(path)



# load outputs from earlier scripts
logreg_df = pd.read_csv(LOGREG_COEF_CSV)
rf_df = pd.read_csv(RF_IMPORTANCE_CSV)
gb_df = pd.read_csv(GB_IMPORTANCE_CSV)
rf_shap_df = pd.read_csv(RF_SHAP_CSV)
gb_shap_df = pd.read_csv(GB_SHAP_CSV)

check_columns(
    logreg_df,
    ["Feature", "Coefficient", "AbsCoefficient", "Direction", "LogRegRank"],
    "logreg_df"
)

check_columns(
    rf_df,
    ["Feature", "RFImportance", "RFRank"],
    "rf_df"
)

check_columns(
    gb_df,
    ["Feature", "GBImportance", "GBRank"],
    "gb_df"
)

check_columns(
    rf_shap_df,
    ["Feature", "RFSHAP", "RFSHAPRank", "RFSHAPDirectionCorr", "RFSHAPDirection"],
    "rf_shap_df"
)

check_columns(
    gb_shap_df,
    ["Feature", "GBSHAP", "GBSHAPRank", "GBSHAPDirectionCorr", "GBSHAPDirection"],
    "gb_shap_df"
)



# keep only what we need
logreg_keep = logreg_df[
    ["Feature", "Coefficient", "AbsCoefficient", "Direction", "LogRegRank"]
].rename(columns={
    "Direction": "LogRegDirection"
})

rf_keep = rf_df[
    ["Feature", "RFImportance", "RFRank"]
]

gb_keep = gb_df[
    ["Feature", "GBImportance", "GBRank"]
]

rf_shap_keep = rf_shap_df[
    ["Feature", "RFSHAP", "RFSHAPRank", "RFSHAPDirectionCorr", "RFSHAPDirection"]
]

gb_shap_keep = gb_shap_df[
    ["Feature", "GBSHAP", "GBSHAPRank", "GBSHAPDirectionCorr", "GBSHAPDirection"]
]



# merge into one comparison table
comparison_df = logreg_keep.merge(rf_keep, on="Feature", how="inner")
comparison_df = comparison_df.merge(gb_keep, on="Feature", how="inner")
comparison_df = comparison_df.merge(rf_shap_keep, on="Feature", how="inner")
comparison_df = comparison_df.merge(gb_shap_keep, on="Feature", how="inner")

comparison_df["LogRegNorm"] = normalise_by_max(comparison_df["AbsCoefficient"])
comparison_df["RFNorm"] = normalise_by_max(comparison_df["RFImportance"])
comparison_df["GBNorm"] = normalise_by_max(comparison_df["GBImportance"])
comparison_df["RFSHAPNorm"] = normalise_by_max(comparison_df["RFSHAP"])
comparison_df["GBSHAPNorm"] = normalise_by_max(comparison_df["GBSHAP"])

comparison_df["AverageNorm"] = comparison_df[
    ["LogRegNorm", "RFNorm", "GBNorm", "RFSHAPNorm", "GBSHAPNorm"]
].mean(axis=1)

comparison_df = comparison_df.sort_values("AverageNorm", ascending=False).reset_index(drop=True)
comparison_df.to_csv(COMPARISON_CSV, index=False)

print("Saved full comparison table:")
print(COMPARISON_CSV)
print()



# top 10 by each method
top10_df = pd.DataFrame({
    "LogReg Top 10": comparison_df.sort_values("LogRegRank")["Feature"].head(10).tolist(),
    "RF Importance Top 10": comparison_df.sort_values("RFRank")["Feature"].head(10).tolist(),
    "RF SHAP Top 10": comparison_df.sort_values("RFSHAPRank")["Feature"].head(10).tolist(),
    "GB Importance Top 10": comparison_df.sort_values("GBRank")["Feature"].head(10).tolist(),
    "GB SHAP Top 10": comparison_df.sort_values("GBSHAPRank")["Feature"].head(10).tolist()
})

top10_df.to_csv(TOP10_TABLE_CSV, index=False)

save_table_png(
    top10_df,
    TOP10_TABLE_PNG,
    title="Top 10 Features by Interpretability Method"
)

print("Saved top-10 table:")
print(TOP10_TABLE_CSV)
print(TOP10_TABLE_PNG)
print()



# top-10 overlap
top_lists = {
    "LogReg": set(top10_df["LogReg Top 10"]),
    "RFImportance": set(top10_df["RF Importance Top 10"]),
    "RFSHAP": set(top10_df["RF SHAP Top 10"]),
    "GBImportance": set(top10_df["GB Importance Top 10"]),
    "GBSHAP": set(top10_df["GB SHAP Top 10"])
}

overlap_rows = []
for feature in comparison_df["Feature"]:
    count = sum(feature in s for s in top_lists.values())
    avg_norm = comparison_df.loc[
        comparison_df["Feature"] == feature, "AverageNorm"
    ].iloc[0]

    overlap_rows.append({
        "Feature": feature,
        "Top10Count": count,
        "AverageNorm": avg_norm
    })

overlap_df = pd.DataFrame(overlap_rows)
overlap_df = overlap_df.sort_values(
    ["Top10Count", "AverageNorm"],
    ascending=[False, False]
).reset_index(drop=True)

overlap_df.to_csv(TOP10_OVERLAP_CSV, index=False)

plot_df = overlap_df.head(10).iloc[::-1]

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(plot_df["Feature"], plot_df["Top10Count"])
ax.set_title("Features Appearing Most Often in Top-10 Lists")
ax.set_xlabel("Number of Methods")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig(AGREEMENT_PNG, dpi=300, bbox_inches="tight")
plt.close()

print("Saved overlap summary:")
print(TOP10_OVERLAP_CSV)
print(AGREEMENT_PNG)
print()



# rank similarity
rank_cols = [
    "LogRegRank", "RFRank", "RFSHAPRank", "GBRank", "GBSHAPRank"
]

rank_similarity = comparison_df[rank_cols].corr(method="spearman")
rank_similarity.to_csv(RANK_SIMILARITY_CSV)

print("Saved rank similarity matrix:")
print(RANK_SIMILARITY_CSV)
print()



# directional agreement
# built-in tree feature importance has no direction
# so here I only compare logreg direction with SHAP direction
direction_df = comparison_df[
    ["Feature", "LogRegDirection", "RFSHAPDirection", "GBSHAPDirection"]
].copy()

direction_df["RF_Agrees_With_LogReg"] = np.where(
    (direction_df["LogRegDirection"].isin(["Positive", "Negative"])) &
    (direction_df["RFSHAPDirection"].isin(["Positive", "Negative"])),
    direction_df["LogRegDirection"] == direction_df["RFSHAPDirection"],
    np.nan
)

direction_df["GB_Agrees_With_LogReg"] = np.where(
    (direction_df["LogRegDirection"].isin(["Positive", "Negative"])) &
    (direction_df["GBSHAPDirection"].isin(["Positive", "Negative"])),
    direction_df["LogRegDirection"] == direction_df["GBSHAPDirection"],
    np.nan
)

direction_df.to_csv(DIRECTIONAL_AGREEMENT_CSV, index=False)

print("Saved directional agreement table:")
print(DIRECTIONAL_AGREEMENT_CSV)
print()


# comparison heatmap
heatmap_cols = ["LogRegNorm", "RFNorm", "RFSHAPNorm", "GBNorm", "GBSHAPNorm"]
heatmap_labels = ["LogReg", "RF Imp.", "RF SHAP", "GB Imp.", "GB SHAP"]

heatmap_df = comparison_df[["Feature", "AverageNorm"] + heatmap_cols].copy()
heatmap_df = heatmap_df.sort_values("AverageNorm", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8.5, 8))
im = ax.imshow(heatmap_df[heatmap_cols].to_numpy(), aspect="auto")

ax.set_xticks(np.arange(len(heatmap_labels)))
ax.set_xticklabels(heatmap_labels)
ax.set_yticks(np.arange(len(heatmap_df)))
ax.set_yticklabels(heatmap_df["Feature"])
ax.set_title("Normalised Feature Importance Across Methods")
ax.set_xlabel("Method")
ax.set_ylabel("Feature")

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(HEATMAP_PNG, dpi=300, bbox_inches="tight")
plt.close()

print("Saved heatmap:")
print(HEATMAP_PNG)
print()


# quick summary text
rf_agreement_rate = pd.Series(direction_df["RF_Agrees_With_LogReg"]).dropna().mean()
gb_agreement_rate = pd.Series(direction_df["GB_Agrees_With_LogReg"]).dropna().mean()

summary_lines = [
    "Interpretability comparison summary",
    "---------------------------------",
    "",
    "This file compares five explanation views:",
    "- Logistic Regression absolute coefficients",
    "- Random Forest built-in feature importance",
    "- Random Forest SHAP importance",
    "- Gradient Boosting built-in feature importance",
    "- Gradient Boosting SHAP importance",
    "",
    "Top features appearing most often across methods:"
]

for _, row in overlap_df.head(10).iterrows():
    summary_lines.append(
        f"- {row['Feature']}: in top 10 of {int(row['Top10Count'])} methods"
    )

summary_lines.extend([
    "",
    "Directional agreement with Logistic Regression:",
    f"- RF SHAP agreement rate: {rf_agreement_rate:.3f}" if not pd.isna(rf_agreement_rate) else "- RF SHAP agreement rate: unclear",
    f"- GB SHAP agreement rate: {gb_agreement_rate:.3f}" if not pd.isna(gb_agreement_rate) else "- GB SHAP agreement rate: unclear",
    "",
    "Note: built-in tree feature importance has no direction, so directional comparison is only done for SHAP."
])

with open(SUMMARY_TEXT, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("Saved summary text:")
print(SUMMARY_TEXT)
print()

print("Done.")