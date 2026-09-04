import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# paths
csv_path = Path("Project/Outputs/Tables/interpretability_directional_agreement.csv")
fig_dir = Path("Project/Outputs/Figures")
fig_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

# optional: focus on the most important features first
feature_order = [
    "GenHlth", "HighBP", "BMI", "Age", "HighChol",
    "Income", "DiffWalk", "CholCheck", "HeartDiseaseorAttack",
    "Sex", "HvyAlcoholConsump", "PhysHlth"
]

# keep only features that are actually present
feature_order = [f for f in feature_order if f in df["Feature"].values]
df = df.set_index("Feature").loc[feature_order].reset_index()

# map direction strings to numbers for plotting
direction_map = {
    "Negative": -1,
    "Zero": 0,
    "Positive": 1,
    "Unclear": 0
}

# shorter method names
col_map = {
    "LogRegDirection": "LogReg",
    "RFSHAPDirection": "RFSHAP",
    "GBSHAPDirection": "GBSHAP"
}

plot_df = (
    df.set_index("Feature")[["LogRegDirection", "RFSHAPDirection", "GBSHAPDirection"]]
    .rename(columns=col_map)
    .replace(direction_map)
)

annot_df = (
    df.set_index("Feature")[["LogRegDirection", "RFSHAPDirection", "GBSHAPDirection"]]
    .rename(columns=col_map)
)

plt.figure(figsize=(6, 6))
sns.heatmap(
    plot_df,
    annot=annot_df,
    fmt="",
    cmap="bwr",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "Direction (-1 = Negative, +1 = Positive)"}
)

plt.title("Directional Agreement Across Interpretation Methods")
plt.xlabel("Method")
plt.ylabel("Feature")
plt.tight_layout()

out_path = fig_dir / "interpretability_directional_agreement_heatmap.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved to: {out_path}")