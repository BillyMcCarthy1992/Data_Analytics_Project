import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# paths
table_path = Path("Project/Outputs/Tables/interpretability_comparison_table.csv")
fig_dir = Path("Project/Outputs/Figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# load comparison table
df = pd.read_csv(table_path)

# pick top 10 features by average normalized importance
top10 = (
    df.sort_values("AverageNorm", ascending=False)
      .head(10)
      .copy()
)

# columns to show in the heatmap
heatmap_cols = ["LogRegNorm", "RFNorm", "GBNorm", "RFSHAPNorm", "GBSHAPNorm"]

# nicer labels
label_map = {
    "LogRegNorm": "LogReg",
    "RFNorm": "RF Importance",
    "GBNorm": "GB Importance",
    "RFSHAPNorm": "RF SHAP",
    "GBSHAPNorm": "GB SHAP"
}

heatmap_data = top10.set_index("Feature")[heatmap_cols]
heatmap_data = heatmap_data.rename(columns=label_map)

plt.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={"label": "Normalized Importance"}
)

plt.title("Interpretability Comparison Heatmap (Top 10 Features)")
plt.xlabel("Explanation Method")
plt.ylabel("Feature")
plt.tight_layout()

out_path = fig_dir / "interpretability_comparison_heatmap_top10.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved to: {out_path}")