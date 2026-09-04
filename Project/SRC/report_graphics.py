import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# save folder
fig_dir = Path("Project/Outputs/Figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# baseline (unweighted) results
# ----------------------------
baseline_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Accuracy": [0.862149, 0.864219, 0.864692, 0.864357],
    "Precision": [0.517305, 0.554022, 0.597514, 0.543549],
    "Recall": [0.158580, 0.130570, 0.088414, 0.165087],
    "F1": [0.242746, 0.211334, 0.154036, 0.253255],
    "ROC_AUC": [0.819160, 0.802443, 0.822311, 0.826425]
})

# ----------------------------
# tuned results
# F1-tuned and recall-tuned ended up the same in practice,
# so this just uses the final tuned values once
# ----------------------------
tuned_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Recall": [0.158438, 0.224784, 0.175131, 0.169614],
    "F1": [0.242661, 0.292337, 0.257407, 0.258628]
})

# shorter display names for the x-axis
name_map = {
    "Logistic Regression": "LogReg",
    "Decision Tree": "DT",
    "Random Forest": "RF",
    "Gradient Boosting": "GB"
}

baseline_df["ShortModel"] = baseline_df["Model"].map(name_map)
tuned_df["ShortModel"] = tuned_df["Model"].map(name_map)

# ============================================================
# Figure V.1
# Unweighted model comparison chart
# Accuracy, Precision, Recall, F1, ROC-AUC
# ============================================================
metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
plot_df = baseline_df.set_index("ShortModel")[metrics]

ax = plot_df.plot(kind="bar", figsize=(11, 6), width=0.8)
plt.title("Unweighted Baseline Model Performance")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 0.9)
plt.legend(title="Metric")
plt.tight_layout()

out_path = fig_dir / "figure_V1_unweighted_model_comparison.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(f"Saved Figure V.1 to: {out_path}")

# ============================================================
# Figure V.2
# Minority-class performance visual
# Recall and F1 only
# ============================================================
minority_df = baseline_df.set_index("ShortModel")[["Recall", "F1"]]

ax = minority_df.plot(kind="bar", figsize=(8, 5), width=0.75)
plt.title("Minority-Class Performance in the Unweighted Setup")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 0.35)
plt.legend(title="Metric")
plt.tight_layout()

out_path = fig_dir / "figure_V2_minority_class_performance.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(f"Saved Figure V.2 to: {out_path}")

# ============================================================
# Figure V.3
# Baseline vs tuned comparison
# grouped bars for Recall and F1
# ============================================================
compare_df = baseline_df[["Model", "ShortModel", "Recall", "F1"]].copy()
compare_df = compare_df.rename(columns={
    "Recall": "Baseline Recall",
    "F1": "Baseline F1"
})

compare_df = compare_df.merge(
    tuned_df[["Model", "Recall", "F1"]],
    on="Model",
    how="left"
).rename(columns={
    "Recall": "Tuned Recall",
    "F1": "Tuned F1"
})

plot_cols = ["Baseline Recall", "Tuned Recall", "Baseline F1", "Tuned F1"]
plot_df = compare_df.set_index("ShortModel")[plot_cols]

ax = plot_df.plot(kind="bar", figsize=(10, 6), width=0.8)
plt.title("Baseline vs Tuned Performance (Recall and F1)")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 0.35)
plt.legend(title="Metric")
plt.tight_layout()

out_path = fig_dir / "figure_V3_baseline_vs_tuned.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(f"Saved Figure V.3 to: {out_path}")