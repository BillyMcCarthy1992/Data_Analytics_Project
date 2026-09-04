import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# save folder
fig_dir = Path("Project/Outputs/Figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# weighted comparison results
df = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting",
        "Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [0.730054, 0.722150, 0.731611, 0.697848],
    "Precision": [0.310819, 0.306923, 0.310839, 0.284252],
    "Recall": [0.770123, 0.790211, 0.761069, 0.769840],
    "F1": [0.442890, 0.442123, 0.441400, 0.415198],
    "ROC_AUC": [0.822961, 0.826593, 0.819621, 0.801105],
})

metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

# put models on x-axis and metrics as grouped bars
plot_df = df.set_index("Model")[metrics]

ax = plot_df.plot(
    kind="bar",
    figsize=(11, 6),
    width=0.8
)

plt.title("Weighted Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=15)
plt.ylim(0, 0.9)
plt.legend(title="Metric")
plt.tight_layout()

out_path = fig_dir / "weighted_model_comparison_barchart.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved to: {out_path}")
