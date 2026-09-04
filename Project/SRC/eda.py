import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("Project/Data/Raw/cdc_diabetes_health_indicators.csv")

# 1. target class distribution
df["Diabetes_binary"].value_counts().sort_index().plot(kind="bar")
plt.title("Diabetes Target Class Distribution")
plt.xlabel("Diabetes_binary")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("Project/Outputs/Figures/target_class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# 2. BMI by diabetes class
df.boxplot(column="BMI", by="Diabetes_binary", grid=False)
plt.title("BMI by Diabetes Class")
plt.suptitle("")
plt.xlabel("Diabetes_binary")
plt.ylabel("BMI")
plt.tight_layout()
plt.savefig("Project/Outputs/Figures/bmi_by_diabetes_class.png", dpi=300, bbox_inches="tight")
plt.show()

# 3. diabetes prevalence by age category
age_prev = df.groupby("Age")["Diabetes_binary"].mean() * 100
age_prev.plot(marker="o")
plt.title("Diabetes Prevalence by Age Category")
plt.xlabel("Age Category")
plt.ylabel("Diabetes Prevalence (%)")
plt.tight_layout()
plt.savefig("Project/Outputs/Figures/diabetes_prevalence_by_age.png", dpi=300, bbox_inches="tight")
plt.show()

# 4. diabetes prevalence by general health
genhlth_prev = df.groupby("GenHlth")["Diabetes_binary"].mean() * 100
genhlth_prev.plot(marker="o")
plt.title("Diabetes Prevalence by General Health")
plt.xlabel("GenHlth")
plt.ylabel("Diabetes Prevalence (%)")
plt.tight_layout()
plt.savefig("Project/Outputs/Figures/diabetes_prevalence_by_genhlth.png", dpi=300, bbox_inches="tight")
plt.show()

# 5. diabetes prevalence by physical activity
phys_prev = df.groupby("PhysActivity")["Diabetes_binary"].mean() * 100
phys_prev.plot(kind="bar")
plt.title("Diabetes Prevalence by Physical Activity")
plt.xlabel("PhysActivity")
plt.ylabel("Diabetes Prevalence (%)")
plt.tight_layout()
plt.savefig("Project/Outputs/Figures/diabetes_prevalence_by_physactivity.png", dpi=300, bbox_inches="tight")
plt.show()