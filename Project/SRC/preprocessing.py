from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# load the dataset
df = pd.read_csv("Project/Data/Raw/cdc_diabetes_health_indicators.csv")

# just check duplicates for reference, but keep all rows
n_rows = len(df)
n_duplicates = df.duplicated().sum()

print("Total rows in dataset:", n_rows)
print("Exact duplicate rows found:", n_duplicates)
print("No duplicate rows were removed.")
print()

# set up target and feature groups
target_col = "Diabetes_binary"

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

X = df[feature_cols]
y = df[target_col]

print("X shape:", X.shape)
print("y shape:", y.shape)
print()

# 80/20 split, keeping the class balance the same
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print()

print("Target proportions in full dataset:")
print(y.value_counts(normalize=True).sort_index())
print()

print("Target proportions in training set:")
print(y_train.value_counts(normalize=True).sort_index())
print()

print("Target proportions in test set:")
print(y_test.value_counts(normalize=True).sort_index())
print()

# logistic regression needs scaling for the numeric columns
# leaving binary and ordinal columns as they are
logreg_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols)
    ],
    remainder="passthrough"
)

# fit only on the training data
X_train_logreg = logreg_preprocessor.fit_transform(X_train)
X_test_logreg = logreg_preprocessor.transform(X_test)

print("Logistic Regression preprocessing complete.")
print("Scaled train matrix shape:", X_train_logreg.shape)
print("Scaled test matrix shape:", X_test_logreg.shape)
print()

# tree models can just use the original values
X_train_tree = X_train.copy()
X_test_tree = X_test.copy()

print("Tree-based model data ready.")
print("Unscaled train shape:", X_train_tree.shape)
print("Unscaled test shape:", X_test_tree.shape)
print()

# save everything so the later scripts can just load it
processed_dir = Path("Project/Data/Processed")
processed_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(processed_dir / "cleaned_diabetes.csv", index=False)
X_train.to_csv(processed_dir / "X_train.csv", index=False)
X_test.to_csv(processed_dir / "X_test.csv", index=False)
y_train.to_csv(processed_dir / "y_train.csv", index=False)
y_test.to_csv(processed_dir / "y_test.csv", index=False)

print(f"Saved dataset and train/test splits to {processed_dir}")