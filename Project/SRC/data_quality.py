from pathlib import Path
import pandas as pd

# load the dataset
csv_path = Path("Project/Data/Raw/cdc_diabetes_health_indicators.csv")
df = pd.read_csv(csv_path)

# start with the obvious checks
print("MISSING VALUES BY COLUMN")
print(df.isnull().sum())
print()

# percentages are handy too, easier to read quickly
print("MISSING VALUE PERCENTAGES")
print((df.isnull().mean() * 100).round(2))
print()

# check duplicates as well
num_duplicates = df.duplicated().sum()
dup_pct = df.duplicated().mean() * 100

print("DUPLICATE ROWS")
print(f"Number of duplicate rows: {num_duplicates}")
print(f"Percentage of duplicate rows: {dup_pct:.2f}%")
print()

# just show a few examples so I can see what they look like
print("EXAMPLE DUPLICATE ROWS")
print(df[df.duplicated()].head())
print()

# look at the coding for the main variables
key_columns = [
    "BMI", "Age", "GenHlth", "Education", "Income",
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "DiffWalk", "Sex", "Diabetes_binary"
]

print("UNIQUE VALUES FOR KEY COLUMNS")
for col in key_columns:
    unique_vals = sorted(df[col].dropna().unique())
    print(f"{col}:")
    print(f"  Number of unique values: {len(unique_vals)}")

    if len(unique_vals) <= 20:
        print(f"  Unique values: {unique_vals}")
    else:
        print(f"  Min = {df[col].min()}, Max = {df[col].max()}")

    print()

# general descriptive stats
print("DESCRIPTIVE STATISTICS")
print(df.describe().T)
print()

# binary columns should really just be 0/1
binary_cols = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "DiffWalk", "Sex", "Diabetes_binary"
]

print("BINARY COLUMN CHECKS (should usually be only 0/1)")
for col in binary_cols:
    vals = sorted(df[col].dropna().unique())
    invalid = [v for v in vals if v not in [0, 1]]
    print(f"{col}: unique values = {vals}")
    if invalid:
        print(f"  WARNING: unexpected values found -> {invalid}")
print()

# these are coded/ordinal so I want to make sure the ranges look sensible
print("ORDINAL / CODED VARIABLE RANGE CHECKS")
print(f"Age: min = {df['Age'].min()}, max = {df['Age'].max()}, unique = {sorted(df['Age'].unique())}")
print(f"GenHlth: min = {df['GenHlth'].min()}, max = {df['GenHlth'].max()}, unique = {sorted(df['GenHlth'].unique())}")
print(f"Education: min = {df['Education'].min()}, max = {df['Education'].max()}, unique = {sorted(df['Education'].unique())}")
print(f"Income: min = {df['Income'].min()}, max = {df['Income'].max()}, unique = {sorted(df['Income'].unique())}")
print()

# BMI is one of the main numeric variables, so worth a quick sanity check
print("BMI RANGE CHECK")
print(f"BMI min = {df['BMI'].min()}, max = {df['BMI'].max()}")
print("Top 10 largest BMI values:")
print(df["BMI"].sort_values(ascending=False).head(10).tolist())
print()

# not deleting anything here, just seeing if there are obvious extremes
print("ROWS WITH VERY LOW OR VERY HIGH BMI (example threshold check)")
extreme_bmi = df[(df["BMI"] < 15) | (df["BMI"] > 60)]
print(extreme_bmi[["BMI", "Age", "GenHlth", "Diabetes_binary"]].head(20))
print(f"Number of rows with BMI < 15 or BMI > 60: {len(extreme_bmi)}")
print()

# save a summary table as well because this will probably be useful later
quality_summary = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str).values,
    "missing_count": df.isnull().sum().values,
    "missing_pct": (df.isnull().mean() * 100).round(2).values,
    "n_unique": df.nunique().values,
    "min": [df[c].min() for c in df.columns],
    "max": [df[c].max() for c in df.columns]
})

print("QUALITY SUMMARY TABLE")
print(quality_summary)

quality_summary.to_csv("Project/Outputs/Tables/data_quality_summary.csv", index=False)