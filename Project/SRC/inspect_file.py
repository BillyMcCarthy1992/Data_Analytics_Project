from pathlib import Path
import pandas as pd

# load the dataset
csv_path = Path("Project/Data/Raw/cdc_diabetes_health_indicators.csv")
df = pd.read_csv(csv_path)

print("Dataset loaded.\n")

# quick first look
print("Shape:")
print(df.shape)
print()

print("Column names:")
print(df.columns.tolist())
print()

print("First 5 rows:")
print(df.head())
print()

print("Data types:")
print(df.dtypes)
print()

# check the target variable properly before doing anything else
target_col = "Diabetes_binary"

print(f"Target column: {target_col}")
print("\nUnique values in target:")
print(sorted(df[target_col].dropna().unique()))

print("\nValue counts in target:")
print(df[target_col].value_counts(dropna=False).sort_index())

print("\nTarget proportions:")
print(df[target_col].value_counts(normalize=True, dropna=False).sort_index())
print()

# quick missing value check
print("Missing values by column:")
print(df.isnull().sum())
print()

# also worth checking duplicates early
print("Number of duplicate rows:")
print(df.duplicated().sum())
print()

# first pass at grouping the variable types
binary_cols = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "DiffWalk", "Sex", "Diabetes_binary"
]

ordinal_cols = [
    "GenHlth", "Age", "Education", "Income"
]

continuous_or_count_cols = [
    "BMI", "MentHlth", "PhysHlth"
]

print("Likely binary columns:")
print(binary_cols)
print()

print("Likely ordinal columns:")
print(ordinal_cols)
print()

print("Likely continuous / count-like columns:")
print(continuous_or_count_cols)
print()

# checking the values column by column helps confirm the coding
for col in df.columns:
    unique_vals = sorted(df[col].dropna().unique())
    n_unique = len(unique_vals)

    print(f"{col}:")
    print(f"  Number of unique values: {n_unique}")

    if n_unique <= 15:
        print(f"  Unique values: {unique_vals}")
    else:
        print(f"  Min = {df[col].min()}, Max = {df[col].max()}")

    print()

# make a simple first-pass data dictionary
data_dict = pd.DataFrame({
    "column_name": df.columns,
    "likely_type": [
        "binary",            # HighBP
        "binary",            # HighChol
        "binary",            # CholCheck
        "continuous",        # BMI
        "binary",            # Smoker
        "binary",            # Stroke
        "binary",            # HeartDiseaseorAttack
        "binary",            # PhysActivity
        "binary",            # Fruits
        "binary",            # Veggies
        "binary",            # HvyAlcoholConsump
        "binary",            # AnyHealthcare
        "binary",            # NoDocbcCost
        "ordinal",           # GenHlth
        "count/discrete",    # MentHlth
        "count/discrete",    # PhysHlth
        "binary",            # DiffWalk
        "binary",            # Sex
        "ordinal",           # Age
        "ordinal",           # Education
        "ordinal",           # Income
        "binary target"      # Diabetes_binary
    ],
    "first_pass_meaning": [
        "High blood pressure indicator",
        "High cholesterol indicator",
        "Cholesterol check indicator",
        "Body Mass Index",
        "Smoking indicator",
        "Stroke history indicator",
        "Heart disease or heart attack history indicator",
        "Physical activity indicator",
        "Fruit consumption indicator",
        "Vegetable consumption indicator",
        "Heavy alcohol consumption indicator",
        "Any healthcare coverage/access indicator",
        "Could not see doctor because of cost indicator",
        "General health self-rating",
        "Poor mental health days",
        "Poor physical health days",
        "Difficulty walking indicator",
        "Sex variable",
        "Age category",
        "Education level",
        "Income category",
        "Diabetes outcome target"
    ]
})

print("First-pass data dictionary:")
print(data_dict)

# save this because it'll probably be useful later in the write-up
data_dict.to_csv("Project/Outputs/Tables/data_dictionary_first_pass.csv", index=False)