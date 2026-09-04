from pathlib import Path
from ucimlrepo import fetch_ucirepo
import pandas as pd

# grabbing the dataset from UCI
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

# put features and target together into one dataframe
df = pd.concat([X, y], axis=1)

# save it so I can use for the rest of the project
output_path = Path("Project/Data/Raw/cdc_diabetes_health_indicators.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print(f"CSV saved to: {output_path}")