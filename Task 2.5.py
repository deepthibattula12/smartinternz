import pandas as pd
# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('C:\\Users\\win10\\OneDrive\\Desktop\\deepthi\\
21-1212\\archive\\diabetes_prediction_dataset.csv')
# Step 2: Separate features (X) and target (y)
# Assuming 'target_column' is the name of your target column
target_column = 'bmi' # Replace with the actual name of your
target column
X = df.drop(columns=[target_column]) # X contains all columns
except the target column
y = df[target_column] # y contains only the target column
# Step 3: Optionally, you can print the first few rows to verify
print("X (features):\n", X.head())
print("y (target):\n", y.head())
