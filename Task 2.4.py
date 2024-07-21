import pandas as pd
from sklearn.model_selection import train_test_split
# Read the CSV file into a DataFrame
df = pd.read_csv('C:\\Users\\win10\\OneDrive\\Desktop\\deepthi\\
21-1212\\archive\\diabetes_prediction_dataset.csv')
# Display the first few rows of the DataFrame
print(df.head())
# Separate features and target
# Assuming 'target' is the column name for the target variable
X = df.drop('bmi', axis=1)
y = df['bmi']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Display the splits
print("X_bmi:\n", X_train)
print("X_bmi:\n", X_test)
print("y_bmi:\n", y_train)
print("y_bmi:\n", y_test)
