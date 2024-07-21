import pandas as pd
df=pd.read_csv("C:\\Users\\win10\\OneDrive\\Desktop\\deept
hi\\21-1212\\archive\\diabetes_prediction_dataset.csv")
print(df.isnull().sum())
# Remove rows with null values
df_cleaned = df.dropna()
# Check again for null values
print(df_cleaned.isnull().sum())
