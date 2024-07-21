import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('C:\\Users\\win10\\AppData\\Local\\Programs\\Python\\Python311\\archive (1)\\creditcard.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())
# Display the summary statistics of the dataset
print("\nSummary statistics of the dataset:")
print(df.describe())
# Display information about the dataset
print("\nInformation about the dataset:")
print(df.info())
# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Drop any rows with missing values
df = df.dropna()
# Plot the distribution of the 'Class' variable
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Distribution of Classes')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
