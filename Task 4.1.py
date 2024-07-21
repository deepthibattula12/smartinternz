import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\win10\\Downloads\\archive (2)\\data.csv')

# Check the datatype of columns
data_types = df.dtypes
print(data_types)
