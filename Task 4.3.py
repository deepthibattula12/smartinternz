import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Load the dataset
df = pd.read_csv('C:\\Users\\win10\\Downloads\\archive (2)\\data.csv')

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical variables
label_encoders = {}
for column in ['street', 'city', 'statezip', 'country']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Normalize/scale the numerical features
scaler = StandardScaler()
numerical_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                      'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
                      'yr_built', 'yr_renovated']

df[numerical_features] = scaler.fit_transform(df[numerical_features])

print(df.head())
