import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('C:\\Users\\win10\\Downloads\\archive (2)\\data.csv')

# Check the datatype of columns
data_types = df.dtypes

# Perform descriptive statistics
descriptive_stats = df.describe()

# Do Preprocessing
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


# Task 5: Build ML Model with Linear Regression (Target column is price)
# Define features (X) and target (y)
X = df.drop(columns=['date', 'price'])
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
