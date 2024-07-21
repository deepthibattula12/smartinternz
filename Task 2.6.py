import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Load the dataset
df = pd.read_csv('C:\\Users\\win10\\OneDrive\\Desktop\\deepthi\\
21-1212\\archive\\diabetes_prediction_dataset.csv')
# Preprocess the data
# Encode categorical variables
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])
df['smoking_history'] =
label_encoder.fit_transform(df['smoking_history'])
# Split the data into features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
# Train Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
y_pred_random_forest = random_forest.predict(X_test)
random_forest_accuracy = accuracy_score(y_test, y_pred_random_forest)
# Train Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_test)
decision_tree_accuracy = accuracy_score(y_test, y_pred_decision_tree)
# Print the accuracies
print(f'Logistic Regression Test Accuracy:
{log_reg_accuracy:.2f}')
print(f'Random Forest Test Accuracy:
{random_forest_accuracy:.2f}')
print(f'Decision Tree Test Accuracy:
{decision_tree_accuracy:.2f}')
