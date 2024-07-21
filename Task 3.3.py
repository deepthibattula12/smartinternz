import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('C:\\Users\\win10\\AppData\\Local\\Programs\\Python\\Python311\\archive (1)\\creditcard.csv')

# Preprocess the data (drop rows with missing values)
df = df.dropna()

# Define the features and the target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a baseline Logistic Regression model on the imbalanced dataset
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics for the baseline model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print evaluation metrics for the baseline model
print("Baseline Model Evaluation:")
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'ROC-AUC: {roc_auc:.2f}')

# Print the classification report for the baseline model
print("\nClassification Report (Baseline):")
print(classification_report(y_test, y_pred))

# Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train a Logistic Regression model on the balanced dataset
log_reg_smote = LogisticRegression(max_iter=1000)
log_reg_smote.fit(X_train_smote, y_train_smote)

# Make predictions on the test set
y_pred_smote = log_reg_smote.predict(X_test)
y_pred_proba_smote = log_reg_smote.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics for the SMOTE model
accuracy_smote = accuracy_score(y_test, y_pred_smote)
precision_smote = precision_score(y_test, y_pred_smote)
recall_smote = recall_score(y_test, y_pred_smote)
f1_smote = f1_score(y_test, y_pred_smote)
roc_auc_smote = roc_auc_score(y_test, y_pred_proba_smote)

# Print evaluation metrics for the SMOTE model
print("\nSMOTE Model Evaluation:")
print(f'Accuracy (SMOTE): {accuracy_smote:.2f}')
print(f'Precision (SMOTE): {precision_smote:.2f}')
print(f'Recall (SMOTE): {recall_smote:.2f}')
print(f'F1 Score (SMOTE): {f1_smote:.2f}')
print(f'ROC-AUC (SMOTE): {roc_auc_smote:.2f}')

# Print the classification report for the SMOTE model
print("\nClassification Report (SMOTE):")
print(classification_report(y_test, y_pred_smote))

# Plot ROC curve for the SMOTE model
fpr_smote, tpr_smote, _ = roc_curve(y_test, y_pred_proba_smote)
roc_auc_smote = auc(fpr_smote, tpr_smote)

plt.figure()
plt.plot(fpr_smote, tpr_smote, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_smote:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (SMOTE)')
plt.legend(loc='lower right')
plt.show()
