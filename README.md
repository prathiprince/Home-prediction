# Home-prediction
capstone project
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
# Load the data from the given CSV files
application_train = pd.read_csv(r'C:\Users\prath\OneDrive\Desktop\application_train.csv')
bureau = pd.read_csv(r'C:\Users\prath\OneDrive\Desktop\bureau.csv')
bureau_balance = pd.read_csv(r'C:\Users\prath\OneDrive\Desktop\bureau_balance.csv')
pos_cash_balance = pd.read_csv(r'C:\Users\prath\OneDrive\Desktop\POS_CASH_balance.csv')
credit_card_balance = pd.read_csv(r'C:\Users\prath\OneDrive\Desktop\credit_card_balance.csv')
previous_application = pd.read_csv(r'C:\Users\prath\OneDrive\Desktop\previous_application.csv')
installments_payments = pd.read_csv(r'C:\Users\prath\OneDrive\Desktop\installments_payments.csv')

# Take a look at the structure of the main data
application_train.head()
# Check for missing values
missing_values = application_train.isnull().sum()
missing_values = missing_values[missing_values > 0]
missing_values.sort_values(ascending=False).head()

# Drop columns with more than 50% missing data or apply imputation
threshold = len(application_train) * 0.5
application_train = application_train.dropna(thresh=threshold, axis=1)

# Fill remaining missing data with median or mode based on feature type
application_train = application_train.fillna(application_train.median())
# Separate numeric and non-numeric columns
numeric_columns = application_train.select_dtypes(include=[np.number]).columns
categorical_columns = application_train.select_dtypes(include=[object]).columns

# Fill missing values in numeric columns with median
application_train[numeric_columns] = application_train[numeric_columns].fillna(application_train[numeric_columns].median())

# Fill missing values in categorical columns with mode (most frequent value)
application_train[categorical_columns] = application_train[categorical_columns].apply(lambda x: x.fillna(x.mode()[0]))

# Now, the DataFrame should have no missing values and all columns filled appropriately
application_train.isnull().sum()  # Check if all missing values are handled
# Target distribution
sns.countplot(application_train['TARGET'])
plt.title('Target Distribution (1: Defaulter, 0: Not Defaulter)')
plt.show()

# Analyze key features
plt.figure(figsize=(10, 5))
sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=application_train)
plt.title('Income vs Target')
plt.show()

# Correlation Matrix
corr = application_train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap='coolwarm', annot=False, linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
# Create new features
application_train['CREDIT_INCOME_RATIO'] = application_train['AMT_CREDIT'] / application_train['AMT_INCOME_TOTAL']
application_train['LOAN_TERM'] = application_train['AMT_ANNUITY'] / application_train['AMT_CREDIT']

# Encode categorical variables using one-hot encoding or LabelEncoder
le = LabelEncoder()
application_train['CODE_GENDER'] = le.fit_transform(application_train['CODE_GENDER'])
application_train = pd.get_dummies(application_train, drop_first=True)
# Separate features and target
X = application_train.drop('TARGET', axis=1)
y = application_train['TARGET']

# Handling imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred_log = log_reg.predict(X_test)

# Evaluation
print('Logistic Regression Accuracy:', accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)

# Evaluation
print('Random Forest Accuracy:', accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
# XGBoost Classifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb.predict(X_test)

# Evaluation
print('XGBoost Accuracy:', accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
# ROC-AUC Score for each model
roc_log = roc_auc_score(y_test, y_pred_log)
roc_rf = roc_auc_score(y_test, y_pred_rf)
roc_xgb = roc_auc_score(y_test, y_pred_xgb)

print('Logistic Regression ROC-AUC:', roc_log)
print('Random Forest ROC-AUC:', roc_rf)
print('XGBoost ROC-AUC:', roc_xgb)

# Compare performance
models = ['Logistic Regression', 'Random Forest', 'XGBoost']
roc_scores = [roc_log, roc_rf, roc_xgb]
plt.bar(models, roc_scores, color=['blue', 'green', 'orange'])
plt.title('Model Comparison (ROC-AUC)')
plt.ylabel('ROC-AUC Score')
plt.show()
