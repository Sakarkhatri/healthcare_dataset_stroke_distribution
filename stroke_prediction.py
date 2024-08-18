# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Data cleaning and preprocessing
# Load the dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())
print("\nFirst few rows of the dataset:")
print(df.head())

# Handle missing values
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# Convert categorical variables to numerical
le = LabelEncoder()
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Drop unnecessary columns
df.drop(['id'], axis=1, inplace=True)

# 2. Data Visualization
# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Distribution of stroke occurrences
plt.figure(figsize=(8, 6))
sns.countplot(x='stroke', data=df)
plt.title('Distribution of Stroke Occurrences')
plt.tight_layout()
plt.savefig('stroke_distribution.png')
plt.close()

# Age distribution for stroke and non-stroke patients
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='age', hue='stroke', shade=True)
plt.title('Age Distribution for Stroke and Non-Stroke Patients')
plt.tight_layout()
plt.savefig('age_distribution.png')
plt.close()

# 3. Analyze the data
# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Group by analysis
print("\nMean values grouped by stroke:")
print(df.groupby('stroke').mean())

# 4. Implement machine learning algorithms
# Prepare the data
X = df.drop('stroke', axis=1)
y = df['stroke']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a function to train and evaluate models
def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{model.__class__.__name__}:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

# Logistic Regression
lr_model = LogisticRegression(random_state=42)
train_evaluate_model(lr_model, X_train_scaled, X_test_scaled, y_train, y_test)

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
train_evaluate_model(rf_model, X_train_scaled, X_test_scaled, y_train, y_test)

# Support Vector Machine
svm_model = SVC(random_state=42)
train_evaluate_model(svm_model, X_train_scaled, X_test_scaled, y_train, y_test)

# Decision Tree Classifier (replacing XGBoost)
dt_model = DecisionTreeClassifier(random_state=42)
train_evaluate_model(dt_model, X_train_scaled, X_test_scaled, y_train, y_test)

# 5. Compare and analyze the results
# Feature importance (using Random Forest)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("Feature Importance:")
print(feature_importance)

print("\nAnalysis complete. Visualizations have been saved as PNG files in the current directory.")