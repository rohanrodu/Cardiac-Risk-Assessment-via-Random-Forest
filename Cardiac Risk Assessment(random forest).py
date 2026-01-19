import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("heart.csv")
heart_df = df.copy()

# Standardize target column name
heart_df = heart_df.rename(columns={'condition': 'target'})

# Split features and target
X = heart_df.drop(columns='target')
y = heart_df.target

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'random_state': [123]
}

# Initialize and fit GridSearchCV
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Select the best model
best_rf_model = grid_search.best_estimator_

# Cross-Validation Score
cv_scores = cross_val_score(best_rf_model, X, y, cv=5)

# Predictions
y_pred = best_rf_model.predict(X_test)

# Confusion Matrix Metrics
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 50)
print("ACADEMIC PROJECT: HEART DISEASE PREDICTION REPORT")
print("-" * 50)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Cross-Validation Mean Accuracy: {np.mean(cv_scores)*100:.2f}%")
print(f"Test Set Accuracy: {accuracy*100:.2f}%")
print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def check_patient(patient_data, name="Patient"):
    # Convert dict to DataFrame
    patient_df = pd.DataFrame(patient_data)
    
    # Ensure column order matches training data
    patient_df = patient_df[X.columns]
    
    # Predict
    result = best_rf_model.predict(patient_df)
    prob = best_rf_model.predict_proba(patient_df)
    
    print(f"\n>>> DIAGNOSIS FOR: {name}")
    if result[0] == 1:
        print(f"RESULT: HEART DISEASE DETECTED")
        print(f"CONFIDENCE: {prob[0][1]*100:.2f}%")
    else:
        print(f"RESULT: NO HEART DISEASE (HEALTHY)")
        print(f"CONFIDENCE: {prob[0][0]*100:.2f}%")
    print("-" * 30)

# Example 1: Healthy Person
healthy_person = {
    'age': [52], 'sex': [1], 'cp': [2], 'trestbps': [172], 'chol': [199],
    'fbs': [1], 'restecg': [1], 'thalach': [162], 'exang': [0],
    'oldpeak': [0.5], 'slope': [2], 'ca': [0], 'thal': [3]
}

# Example 2: High Risk Person
high_risk_person = {
    'age': [65], 'sex': [1], 'cp': [3], 'trestbps': [160], 'chol': [285],
    'fbs': [1], 'restecg': [2], 'thalach': [105], 'exang': [1],
    'oldpeak': [2.8], 'slope': [2], 'ca': [3], 'thal': [2]
}

# Run the checks
check_patient(healthy_person, "New Person (Healthy)")
check_patient(high_risk_person, "New Person (High Risk)")