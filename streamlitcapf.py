# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Streamlit setup
st.title("Readmission Prediction Model Comparison")
st.write("This app compares Random Forest, Decision Tree, and SVM models to predict heart disease.")

# Load dataset
df = pd.read_csv(r"C:\Users\nithi\OneDrive\Desktop\ML PACKAGE FINAL F\readmission.csv")

# Preprocessing
dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

# Splitting dataset
X = dataset.drop('target', axis=1)
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)

# Function to calculate train and test accuracy with better overfitting detection
def check_overfitting(model, model_name):
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    st.write(f"{model_name} - Training Accuracy: {train_acc:.4f}, Testing Accuracy: {test_acc:.4f}")
    
    # Define a threshold for detecting overfitting
    if train_acc == 1.0 and test_acc < 1.0:
        st.write(f"**{model_name} is overfitting** since training accuracy is 100% but testing accuracy is below 100%.")
        return True  # Indicates overfitting
    else:
        st.write(f"**{model_name} is not overfitting.**")
        return False

# Hyperparameter tuning function using GridSearchCV
def tune_hyperparameters(model, param_grid, model_name):
    st.write(f"Tuning hyperparameters for {model_name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    st.write(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")
    return best_model

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(kernel='linear')
}

# Dictionary to store final results for comparison
final_results = {}
best_model_name = None
best_accuracy = 0

for model_name, model in models.items():
    st.write(f"### Training {model_name}...")
    model.fit(X_train, y_train)
    
    # Check if overfitting
    is_overfitting = check_overfitting(model, model_name)
    
    if is_overfitting:
        # If overfitting, tune hyperparameters
        if model_name == 'Random Forest':
            param_grid_rf = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8, 10, None],
                'criterion': ['gini', 'entropy']
            }
            model = tune_hyperparameters(model, param_grid_rf, model_name)
        elif model_name == 'Decision Tree':
            param_grid_dt = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = tune_hyperparameters(model, param_grid_dt, model_name)
        elif model_name == 'SVM':
            param_grid_svm = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto']
            }
            model = tune_hyperparameters(model, param_grid_svm, model_name)
        
        # Retrain the model with best parameters after tuning
        st.write(f"Retraining {model_name} after hyperparameter tuning...")
        model.fit(X_train, y_train)
        
        # Re-check for overfitting after tuning
        is_still_overfitting = check_overfitting(model, model_name)
        
        if is_still_overfitting:
            st.write(f"Can't solve overfitting for {model_name}. Moving to next model.")
    
    # Store final accuracy results
    final_test_accuracy = model.score(X_test, y_test)
    final_results[model_name] = final_test_accuracy
    
    # Track the best model
    if final_test_accuracy > best_accuracy:
        best_accuracy = final_test_accuracy
        best_model_name = model_name

# Print final result
st.write(f"## The best model is {best_model_name} with a test accuracy of {best_accuracy:.4f}.")

# Plot comparison of accuracies
model_names = list(final_results.keys())
accuracy_scores = list(final_results.values())

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracy_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
plt.xlabel('Machine Learning Models')
plt.ylabel('Test Accuracy')
plt.title('Comparison of Model Accuracy Scores')
plt.xticks(rotation=45)
st.pyplot(plt)
