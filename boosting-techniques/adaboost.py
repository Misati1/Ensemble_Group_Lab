import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data (make sure these CSVs exist)
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()  # squeeze to convert single-column df to Series
y_test = pd.read_csv('y_test.csv').squeeze()

def train_evaluate_adaboost(learning_rate):
    # Create AdaBoost with DecisionTree stumps (max_depth=1)
    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=50,
        learning_rate=learning_rate,
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Learning Rate: {learning_rate}")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot feature importances
    plt.figure(figsize=(10,6))
    feature_importances = model.feature_importances_
    plt.barh(X_train.columns, feature_importances)
    plt.xlabel("Feature Importance")
    plt.title(f"AdaBoost Feature Importances (learning_rate={learning_rate})")
    plt.show()
    
    return acc

learning_rates = [0.5, 1.0, 1.5]
results = {}

for lr in learning_rates:
    print("="*40)
    acc = train_evaluate_adaboost(lr)
    results[lr] = acc

print("\nSummary of Accuracies:")
for lr, acc in results.items():
    print(f"Learning rate {lr}: Accuracy = {acc:.4f}")

