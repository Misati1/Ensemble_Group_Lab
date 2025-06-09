# tune_gradient_boosting.py

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# === Load preprocessed data ===
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()

# === Define parameter grid ===
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# === Create and run GridSearch ===
gb = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# === Best model ===
best_model = grid_search.best_estimator_
print("\nBest Parameters:")
print(grid_search.best_params_)

# === Evaluate on test set ===
y_pred = best_model.predict(X_test)
print("\nImproved Accuracy:", accuracy_score(y_test, y_pred))
print("\nImproved Classification Report:\n", classification_report(y_test, y_pred))

