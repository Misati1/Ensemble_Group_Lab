import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Preprocessed Data ===
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()

# === Train XGBoost Classifier ===
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# === Evaluate Model ===
xgb_acc = accuracy_score(y_test, y_pred)
print("XGBoost Classifier Performance:")
print(f"Accuracy: {xgb_acc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
