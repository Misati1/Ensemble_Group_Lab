import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()

def train_evaluate_adaboost():
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print("AdaBoost Classifier Performance:")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return acc, classification_report(y_test, y_pred, output_dict=True)

def train_evaluate_gb():
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Gradient Boosting Classifier Performance:")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return acc, classification_report(y_test, y_pred, output_dict=True)

if __name__ == "__main__":
    ada_acc, ada_report = train_evaluate_adaboost()
    gb_acc, gb_report = train_evaluate_gb()

    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 
                   'Precision (class 0)', 'Precision (class 1)', 
                   'Recall (class 0)', 'Recall (class 1)', 
                   'F1-score (class 0)', 'F1-score (class 1)'],
        'AdaBoost': [
            ada_acc,
            ada_report['0']['precision'],
            ada_report['1']['precision'],
            ada_report['0']['recall'],
            ada_report['1']['recall'],
            ada_report['0']['f1-score'],
            ada_report['1']['f1-score'],
        ],
        'Gradient Boosting': [
            gb_acc,
            gb_report['0']['precision'],
            gb_report['1']['precision'],
            gb_report['0']['recall'],
            gb_report['1']['recall'],
            gb_report['0']['f1-score'],
            gb_report['1']['f1-score'],
        ]
    })

    print("\nComparison of AdaBoost vs Gradient Boosting:")
    print(comparison)

