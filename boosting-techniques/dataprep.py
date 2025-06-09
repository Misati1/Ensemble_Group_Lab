# === Required Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === Load the Dataset ===
df = pd.read_csv("../winequality-red.csv")

# === Basic Exploration ===
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# === Distribution of Original Quality Ratings ===
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=df, palette="Set2")
plt.title("Distribution of Wine Quality Ratings")
plt.xlabel("Wine Quality")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# === Convert Quality to Binary (Good vs Not Good) ===
df['good_quality'] = (df['quality'] >= 7).astype(int)

# Drop original quality column
df.drop('quality', axis=1, inplace=True)

# === Define Features and Labels ===
X = df.drop('good_quality', axis=1)
y = df['good_quality']

# === Split into Train and Test Sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Standardize the Features ===
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# === Check Class Balance ===
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette="Set1")
plt.title("Binary Wine Quality (0 = Not Good, 1 = Good)")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

print("Class balance in y_train:\n", y_train.value_counts(normalize=True))
# === Save preprocessed data to CSV ===
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\n✅ Preprocessed data saved as CSV files.")

