# wine_quality_analysis.py

# === Required Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === Load the Dataset ===
df = pd.read_csv("winequality-red.csv")

# === Basic Exploration ===
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# === Distribution of Quality Ratings ===
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=df, palette="Set2")
plt.title("Distribution of Wine Quality Ratings")
plt.xlabel("Wine Quality")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
