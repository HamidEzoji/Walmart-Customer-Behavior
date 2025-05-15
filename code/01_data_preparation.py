import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans


# Load dataset
df = pd.read_csv("https://drive.google.com/uc?export=download&id=1CFh9rxshWQV7MqYLFn753phj_65UiiyJ", parse_dates=["Purchase_Date"])
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], errors='coerce')
print(df.head())

# Checking the number of rows and columns
print(f"Dataset Shape: {df.shape}")

# Column types & non-null counts
print("\nColumn Information:")
df.info()

# Missing values per column
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate entries
print(f"\nDuplicate Rows: {df.duplicated().sum()}")
# Drop duplicates
df = df.drop_duplicates()

# Detect hidden/null-like strings
for col in df.columns:
    print(f"{col} unique values:", df[col].unique()[:10])

for col in df.select_dtypes(include='object'):
    print(f"{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts(dropna=False))
    print("\n")

# Check for hidden missing value placeholders
for col in df.columns:
    print(col, df[col].isin(['N/A', 'null', '', '-']).sum())