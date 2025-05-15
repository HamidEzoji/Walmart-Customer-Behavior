def prepare_features(df_cust, encode_high_card=False, df_full=None):
    X = df_cust.copy()
    X['repeat']      = (X['visits'] > 1).astype(int)
    X['recent_high'] = (X['recency_days'] < 30).astype(int)
    # keep City here so we can encode it
    X = X.drop(columns=['Customer_ID','first_purchase','last_purchase'])
    if encode_high_card and df_full is not None:
        city_counts  = df_full['City'].value_counts()
        X['city_freq'] = X['City'].map(city_counts)
        X = X.drop(columns=['City'])
    return X

# Convert to numeric where needed
df['Discount_Applied'] = df['Discount_Applied'].map({'Yes': 1, 'No': 0})
df['Repeat_Customer'] = df['Repeat_Customer'].map({'Yes': 1, 'No': 0})

# Now safe to aggregate
customer_df = df.groupby("Customer_ID").agg(
    last_purchase=("Purchase_Date", "max"),
    visits=("Purchase_Date", "count"),
    total_spent=("Purchase_Amount", "sum"),
    avg_rating=("Rating", "mean"),
    first_purchase=("Purchase_Date", "min"),
    n_categories=("Category", pd.Series.nunique),
    discount_rate=("Discount_Applied", "mean"),
    repeat=("Repeat_Customer", "max")
).reset_index()

# Additional features
customer_df["recency_days"] = (df["Purchase_Date"].max() - customer_df["last_purchase"]).dt.days
customer_df["frequency"] = customer_df["visits"] / ((customer_df["last_purchase"] - customer_df["first_purchase"]).dt.days + 1)
customer_df["recent_high"] = (customer_df["total_spent"] > customer_df["total_spent"].median()).astype(int)

# — Customer Segmentation with KMeans (Safe Copy) —
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

# Define the features used for clustering
cluster_features = [
    'visits',
    'recency_days',
    'frequency',
    'total_spent',
    'avg_rating',
    'n_categories'
]

# Create a separate copy to protect customer_df
segmentation_df = customer_df.copy()

# Drop rows with missing values only for segmentation
segmentation_df = segmentation_df.dropna(subset=cluster_features)

# Build the clustering pipeline
cluster_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=4, random_state=42))
])

# Apply clustering and add the labels
segmentation_df['cluster'] = cluster_pipeline.fit_predict(segmentation_df[cluster_features])

# Show the distribution of customers per cluster
print("Cluster counts:\n", segmentation_df['cluster'].value_counts())