# Customer‐Level Aggregation (RFM + Repeat Flag)

# 1a) Ensure Purchase_Date is datetime
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])

# 1b) First purchase per customer (for feature engineering)
first_df = (
    df
    .sort_values('Purchase_Date')
    .drop_duplicates('Customer_ID', keep='first')
    .loc[:, [
        'Customer_ID','Purchase_Date','Age','Gender','City',
        'Category','Purchase_Amount','Discount_Applied','Rating','Payment_Method'
    ]]
    .reset_index(drop=True)
)

# 1c) Aggregate the provided Repeat_Customer flag to get “ever repeat”
repeat_df = (
    df
    .assign(is_repeat = lambda d: (d['Repeat_Customer']=='Yes').astype(int))
    .groupby('Customer_ID')['is_repeat']
    .max()              # 1 if any purchase was a repeat
    .reset_index()
    .rename(columns={'is_repeat':'repeat_ever'})
)

# 1d) Merge the label back onto first_df
first_df = first_df.merge(repeat_df, on='Customer_ID', how='left')

# Sanity check
print("Total customers:    ", first_df.shape[0])
print("Ever-repeat count:  ", first_df['repeat_ever'].sum())
print("Non-repeat count:   ", (first_df['repeat_ever']==0).sum())

#  2. Feature engineering on first purchases 
first = first_df.copy()

# 2a) Temporal features from the purchase date
first['month']     = first['Purchase_Date'].dt.month
first['weekday']   = first['Purchase_Date'].dt.day_name()

# 2b) Flag and code features
first['discount_flag'] = (first['Discount_Applied'] == 'Yes').astype(int)
first['gender_code']   = first['Gender'].map({'Male':0, 'Female':1}).fillna(2).astype(int)

# 2c) Define the feature columns and build X_first, y_first
feature_cols = [
    'Age',
    'Purchase_Amount',
    'Rating',
    'month',
    'weekday',
    'discount_flag',
    'gender_code',
    'Category',
    'Payment_Method'
]
X_first = first[feature_cols]
y_first = first['repeat_ever']

# Quick check
print("X_first shape:", X_first.shape)
print("Feature columns:", feature_cols)
print("Target distribution:\n", y_first.value_counts())

from sklearn.model_selection import train_test_split

#  3. Stratified random train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X_first, y_first,
    test_size=0.20,
    random_state=42,
    stratify=y_first
)

# Quick sanity check
print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])
print("Train distribution:\n", y_train.value_counts())
print("Test distribution:\n", y_test.value_counts())

#  4. SMOTE + XGB classification pipeline on RFM features 
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

# 4a) Define exactly the columns present in customer_df
feature_cols = [
    'visits',
    'recency_days',
    'frequency',
    'recent_high',
    'n_categories',
    'discount_rate',
    'avg_rating'
]
X = customer_df[feature_cols]
y = customer_df['repeat'].astype(int)

# 4b) Stratified train/test split to preserve both classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train class balance:\n", y_train.value_counts())
print("Test class balance:\n", y_test.value_counts())

# 4c) Build pipeline: scale → SMOTE → XGBoost
pipe = ImbPipeline([
    ('scale', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', XGBClassifier(
        random_state=42,
        learning_rate=0.05,
        max_depth=4,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

# 4d) Fit & evaluate
pipe.fit(X_train, y_train)
y_pred_proba = pipe.predict_proba(X_test)[:, 1]

print("Test AUC:", roc_auc_score(y_test, y_pred_proba))
print(classification_report(y_test, y_pred_proba >= 0.5))