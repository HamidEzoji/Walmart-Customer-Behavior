#  Classification on Enriched RFM Features 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# 1) Build X, y from customer_df
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

print("Class balance:\n", y.value_counts())

# 2) Preprocessor for numeric features
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), feature_cols)
], remainder='drop')

# 3) Define & wrap models
models = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'RandomForest'      : RandomForestClassifier(class_weight='balanced', random_state=42)
}
def make_pipe(m): return Pipeline([('prep', preprocessor), ('clf', m)])

# 4) Stratified split
X_tr, X_test, y_tr, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tr, y_tr, test_size=0.25, random_state=42, stratify=y_tr
)

print("Train balance:\n", y_train.value_counts())
print("Val   balance:\n", y_val.value_counts())

# 5) Fit & evaluate on validation
for name, mdl in models.items():
    pipe = make_pipe(mdl)
    pipe.fit(X_train, y_train)
    p_val = pipe.predict_proba(X_val)[:,1]
    print(f"{name} → Val AUC: {roc_auc_score(y_val,p_val):.3f}, Acc: {pipe.score(X_val,y_val):.3f}")

# 6) Hyperparameter tuning example
from sklearn.model_selection import GridSearchCV
param_grid = {'clf__n_estimators': [50,100], 'clf__max_depth': [5,10,None]}
gs = GridSearchCV(make_pipe(models['RandomForest']), param_grid, cv=3)
gs.fit(X_train, y_train)
print("Best RF params:", gs.best_params_)
print("RF Test AUC:", roc_auc_score(y_test, gs.predict_proba(X_test)[:,1]))
print(classification_report(y_test, gs.predict(X_test)))