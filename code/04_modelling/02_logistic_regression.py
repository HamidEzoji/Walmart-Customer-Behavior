#  Regression to Predict Total Spend 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1) Build X,y without leaking total_spent
reg_y = customer_df['total_spent']
reg_X = customer_df[[
    'visits',
    'recency_days',
    'frequency',
    'recent_high'
]]

# 2) Split
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    reg_X, reg_y, test_size=0.2, random_state=42
)

# 3) Pipeline
reg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('reg',    RandomForestRegressor(random_state=42))
])

# 4) Fit & evaluate
reg_pipe.fit(Xr_tr, yr_tr)
y_pred = reg_pipe.predict(Xr_te)
print("MAE:", mean_absolute_error(yr_te, y_pred))
print("R² :", r2_score(yr_te, y_pred))