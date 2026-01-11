import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
df = pd.read_csv("sensor_data.csv")

# Features and target
X = df[["ambient_temp", "voltage", "current"]]
y = df["power"]

# --------------------------------------------------
# 2. Train / test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# --------------------------------------------------
# 3. Linear Regression
# --------------------------------------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_lr = lin_reg.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)

print("Linear Regression MSE:", mse_lr)

# Inspect learned coefficients
print("Linear Regression coefficients:")
for name, coef in zip(X.columns, lin_reg.coef_):
    print(f"  {name:15s}: {coef:.4f}")

# --------------------------------------------------
# 4. Random Forest Regressor
# --------------------------------------------------
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf_reg.fit(X_train, y_train)

y_pred_rf = rf_reg.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)

print("\nRandom Forest MSE:", mse_rf)

# --------------------------------------------------
# 5. Optional: Feature importance (Random Forest)
# --------------------------------------------------
print("\nRandom Forest feature importances:")
for name, imp in zip(X.columns, rf_reg.feature_importances_):
    print(f"  {name:15s}: {imp:.4f}")

