"""Train and evaluate models on the real California Housing dataset."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from src.load_data import load

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

df = load()
print(f"Loaded {len(df)} homes, columns: {list(df.columns)}")

X = df[FEATURES]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}

print("\n--- Model comparison ---")
best_name, best_r2, best_model = None, -1, None
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name:<18} MAE ${mae:>10,.0f} | RMSE ${rmse:>10,.0f} | R2 {r2:.4f}")
    if r2 > best_r2:
        best_name, best_r2, best_model = name, r2, model

print(f"\nBest model: {best_name} (R2 {best_r2:.4f})")

importance = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n--- Feature importance ---")
for feat, val in importance.items():
    print(f"{feat:<12} {val:.3f}")

y_pred = best_model.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.25, s=8)
plt.xlabel("Actual median house value")
plt.ylabel("Predicted")
plt.title(f"Actual vs Predicted ({best_name})")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.tight_layout()
plt.savefig("predictions.png", dpi=120)
print("\nSaved plot to predictions.png")
