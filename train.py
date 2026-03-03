import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# 1. Define base paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "winequality-red.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "app", "artifacts")

# Create artifact directory
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------
data = pd.read_csv(DATA_PATH, sep=";")

# --------------------------------------------------
# 3. Split features and target
# --------------------------------------------------
X = data.drop("quality", axis=1)
y = data["quality"]

# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------------------------------
# 5. Pipeline (Scaling + Model)
# --------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(max_iter=10000))
])

# --------------------------------------------------
# 6. Hyperparameter tuning
# --------------------------------------------------
param_grid = {
    "lasso__alpha": [0.001, 0.01, 0.05, 0.1, 0.5]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="r2"
)

# --------------------------------------------------
# 7. Train model
# --------------------------------------------------
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# --------------------------------------------------
# 8. Prediction
# --------------------------------------------------
y_pred = best_model.predict(X_test)

# --------------------------------------------------
# 9. Metrics
# --------------------------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best alpha:", grid.best_params_["lasso__alpha"])
print("MSE:", mse)
print("R2 Score:", r2)

# --------------------------------------------------
# 10. Save model
# --------------------------------------------------
joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "model.pkl"))

# --------------------------------------------------
# 11. Save metrics
# --------------------------------------------------
metrics = {
    "MSE": mse,
    "R2_Score": r2,
    "Best_Alpha": grid.best_params_["lasso__alpha"]
}

with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("Artifacts saved to:", ARTIFACT_DIR)
