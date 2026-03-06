import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# 1. Create output directory
# --------------------------------------------------
os.makedirs("output", exist_ok=True)

# --------------------------------------------------
# 2. Load dataset (Correct separator)
# --------------------------------------------------
data_path = os.path.join("data", "winequality-red.csv")

# Define column names explicitly since the header is malformed
column_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'quality'
]

data = pd.read_csv(data_path, sep=";", header=None, names=column_names, skiprows=1)

# Remove extra spaces from column names (though not necessary now)
data.columns = data.columns.str.strip()

print("Columns in dataset:", data.columns)

# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------
# Automatically select last column as target
target_column = data.columns[-1]

X = data.drop(target_column, axis=1)
y = data[target_column]

# --------------------------------------------------
# 4. Feature Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 5. Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# --------------------------------------------------
# 6. Train Model
# --------------------------------------------------
model = Lasso(alpha=0.5, max_iter=10000)
model.fit(X_train, y_train)

# --------------------------------------------------
# 7. Prediction
# --------------------------------------------------
y_pred = model.predict(X_test)

# --------------------------------------------------
# 8. Evaluation Metrics
# --------------------------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# --------------------------------------------------
# 9. Save Model
# --------------------------------------------------
joblib.dump(model, os.path.join("output", "model.pkl"))

# --------------------------------------------------
# 10. Save Metrics
# --------------------------------------------------
metrics = {
    "MSE": mse,
    "R2_Score": r2
}

with open(os.path.join("output", "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("Model and metrics saved successfully in 'output' folder.")