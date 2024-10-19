import mlflow
from sklearn.model_selection import train_test_split
import pandas as pd

# Import Database
data = pd.read_csv("fake_data.csv")
X = data.drop(columns=["date", "demand"])
X = X.astype('float')
y = data["demand"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define MLflow Model path
run_id = '6c62d61fb28d4e3a9cbd7d03b8c8ff43'
experiment_id = '532243866540897495'
model_path = f'/home/ubuntu/DST_MLOPS_MLflow/mlruns/{experiment_id}/{run_id}/artifacts/rf_apples'

# Load model with sklearn flavor
model = mlflow.sklearn.load_model(model_path)

# Make predictions
predictions = model.predict(X_val)

# Calculate the mean prediction
mean_prediction = predictions.mean()
print(mean_prediction)