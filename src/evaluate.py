import argparse
import os
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--out", type=str, default="metrics/eval.json")
    args = parser.parse_args()

    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))

    model = joblib.load(os.path.join(args.models_dir, "house_price_model.pkl"))
    preds = model.predict(X_test)

    # Calculate regression metrics
    mse = mean_squared_error(y_test, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "mse": mse
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Metrics saved to", args.out)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.4f}")
