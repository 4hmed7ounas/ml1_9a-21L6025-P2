import argparse
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))

    # Train RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=params["random_state"],
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Save model and artifacts for Flask app
    os.makedirs(args.models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.models_dir, "house_price_model.pkl"))

    # Copy encoders and feature info to models directory for Flask
    import shutil
    shutil.copy(os.path.join(args.data_dir, "label_encoders.pkl"),
                os.path.join(args.models_dir, "label_encoders.pkl"))
    shutil.copy(os.path.join(args.data_dir, "feature_list.pkl"),
                os.path.join(args.models_dir, "model_features.pkl"))
    shutil.copy(os.path.join(args.data_dir, "feature_field_map.pkl"),
                os.path.join(args.models_dir, "feature_field_map.pkl"))

    print(f"Model saved to {args.models_dir}/house_price_model.pkl")
    print(f"Model trained on {X_train.shape[0]} samples with {X_train.shape[1]} features")
