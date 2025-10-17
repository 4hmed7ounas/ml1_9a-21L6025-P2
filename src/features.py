import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import yaml
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["features"]

    df = pd.read_csv(args.in_csv)

    # Encode categorical features
    label_encoders = {}
    categorical_features = params["categorical_features"]

    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Split features and target
    target_col = params["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train-test split (no stratify for regression)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"]
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Save numpy arrays
    np.save(os.path.join(args.out_dir, "X_train.npy"), X_train.values)
    np.save(os.path.join(args.out_dir, "X_test.npy"), X_test.values)
    np.save(os.path.join(args.out_dir, "y_train.npy"), y_train.values)
    np.save(os.path.join(args.out_dir, "y_test.npy"), y_test.values)

    # Save label encoders and feature info for Flask app
    joblib.dump(label_encoders, os.path.join(args.out_dir, "label_encoders.pkl"))
    joblib.dump(list(X.columns), os.path.join(args.out_dir, "feature_list.pkl"))

    # Create feature-to-field mapping for Flask
    feature_field_map = {col: col for col in X.columns}
    joblib.dump(feature_field_map, os.path.join(args.out_dir, "feature_field_map.pkl"))

    print(f"Train/test data saved in {args.out_dir}")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
