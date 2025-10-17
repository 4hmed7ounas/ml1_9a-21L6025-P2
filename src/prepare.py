import argparse
import pandas as pd
import os
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="data/processed_data.csv")
    args = parser.parse_args()

    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["prepare"]

    # Read the dataset
    df = pd.read_csv(args.dataset_path)

    # Basic data cleaning
    # Convert Area Size to numeric (handle area conversions)
    df['Area_in_Marla'] = df['Area Size'].fillna(0)

    # Drop rows with missing critical values
    df = df.dropna(subset=['price', 'city', 'property_type', 'bedrooms', 'baths'])

    # Select relevant columns
    columns_to_keep = ['city', 'property_type', 'province_name', 'baths', 'bedrooms', 'Area_in_Marla', 'price']
    df_clean = df[columns_to_keep].copy()

    # Remove outliers (simple approach: keep prices within reasonable range)
    df_clean = df_clean[(df_clean['price'] > 0) & (df_clean['price'] < 1000000000)]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_clean.to_csv(args.out_csv, index=False)
    print(f"Saved cleaned data to {args.out_csv}")
    print(f"Dataset shape: {df_clean.shape}")
