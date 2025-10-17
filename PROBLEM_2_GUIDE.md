# Problem Statement 2 – DVC-Based ML Pipeline and Flask Deployment
## Customized Guide Based on Current Project State

---

## 📋 Overview

You are working on a house price prediction system using the **Pakistan House Price Dataset** from Kaggle. This guide will help you complete the DVC-based ML pipeline and Flask deployment using the **existing project structure** that already has significant components implemented.

**Dataset**: [Pakistan House Price Dataset](https://www.kaggle.com/datasets/jillanisofttech/pakistan-house-price-dataset)

---

## ✅ What's Already Implemented

Your project already has:
- ✅ Git repository initialized and connected to GitHub
- ✅ DVC initialized and configured
- ✅ Complete 4-stage ML pipeline (`dvc.yaml`) for Iris dataset
- ✅ Training scripts in `src/` directory:
  - `src/prepare.py` - Data preparation
  - `src/features.py` - Feature engineering
  - `src/train.py` - Model training
  - `src/evaluate.py` - Model evaluation
- ✅ `params.yaml` for hyperparameter management
- ✅ Flask application (`housepk_app.py`) with routes and templates
- ✅ HTML templates (`templates/index.html`, `templates/result.html`)
- ✅ Custom CSS styling (`static/css/style.css`)
- ✅ `.gitignore` configured to exclude models, data, and artifacts

---

## 🔄 Current Pipeline (Iris Dataset)

Your existing `dvc.yaml` pipeline:

```yaml
stages:
  prepare:
    cmd: python src/prepare.py --out_dir data
    outs:
      - data/iris.csv

  features:
    cmd: python src/features.py --in_csv data/iris.csv --out_dir data --test_size 0.2 --random_state 42
    deps:
      - data/iris_dataset.csv
      - src/features.py
    params:
      - features.test_size
      - features.random_state
    outs:
      - data/X_train.npy
      - data/X_test.npy
      - data/y_train.npy
      - data/y_test.npy

  train:
    cmd: python src/train.py --data_dir data --model_out model.pkl
    deps:
      - src/train.py
      - data/X_train.npy
      - data/y_train.npy
    params:
      - train.n_estimators
      - train.max_depth
      - train.random_state
    outs:
      - model.pkl

  evaluate:
    cmd: python src/evaluate.py --data_dir data --model model.pkl --out metrics/eval.json
    deps:
      - src/evaluate.py
      - data/X_test.npy
      - data/y_test.npy
      - model.pkl
    params:
      - evaluate.average
    metrics:
      - metrics/eval.json:
          cache: false
```

**Location**: `E:\Semster 9\MLops\Labs\Lab8\ml1_9a\dvc.yaml`

---

## 🎯 What You Need to Do

### Step 1: ✅ Set Up Project Repository (ALREADY DONE)

Your repository is already initialized:
- Git repository: ✅
- GitHub remote configured: ✅
- Branch: `feature-api` (current)
- Main branch: `main`

**Action**: Verify remote connection
```bash
git remote -v
git status
```

---

### Step 2: ✅ Integrate DVC for Data Versioning (PARTIALLY DONE)

DVC is initialized, but you need to:

#### 2.1 Configure DVC Remote Storage

```bash
# Option 1: Local storage (recommended for this lab)
dvc remote add -d local_storage ../dvc_storage

# Option 2: Google Drive
dvc remote add -d gdrive gdrive://<folder-id>

# Commit DVC remote configuration
git add .dvc/config
git commit -m "Configure DVC remote storage"
git push
```

#### 2.2 Download Pakistan House Price Dataset

1. Download from Kaggle: https://www.kaggle.com/datasets/jillanisofttech/pakistan-house-price-dataset
2. Place the CSV file in the `data/` directory
3. Rename it to `pakistan_house_prices.csv`

```bash
# Track the dataset with DVC
dvc add data/pakistan_house_prices.csv

# Commit the .dvc file
git add data/pakistan_house_prices.csv.dvc data/.gitignore
git commit -m "Add Pakistan house price dataset with DVC tracking"
git push

# Push data to DVC remote
dvc push
```

**Location**: `E:\Semster 9\MLops\Labs\Lab8\ml1_9a\data\pakistan_house_prices.csv`

---

### Step 3: ✅ Create Configuration Files (ALREADY DONE, NEEDS MODIFICATION)

Your `params.yaml` exists but is configured for Iris dataset. Update it for house price prediction:

**Location**: `E:\Semster 9\MLops\Labs\Lab8\ml1_9a\params.yaml`

**Update to**:
```yaml
prepare:
  dataset_path: data/pakistan_house_prices.csv
  random_seed: 42

features:
  test_size: 0.2
  random_state: 42
  target_column: price  # Adjust based on actual dataset column name
  categorical_features:
    - city
    - property_type
    - province_name
  numeric_features:
    - baths
    - bedrooms
    - Area_in_Marla

train:
  model_type: RandomForestRegressor  # Changed from Classifier to Regressor
  n_estimators: 100
  max_depth: 10  # Increased for regression
  min_samples_split: 2
  min_samples_leaf: 1
  random_state: 42

evaluate:
  metrics:
    - rmse
    - mae
    - r2_score
```

```bash
git add params.yaml
git commit -m "Update params.yaml for house price prediction"
git push
```

---

### Step 4: 🔧 Develop Training Pipeline (NEEDS MODIFICATION)

You have scripts in `src/`, but they need to be adapted for house price prediction (regression instead of classification).

#### 4.1 Modify `src/prepare.py`

**Current**: Loads Iris dataset from sklearn
**Needed**: Load Pakistan house price CSV

**Location**: `E:\Semster 9\MLops\Labs\Lab8\ml1_9a\src\prepare.py`

**Update the script**:
```python
import pandas as pd
import yaml
import argparse
import os

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def prepare_data(dataset_path, out_dir):
    """Load and perform initial data preparation"""
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Basic cleaning (adjust based on dataset)
    df = df.dropna(subset=['price'])  # Remove rows without price

    # Save prepared data
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'house_prices_prepared.csv')
    df.to_csv(out_path, index=False)
    print(f"Prepared data saved to {out_path}")
    print(f"Shape: {df.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='data')
    args = parser.parse_args()

    params = load_params()
    prepare_data(params['prepare']['dataset_path'], args.out_dir)
```

#### 4.2 Modify `src/features.py`

**Current**: Simple train/test split
**Needed**: Feature engineering with categorical encoding

**Location**: `E:\Semster 9\MLops\Labs\Lab8\ml1_9a\src\features.py`

**Update the script**:
```python
import pandas as pd
import numpy as np
import yaml
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def feature_engineering(in_csv, out_dir, test_size, random_state):
    """Perform feature engineering and train/test split"""
    params = load_params()

    # Load data
    df = pd.read_csv(in_csv)

    # Separate features and target
    target_col = params['features']['target_column']
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle categorical features
    categorical_features = params['features']['categorical_features']
    label_encoders = {}

    for col in categorical_features:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Save arrays
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'X_train.npy'), X_train.values)
    np.save(os.path.join(out_dir, 'X_test.npy'), X_test.values)
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train.values)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test.values)

    # Save feature names and encoders for Flask app
    os.makedirs('models', exist_ok=True)
    joblib.dump(X.columns.tolist(), 'models/model_features.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')

    print(f"Features saved. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_csv', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='data')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    feature_engineering(args.in_csv, args.out_dir, args.test_size, args.random_state)
```

#### 4.3 Modify `src/train.py`

**Current**: RandomForestClassifier
**Needed**: RandomForestRegressor

**Location**: `E:\Semster 9\MLops\Labs\Lab8\ml1_9a\src\train.py`

**Update the script**:
```python
import numpy as np
import yaml
import argparse
import joblib
import os
from sklearn.ensemble import RandomForestRegressor  # Changed from Classifier

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def train_model(data_dir, model_out):
    """Train RandomForest regression model"""
    params = load_params()['train']

    # Load training data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))

    # Create and train model
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state'],
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Save model for Flask app
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/house_price_model.pkl')

    # Also save to root for DVC tracking
    joblib.dump(model, model_out)

    print(f"Model trained and saved to {model_out}")
    print(f"Model also saved to models/house_price_model.pkl for Flask app")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_out', type=str, default='model.pkl')
    args = parser.parse_args()

    train_model(args.data_dir, args.model_out)
```

#### 4.4 Modify `src/evaluate.py`

**Current**: Accuracy and F1 score (classification metrics)
**Needed**: RMSE, MAE, R² (regression metrics)

**Location**: `E:\Semster 9\MLops\Labs\Lab8\ml1_9a\src\evaluate.py`

**Update the script**:
```python
import numpy as np
import yaml
import argparse
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(data_dir, model_path, out):
    """Evaluate regression model"""
    # Load test data
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    # Load model
    model = joblib.load(model_path)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2_score': float(r2_score(y_test, y_pred))
    }

    # Save metrics
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation metrics saved to {out}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R² Score: {metrics['r2_score']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model', type=str, default='model.pkl')
    parser.add_argument('--out', type=str, default='metrics/eval.json')
    args = parser.parse_args()

    evaluate_model(args.data_dir, args.model, args.out)
```

#### 4.5 Update `dvc.yaml`

**Location**: `E:\Semster 9\MLops\Labs\Lab8\ml1_9a\dvc.yaml`

**Update to**:
```yaml
stages:
  prepare:
    cmd: python src/prepare.py --out_dir data
    deps:
      - src/prepare.py
      - data/pakistan_house_prices.csv
    params:
      - prepare.dataset_path
    outs:
      - data/house_prices_prepared.csv

  features:
    cmd: python src/features.py --in_csv data/house_prices_prepared.csv --out_dir data --test_size 0.2 --random_state 42
    deps:
      - data/house_prices_prepared.csv
      - src/features.py
    params:
      - features.test_size
      - features.random_state
      - features.target_column
      - features.categorical_features
    outs:
      - data/X_train.npy
      - data/X_test.npy
      - data/y_train.npy
      - data/y_test.npy
      - models/model_features.pkl
      - models/label_encoders.pkl

  train:
    cmd: python src/train.py --data_dir data --model_out model.pkl
    deps:
      - src/train.py
      - data/X_train.npy
      - data/y_train.npy
    params:
      - train
    outs:
      - model.pkl
      - models/house_price_model.pkl

  evaluate:
    cmd: python src/evaluate.py --data_dir data --model model.pkl --out metrics/eval.json
    deps:
      - src/evaluate.py
      - data/X_test.npy
      - data/y_test.npy
      - model.pkl
    metrics:
      - metrics/eval.json:
          cache: false
```

**Commit changes**:
```bash
git add dvc.yaml src/*.py
git commit -m "Update pipeline for house price prediction"
git push
```

---

### Step 5: 🚀 Version Control Model and Outputs

Run the pipeline and track outputs:

```bash
# Run the complete pipeline
dvc repro

# Check status
dvc status

# Track the model with DVC
dvc add model.pkl

# Commit all changes
git add dvc.lock model.pkl.dvc
git commit -m "Train house price prediction model"
git push

# Push DVC artifacts to remote
dvc push
```

**View metrics**:
```bash
# Show metrics
dvc metrics show

# View metrics file directly
cat metrics/eval.json
```

---

### Step 6: ✅ Deploy Model with Flask (ALREADY IMPLEMENTED, NEEDS TESTING)

Your Flask app is already implemented in `housepk_app.py`!

**Location**: `E:\Semster 9\MLops\Labs\Lab8\ml1_9a\housepk_app.py`

**Features**:
- ✅ Model loading from `models/house_price_model.pkl`
- ✅ Feature metadata loading
- ✅ Label encoder handling for categorical features
- ✅ Dynamic form generation
- ✅ Prediction endpoint (`/predict`)
- ✅ API endpoint (`/api/predict`)

**Additional routes** (from merged branches):
- `/signup` - Signup placeholder
- `/login` - Login placeholder
- `/auth` - Authentication placeholder

**No changes needed** - just ensure model artifacts exist in `models/` directory (created by Step 4).

---

### Step 7: 🧪 Test and Validate Deployment

#### 7.1 Run Flask Server

```bash
# Ensure virtual environment is activated
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Run Flask app
python housepk_app.py
```

**Expected output**:
```
 * Running on http://127.0.0.1:5000
```

#### 7.2 Test in Browser

Navigate to: `http://localhost:5000`

You should see a beautiful form with:
- Gradient background (blue/purple theme)
- Dynamic input fields based on dataset features
- Dropdown menus for categorical features (city, property_type, etc.)
- Number inputs for numeric features (bedrooms, bathrooms, area)
- Submit button

**Fill in sample data** and click "Predict Price"

#### 7.3 Test API Endpoint

```bash
# Test health (if you add a health route)
curl http://localhost:5000/

# Test prediction API
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"baths\": 3, \"bedrooms\": 4, \"Area_in_Marla\": 10, \"city\": 0, \"property_type\": 1, \"province_name\": 2}"
```

**Note**: Adjust JSON fields based on actual dataset features and encoded values.

---

### Step 8: 📤 Push Final Code and Artifacts

```bash
# Check status
git status
dvc status

# Stage all changes
git add .
git commit -m "Complete house price prediction ML pipeline with Flask deployment"

# Push to GitHub
git push origin feature-api

# Push DVC artifacts
dvc push

# Merge to main branch
git checkout main
git merge feature-api
git push origin main
```

---

## 📁 Final Project Structure

```
E:\Semster 9\MLops\Labs\Lab8\ml1_9a\
├── .dvc/                              # DVC configuration
├── .git/                              # Git repository
├── .venv/                             # Virtual environment
├── data/
│   ├── pakistan_house_prices.csv.dvc  # DVC tracked dataset
│   ├── house_prices_prepared.csv      # Prepared data
│   ├── X_train.npy                    # Training features
│   ├── X_test.npy                     # Test features
│   ├── y_train.npy                    # Training labels
│   └── y_test.npy                     # Test labels
├── models/
│   ├── house_price_model.pkl          # Trained model (for Flask)
│   ├── model_features.pkl             # Feature names
│   ├── label_encoders.pkl             # Categorical encoders
│   └── feature_field_map.pkl          # Form field mapping
├── metrics/
│   └── eval.json                      # Evaluation metrics
├── src/
│   ├── prepare.py                     # Data preparation
│   ├── features.py                    # Feature engineering
│   ├── train.py                       # Model training
│   └── evaluate.py                    # Model evaluation
├── static/
│   └── css/
│       └── style.css                  # Custom styling
├── templates/
│   ├── index.html                     # Main form
│   └── result.html                    # Prediction result
├── .dvcignore
├── .gitignore
├── dvc.lock                           # Pipeline execution record
├── dvc.yaml                           # Pipeline definition
├── housepk_app.py                     # Flask application
├── model.pkl.dvc                      # DVC tracked model
├── params.yaml                        # Hyperparameters
├── requirements.txt                   # Dependencies
└── README.md
```

---

## 🎯 Key Commands Summary

### DVC Commands
```bash
dvc repro              # Run pipeline
dvc dag                # Show pipeline graph
dvc metrics show       # Display metrics
dvc push               # Push to DVC remote
dvc pull               # Pull from DVC remote
dvc status             # Check DVC status
```

### Git Commands
```bash
git status             # Check Git status
git add .              # Stage all changes
git commit -m "msg"    # Commit changes
git push               # Push to GitHub
git log --oneline      # View commit history
```

### Flask Commands
```bash
python housepk_app.py  # Run Flask server
```

---

## 🎓 Learning Objectives Achieved

✅ Set up Git repository with GitHub integration
✅ Initialize and configure DVC for data versioning
✅ Create reproducible ML pipeline with `dvc.yaml`
✅ Manage hyperparameters with `params.yaml`
✅ Version control datasets and models separately
✅ Train regression model (RandomForest)
✅ Evaluate model with regression metrics (RMSE, MAE, R²)
✅ Deploy model with Flask web application
✅ Create dynamic web forms for user input
✅ Serve predictions via web interface and API

---

## 🔍 Troubleshooting

### Issue: Model files not found
```bash
# Ensure pipeline ran successfully
dvc repro

# Check if models/ directory exists
ls models/

# Re-run training if needed
python src/train.py --data_dir data --model_out model.pkl
```

### Issue: DVC remote push fails
```bash
# Check remote configuration
dvc remote list
dvc remote -v list

# Reconfigure if needed
dvc remote modify local_storage url ../dvc_storage
```

### Issue: Flask app errors on startup
```bash
# Check if all model artifacts exist
ls models/house_price_model.pkl
ls models/model_features.pkl
ls models/label_encoders.pkl

# Check Python dependencies
pip install -r requirements.txt
```

### Issue: Prediction returns error
- Check that feature names match between training and prediction
- Verify categorical values are properly encoded
- Ensure input data types match model expectations

---

## 📝 Next Steps

1. **Improve Model**: Experiment with different hyperparameters in `params.yaml`
2. **Feature Engineering**: Add more features or transformations
3. **Model Comparison**: Try other algorithms (XGBoost, LightGBM)
4. **Deploy to Production**: Use Gunicorn, Docker, or cloud platforms
5. **Add Authentication**: Implement the login/signup functionality
6. **Add Visualizations**: Display feature importance, predictions plots
7. **CI/CD Pipeline**: Automate testing and deployment with GitHub Actions

---

## 📚 Resources

- **Your Files**:
  - `E:\Semster 9\MLops\Labs\Lab8\ml1_9a\dvc.yaml` - Pipeline definition
  - `E:\Semster 9\MLops\Labs\Lab8\ml1_9a\params.yaml` - Parameters
  - `E:\Semster 9\MLops\Labs\Lab8\ml1_9a\housepk_app.py` - Flask app

- **Documentation**:
  - [DVC Documentation](https://dvc.org/doc)
  - [Flask Documentation](https://flask.palletsprojects.com/)
  - [Scikit-learn Documentation](https://scikit-learn.org/)

---

**Good luck with your MLOps pipeline! 🚀**
