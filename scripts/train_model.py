#!/usr/bin/env python3
"""
Script to train the Terraforming Mars VP prediction model.

This script should be run locally to train/retrain the model.
After training, the model.pkl file should be committed and deployed.

Usage:
    python scripts/train_model.py
"""
import sys
import warnings
from pathlib import Path

# Add parent directory to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

# Import card processing functions
sys.path.insert(0, str(BASE_DIR / "scripts"))
from tm_card_score import TAG_KEYS

# Paths
CSV_PATH = BASE_DIR / "datasets" / "games.csv"
if not CSV_PATH.exists():
    CSV_PATH = BASE_DIR / "scripts" / "games.csv"

MODEL_PATH = BASE_DIR / "web" / "model.pkl"


def make_onehot_dense():
    """Create OneHotEncoder with dense output, compatible with different sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore")


def train_model(
    model_type: str = "RandomForest",
    n_estimators: int = 600,
    random_state: int = 42,
    min_samples_leaf: int = 2
):
    """Train the model and save it."""
    print(f"Loading training data from {CSV_PATH}...")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples")
    
    # Target
    y = df["points"]
    X = df.drop(columns=["points"])
    
    # Remove game_id if present
    if "game_id" in X.columns:
        X_no_id = X.drop(columns=["game_id"])
    else:
        X_no_id = X
    
    # Categorize columns
    cat_cols = []
    if "corporation" in X_no_id.columns:
        cat_cols.append("corporation")
    
    num_cols = [c for c in X_no_id.columns if c not in cat_cols]
    
    print(f"Features: {len(num_cols)} numerical, {len(cat_cols)} categorical")
    
    # Split data
    if "game_id" in df.columns and df["game_id"].duplicated().any():
        print("Using GroupShuffleSplit to preserve game groups...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_idx, test_idx = next(gss.split(X_no_id, y, groups=df["game_id"]))
        X_train, X_test = X_no_id.iloc[train_idx], X_no_id.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        print("Using standard train_test_split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_no_id, y, test_size=0.2, random_state=random_state
        )
    
    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Preprocessing for tree-based models
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_onehot_dense()),
            ]), cat_cols),
        ],
        remainder="drop"
    )
    
    # Create model
    if model_type == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=min_samples_leaf
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ("prep", preprocess),
        ("model", model)
    ])
    
    # Train
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETE")
    print("="*50)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"\nModel Type: {model_type}")
    print(f"Parameters: n_estimators={n_estimators}, min_samples_leaf={min_samples_leaf}")
    print(f"\nTraining Set Metrics:")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    print(f"\nTest Set Metrics:")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    print("="*50)
    
    return {
        "model_path": str(MODEL_PATH),
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Terraforming Mars VP prediction model")
    parser.add_argument("--n-estimators", type=int, default=600, help="Number of estimators (default: 600)")
    parser.add_argument("--min-samples-leaf", type=int, default=2, help="Min samples per leaf (default: 2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state (default: 42)")
    
    args = parser.parse_args()
    
    try:
        result = train_model(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state
        )
        print("\n✓ Model training completed successfully!")
        print(f"  Next step: Commit and push {MODEL_PATH} to deploy the new model.")
    except Exception as e:
        print(f"\n✗ Error training model: {e}")
        sys.exit(1)

