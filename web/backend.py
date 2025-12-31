import os
import json
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib

# Import card processing functions
import sys
scripts_dir = str(Path(__file__).parent.parent / "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from tm_card_score import (
    TAG_KEYS, read_json, build_lookup, empty_parameters, 
    apply_item, to_int_or_zero
)

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = Path(__file__).parent.parent
CARDS_PATH = BASE_DIR / "cards" / "cards.json"
CORPORATIONS_PATH = BASE_DIR / "cards" / "corporations.json"
CSV_PATH = BASE_DIR / "datasets" / "games.csv"
MODEL_PATH = BASE_DIR / "web" / "model.pkl"

app = FastAPI(title="Terraforming Mars VP Predictor")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model_pipeline = None
cards_data = None
corporations_data = None


# Pydantic models
class PredictionRequest(BaseModel):
    corporation: str
    cards: List[str]


def make_onehot_dense():
    """Create OneHotEncoder with dense output, compatible with different sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore")
            # For very old versions, we'll need to handle sparse manually
            return encoder


def load_data():
    """Load cards and corporations data."""
    global cards_data, corporations_data
    if cards_data is None:
        cards_data = read_json(CARDS_PATH)
    if corporations_data is None:
        corporations_data = read_json(CORPORATIONS_PATH)
    return cards_data, corporations_data


def calculate_features(corporation: str, cards: List[str]) -> dict:
    """Calculate features from corporation and cards."""
    cards_data, corporations_data = load_data()
    
    card_by_name = build_lookup(cards_data)
    params = empty_parameters()
    
    # Add cards
    for card_name in cards:
        apply_item(params, card_by_name.get(card_name.strip().lower()))
    
    # Add corporation
    params["corporation"] = corporation.strip().lower()
    
    # Build feature dict
    features = {
        **{k: params["tags"].get(k, 0) for k in TAG_KEYS},
        "totalVp": params.get("totalVp", 0),
        "totalPrice": params.get("totalPrice", 0),
        "corporation": params.get("corporation", ""),
    }
    
    return features


def train_model(model_type: str = "RandomForest", n_estimators: int = 600, random_state: int = 42):
    """Train a new model and save it."""
    global model_pipeline
    
    if not CSV_PATH.exists():
        raise ValueError(f"Training data not found at {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    
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
    
    # Split data
    if "game_id" in df.columns and df["game_id"].duplicated().any():
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_idx, test_idx = next(gss.split(X_no_id, y, groups=df["game_id"]))
        X_train, X_test = X_no_id.iloc[train_idx], X_no_id.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_no_id, y, test_size=0.2, random_state=random_state
        )
    
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
    
    # Create model based on type
    if model_type == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state, 
            n_jobs=-1, 
            min_samples_leaf=2
        )
    else:
        # Default to RandomForest
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state, 
            n_jobs=-1, 
            min_samples_leaf=2
        )
    
    # Create pipeline
    model_pipeline = Pipeline(steps=[
        ("prep", preprocess),
        ("model", model)
    ])
    
    # Train
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = model_pipeline.score(X_train, y_train)
    test_score = model_pipeline.score(X_test, y_test)
    
    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, MODEL_PATH)
    
    return {
        "message": "Model trained and saved successfully",
        "train_score": float(train_score),
        "test_score": float(test_score),
        "model_type": model_type,
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test)
    }


def load_model():
    """Load the saved model."""
    global model_pipeline
    if MODEL_PATH.exists():
        model_pipeline = joblib.load(MODEL_PATH)
        return True
    return False


@app.on_event("startup")
async def startup_event():
    """Load model and data on startup. Auto-train if model doesn't exist."""
    print("Starting Terraforming Mars VP Predictor API...")
    load_data()
    print(f"Loaded {len(cards_data)} cards and {len(corporations_data)} corporations.")
    if load_model():
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"⚠️  No saved model found at {MODEL_PATH}")
        print("   Training model automatically...")
        try:
            result = train_model()
            print(f"✓ Model trained successfully!")
            print(f"  Train Score: {result['train_score']:.4f}, Test Score: {result['test_score']:.4f}")
        except Exception as e:
            print(f"✗ Failed to auto-train model: {e}")
            print("   Please run 'python scripts/train_model.py' locally and deploy the model.pkl file.")


@app.get("/")
async def root():
    """Serve the frontend HTML."""
    return FileResponse(BASE_DIR / "web" / "index.html")


@app.get("/cards")
async def get_cards():
    """Get all available cards."""
    cards_data, _ = load_data()
    return {"cards": cards_data}


@app.get("/corporations")
async def get_corporations():
    """Get all available corporations."""
    _, corporations_data = load_data()
    return {"corporations": corporations_data}


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predict VP based on corporation and cards."""
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train a model first.")
    
    try:
        # Calculate features
        features = calculate_features(request.corporation, request.cards)
        
        # Convert to DataFrame with same column order as training data
        # We need to match the exact feature order
        feature_dict = {
            **{k: [features.get(k, 0)] for k in TAG_KEYS},
            "totalVp": [features.get("totalVp", 0)],
            "totalPrice": [features.get("totalPrice", 0)],
            "corporation": [features.get("corporation", "")]
        }
        
        # Create DataFrame
        df = pd.DataFrame(feature_dict)
        
        # Predict
        prediction = model_pipeline.predict(df)[0]
        
        return {
            "predicted_points": float(prediction),
            "corporation": request.corporation,
            "num_cards": len(request.cards),
            "features": features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/status")
async def model_status():
    """Check if model is loaded."""
    return {
        "model_loaded": model_pipeline is not None,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists()
    }

