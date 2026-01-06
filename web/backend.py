import warnings
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib

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


def load_data():
    """Load cards and corporations data."""
    global cards_data, corporations_data
    if cards_data is None:
        cards_data = read_json(CARDS_PATH)
    if corporations_data is None:
        corporations_data = read_json(CORPORATIONS_PATH)
    return cards_data, corporations_data


def calculate_features(corporation, cards):
    cards_data, corporations_data = load_data()
    
    card_by_name = build_lookup(cards_data)
    params = empty_parameters()
    
    for card_name in cards:
        apply_item(params, card_by_name.get(card_name.strip().lower()))
    
    params["corporation"] = corporation.strip().lower()
    
    features = {}

    tags = params.get("tags", {})
    for k in TAG_KEYS:
        if k in tags:
            features[k] = tags[k]
        else:
            features[k] = 0

    features["totalVp"] = params.get("totalVp", 0)
    features["totalPrice"] = params.get("totalPrice", 0)
    features["corporation"] = params.get("corporation", "")
    
    return features

# todo in case of model with card names
def calculate_features_names(corporation, cards):
    """Calculate features from corporation and cards."""
    cards_data, _ = load_data()

    selected_set = set()
    for n in cards:
        selected_set.add(str(n).strip().lower())

    features = {}

    features["corporation"] = str(corporation).strip().lower()

    for card in cards_data:
        col_name = str(card.get("name", "")).strip().lower()
        if col_name == "":
            continue

        if col_name in selected_set:
            features[col_name] = 1
        else:
            features[col_name] = 0

    return features


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
        features = calculate_features(request.corporation, request.cards)

        feature_dict = {}

        #tags
        for k in TAG_KEYS:
            feature_dict[k] = [features.get(k,0)]

        #rest
        feature_dict["totalVp"] = [features.get("totalVp", 0)]
        feature_dict["totalPrice"] = [features.get("totalPrice", 0)]
        feature_dict["corporation"] = [features.get("corporation", "")]

        df = pd.DataFrame(feature_dict)

        prediction = model_pipeline.predict(df)[0]
        
        return {
            "predicted_points": float(prediction),
            "corporation": request.corporation,
            "num_cards": len(request.cards),
            "features": feature_dict
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


@app.get("/model/info")
async def model_info():
    """Get model information and metrics if available."""
    info = {
        "model_loaded": model_pipeline is not None,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists()
    }
    
    # Try to load model info if available (could be stored separately)
    # For now, return basic info
    if model_pipeline is not None:
        info["model_type"] = type(model_pipeline.named_steps['model']).__name__
        info["has_preprocessing"] = 'prep' in model_pipeline.named_steps
    
    return info

