# Terraforming Mars VP Predictor Web App

A web application for predicting Victory Points (VP) in Terraforming Mars based on selected corporation and cards.

## Features

- **VP Prediction**: Select a corporation and cards to get a predicted VP score
- **Model Management**: Train and retrain the machine learning model
- **Modern UI**: Beautiful, responsive web interface
- **RESTful API**: FastAPI backend with comprehensive endpoints

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the training data:
   - `datasets/games.csv` - Training data
   - `cards/cards.json` - Card definitions
   - `cards/corporations.json` - Corporation definitions

3. Run the server:
```bash
python web/run_server.py
```

Or with uvicorn directly:
```bash
uvicorn web.backend:app --reload --host 0.0.0.0 --port 8000
```

4. Open your browser and navigate to:
```
http://localhost:8000
```

## API Endpoints

### GET `/`
Serves the frontend HTML page.

### GET `/cards`
Returns all available cards.

### GET `/corporations`
Returns all available corporations.

### POST `/predict`
Predicts VP based on corporation and cards.

**Request Body:**
```json
{
  "corporation": "Credicor",
  "cards": ["STANDARD TECHNOLOGY", "IMMIGRANT CITY"]
}
```

**Response:**
```json
{
  "predicted_points": 105.5,
  "corporation": "Credicor",
  "num_cards": 2,
  "features": {...}
}
```

### POST `/train`
Trains a new model and saves it.

**Request Body:**
```json
{
  "model_type": "RandomForest",
  "n_estimators": 600,
  "random_state": 42
}
```

**Response:**
```json
{
  "message": "Model trained and saved successfully",
  "train_score": 0.95,
  "test_score": 0.87,
  "model_type": "RandomForest",
  "n_samples_train": 800,
  "n_samples_test": 200
}
```

### GET `/model/status`
Checks if a model is loaded and available.

## Model Management

The model is saved to `web/model.pkl` after training. On server startup, it automatically loads the saved model if available.

To retrain the model:
1. Use the "Retrain Model" button in the web interface, or
2. Send a POST request to `/train` endpoint

## Project Structure

```
web/
├── backend.py          # FastAPI backend server
├── index.html          # Frontend web interface
├── run_server.py       # Server startup script
├── model.pkl          # Saved model (created after training)
└── README.md          # This file
```

## Notes

- The model uses RandomForest by default, but can be extended to support other model types
- The training data should be in CSV format with columns matching the expected features
- Card and corporation names are case-insensitive in the lookup



