# Quick Start Guide

## Running the Web App

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   python web/run_server.py
   ```
   
   Or with uvicorn directly:
   ```bash
   uvicorn web.backend:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Train the model (first time only):**
   ```bash
   python scripts/train_model.py
   ```
   This creates the model file that the web app uses.

4. **Open your browser:**
   Navigate to `http://localhost:8000`

5. **Make predictions:**
   - Select a corporation from the dropdown
   - Select cards by checking the boxes (use search to find cards quickly)
   - Click "Predict VP" to get the predicted victory points

## Features

- **Card Selection**: Search and select multiple cards
- **Corporation Selection**: Choose from available corporations
- **VP Prediction**: Get predicted total points based on your selections
- **Model Management**: Train model locally via script, then deploy
- **Beautiful UI**: Modern, responsive design

## API Endpoints

- `GET /` - Frontend web interface
- `GET /cards` - List all available cards
- `GET /corporations` - List all available corporations
- `POST /predict` - Predict VP (requires corporation and cards)
- `GET /model/status` - Check if model is loaded

## Troubleshooting

- **"Model not loaded" error**: Run `python scripts/train_model.py` to create the model file
- **Import errors**: Make sure all dependencies are installed (`pip install -r requirements.txt`)
- **CSV not found**: Ensure `datasets/games.csv` exists (run `scripts/tm_card_score.py` if needed)

