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

3. **Open your browser:**
   Navigate to `http://localhost:8000`

4. **Train the model (first time only):**
   - Click the "Retrain Model" button in the web interface
   - Wait for training to complete (this may take a minute)
   - The model will be saved automatically

5. **Make predictions:**
   - Select a corporation from the dropdown
   - Select cards by checking the boxes (use search to find cards quickly)
   - Click "Predict VP" to get the predicted victory points

## Features

- **Card Selection**: Search and select multiple cards
- **Corporation Selection**: Choose from available corporations
- **VP Prediction**: Get predicted total points based on your selections
- **Model Management**: Retrain the model with updated data or different parameters
- **Beautiful UI**: Modern, responsive design

## API Endpoints

- `GET /` - Frontend web interface
- `GET /cards` - List all available cards
- `GET /corporations` - List all available corporations
- `POST /predict` - Predict VP (requires corporation and cards)
- `POST /train` - Train/retrain the model
- `GET /model/status` - Check if model is loaded

## Troubleshooting

- **"Model not loaded" error**: Train the model first using the "Retrain Model" button
- **Import errors**: Make sure all dependencies are installed (`pip install -r requirements.txt`)
- **CSV not found**: Ensure `datasets/games.csv` exists (run `scripts/tm_card_score.py` if needed)

