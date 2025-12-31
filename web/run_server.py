#!/usr/bin/env python3
"""
Script to run the FastAPI server for Terraforming Mars VP Predictor.

Usage:
    python web/run_server.py
    
Or directly with uvicorn:
    uvicorn web.backend:app --reload --host 0.0.0.0 --port 8000
"""
import uvicorn
import sys
from pathlib import Path

# Add parent directory to path so we can import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    import os
    # Get port from environment variable (for cloud platforms) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    # Run the server
    uvicorn.run(
        "web.backend:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("ENV") != "production"
    )

